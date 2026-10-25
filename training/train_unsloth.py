"""
Step 3: fine-tune Qwen3.5-4B (the heretic weights, see README) on the samples from build_samples.py, with a
16-bit LoRA. Unsloth advises against 4-bit QLoRA for Qwen3.5 (larger quantization error than usual).

The loss is on Metis's answer only: the prompt (persona, history, people memory, guidance) is masked with -100.
That is what teaches "read the system prompt and be that person" rather than "memorise a personality": the same
weights see 20+ different personas, each defined entirely by the system message.

    pip install "unsloth[cu126] @ git+https://github.com/unslothai/unsloth.git" trl peft accelerate bitsandbytes
    python training/train_unsloth.py --check
    python training/train_unsloth.py --data training/data/samples.jsonl --val training/data/samples.val.jsonl
    python training/train_unsloth.py --merge      # merged 16-bit weights, ready for llama.cpp

`--check` needs only transformers: it verifies the chat template, the prompt/target split and the lengths without
loading the model. Run it after rebuilding the dataset.

Defaults fit a 12 GB card (~10 GB with nothing else on the GPU: close games and stop Metis): bf16 LoRA r=16,
2048 tokens, batch 1 x 8 accumulation, 3 epochs. The adapter lands in
training/out/metis-lora; `--merge` writes training/out/metis-merged-16bit.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
# build_samples.py writes samples*.jsonl; render_prompts.mjs turns them into the runtime's exact text.
DEFAULT_DATA = HERE / "data" / "samples.rendered.jsonl"
DEFAULT_VAL = HERE / "data" / "samples.val.rendered.jsonl"
# external_data.py (When2Call tool decisions + general chat), rendered the same way.
DEFAULT_EXTRA = HERE / "data" / "external.rendered.jsonl"
DEFAULT_OUT = HERE / "out" / "metis-lora"
DEFAULT_MERGED = HERE / "out" / "metis-merged-16bit"
# Uncensored first, then fine-tuned: training afterwards repairs most of what uncensoring costs, while running
# heretic after training would edit the weights just trained.
DEFAULT_MODEL = "tvall43/Qwen3.5-4B-heretic"


def load_samples(path: Path, tasks: Optional[set[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            if tasks and sample.get("task") not in tasks:
                continue
            samples.append(sample)
            if limit and len(samples) >= limit:
                break
    if not samples:
        raise SystemExit(f"no samples in {path}")
    return samples


def encode(tokenizer: Any, sample: Dict[str, Any], max_seq_length: int) -> Optional[Dict[str, List[int]]]:
    """
    One training example: full chat tokens, with the prompt masked.

    Thinking is disabled the same way the runtime disables it (config model.thinking: false), so the model is
    trained on the exact shape it will see: no reasoning block before the answer. prompt_ids must be a prefix of
    full_ids; `--check` verifies that before any GPU time is spent.
    """
    if "prompt_text" in sample:
        # Rendered by render_prompts.mjs with the runtime's chat wrapper (node-llama-cpp), which is what the model
        # really sees: the HF template refuses the reply guidance's second system message.
        prompt_ids = tokenizer(sample["prompt_text"], add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(sample["answer_text"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + len(answer_ids) > max_seq_length:
            return None
        return {"input_ids": prompt_ids + answer_ids, "labels": [-100] * len(prompt_ids) + answer_ids}
    messages = sample["messages"]
    answer = sample["answer"]
    template_args = {"tokenize": True, "enable_thinking": False}
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **template_args)
    full_ids = tokenizer.apply_chat_template([*messages, {"role": "assistant", "content": answer}],
                                             add_generation_prompt=False, **template_args)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise SystemExit("chat template is not prefix-stable: the prompt is not the start of the full conversation")
    if len(full_ids) > max_seq_length:
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if all(label == -100 for label in labels):
        return None
    return {"input_ids": full_ids, "labels": labels}


def personas_of(samples: List[Dict[str, Any]]) -> set[str]:
    return {(sample.get("meta") or {}).get("persona", "") for sample in samples} - {""}


def check_leakage(train_samples: List[Dict[str, Any]], val_samples: List[Dict[str, Any]]) -> None:
    """A val persona in the training file would make the held-out evaluation meaningless."""
    leaked = sorted(personas_of(train_samples) & personas_of(val_samples))
    if leaked:
        raise SystemExit(f"val personas in the training data: {', '.join(leaked)}; rebuild with --split train")


def report(samples: List[Dict[str, Any]]) -> None:
    print(f"{len(samples)} examples")
    print("  tasks:    " + ", ".join(f"{name}: {count}" for name, count in Counter(
        sample["task"] for sample in samples).most_common()))
    print("  personas: " + ", ".join(f"{name}: {count}" for name, count in Counter(
        (sample.get("meta") or {}).get("persona", "?") for sample in samples).most_common()))
    print("  splits:   " + ", ".join(f"{name}: {count}" for name, count in Counter(
        (sample.get("meta") or {}).get("split", "train") for sample in samples).most_common()))


def check(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    loaded: List[Tuple[Path, List[Dict[str, Any]]]] = []
    for path, label in ((*((p, "train") for p in args.data), (args.val, "val"))):
        if not path.exists():
            print(f"{label}: {path} missing, skipped")
            continue
        loaded.append((path, load_samples(path, args.tasks, args.limit)))
    check_leakage(loaded[0][1], loaded[1][1] if len(loaded) > 1 else [])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for path, samples in loaded:
        print(f"\n== {path}")
        report(samples)
        lengths, skipped = [], 0
        for sample in samples:
            encoded = encode(tokenizer, sample, args.max_seq_length)
            if encoded is None:
                skipped += 1
                continue
            lengths.append(len(encoded["input_ids"]))
        lengths.sort()
        if lengths:
            print(f"  tokens:   min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}"
                  f" (max_seq_length {args.max_seq_length})")
        print(f"  skipped:  {skipped} too long for max_seq_length")
        first = next((sample for sample in samples if encode(tokenizer, sample, args.max_seq_length)), None)
        if first:
            encoded = encode(tokenizer, first, args.max_seq_length)
            prompt_length = len(encoded["input_ids"]) - sum(1 for label in encoded["labels"] if label != -100)
            prompt = tokenizer.decode(encoded["input_ids"][:prompt_length])
            target = tokenizer.decode(encoded["input_ids"][prompt_length:])
            print(f"\n  -- {first['id']} --")
            print(f"  last prompt tokens: ...{prompt[-400:]!r}")
            print(f"  target: {target!r}")
        print("  chat template is prefix-stable")


def train(args: argparse.Namespace) -> None:
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from unsloth import FastLanguageModel

    train_samples = [s for path in args.data for s in load_samples(path, args.tasks, args.limit)]
    val_samples = (load_samples(Path(args.val), args.tasks, args.limit)
                   if args.val and Path(args.val).exists() else [])
    check_leakage(train_samples, val_samples)
    train_samples = balance(train_samples, args.max_per_task, args.seed)
    report(train_samples)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=args.four_bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    train_rows = []
    skipped = 0
    for sample in train_samples:
        encoded = encode(tokenizer, sample, args.max_seq_length)
        if encoded is None:
            skipped += 1
        else:
            train_rows.append(encoded)
    if skipped:
        print(f"skipped {skipped} examples longer than {args.max_seq_length} tokens")
    if not train_rows:
        raise SystemExit("no trainable examples; raise --max-seq-length")

    eval_rows = []
    for sample in val_samples:
        encoded = encode(tokenizer, sample, args.max_seq_length)
        if encoded is not None:
            eval_rows.append(encoded)
    if val_samples:
        print(f"validation: {len(eval_rows)} of {len(val_samples)} examples")

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100,
                                      pad_to_multiple_of=8, return_tensors="pt")
    bf16 = torch.cuda.is_bf16_supported()
    strategy = "steps" if eval_rows else "no"
    training_args = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=bf16,
        fp16=not bf16,
        optim="adamw_8bit",
        logging_steps=args.log_every,
        eval_strategy=strategy,
        eval_steps=args.eval_every if eval_rows else None,
        # Saved when evaluated, so the best checkpoint on the held-out personas is the one kept: with few distinct
        # conversations (augmented), the last step is not always the best.
        save_strategy=strategy if eval_rows else "epoch",
        save_steps=args.eval_every if eval_rows else None,
        load_best_model_at_end=bool(eval_rows),
        metric_for_best_model="eval_loss" if eval_rows else None,
        save_total_limit=2,
        # NEFTune: noise on the input embeddings while training; it helps small instruction datasets generalise.
        neftune_noise_alpha=args.neftune or None,
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows) if eval_rows else None,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print(f"adapter saved to {args.out}")

    if args.merge:
        merged = Path(args.merged)
        print(f"merging into 16-bit weights at {merged} ...")
        model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
        print(f"done: {merged}\nconvert with llama.cpp: python convert_hf_to_gguf.py {merged} --outfile "
              f"{merged.with_suffix('.gguf')} --outtype q8_0")


def balance(samples: List[Dict[str, Any]], max_per_task: int, seed: int) -> List[Dict[str, Any]]:
    """
    At most `max_per_task` examples of each task (a fixed random sample). Augmentation made far more addressee
    examples than replies; unbalanced, the reply style (what matters most) would get the least training.
    """
    import random

    if not max_per_task:
        return samples
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        by_task.setdefault(sample["task"], []).append(sample)
    rng = random.Random(seed)
    kept = []
    for task, items in by_task.items():
        kept += items if len(items) <= max_per_task else rng.sample(items, max_per_task)
    rng.shuffle(kept)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+",
                        default=[DEFAULT_DATA, *([DEFAULT_EXTRA] if DEFAULT_EXTRA.exists() else [])])
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo or local folder of the base model")
    parser.add_argument("--tasks", default=None, help="comma-separated task filter (reply,planner,addressee,facts)")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="only the first N examples (smoke test)")
    # One pass is already several over each conversation: augment.py makes variants of every one.
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-per-task", type=int, default=2000, help="balance the tasks (0: keep everything)")
    parser.add_argument("--neftune", type=float, default=5.0, help="NEFTune noise alpha (0: off)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--4bit", dest="four_bit", action="store_true",
                        help="QLoRA on a 4-bit base (less VRAM; Unsloth advises against it for Qwen3.5)")
    parser.add_argument("--merge", action="store_true", help="also write merged 16-bit weights for llama.cpp")
    parser.add_argument("--check", action="store_true", help="validate tokenization only; needs no Unsloth/GPU")
    args = parser.parse_args()

    args.tasks = {task.strip() for task in args.tasks.split(",")} if args.tasks else None
    if args.check:
        check(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
