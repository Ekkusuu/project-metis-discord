"""
Validation loss of the base model and of LoRA checkpoints on the held-out personas, with the same encoding as
training. Answers "does this checkpoint beat the untouched model?", which the training log can't: it only
starts evaluating after 100 steps.

    python training/eval_loss.py                                # base model
    python training/eval_loss.py training/out/metis-lora/checkpoint-100 ...
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_unsloth import DEFAULT_MODEL, DEFAULT_VAL, encode, load_samples  # noqa: E402


def mean_loss(model, tokenizer, samples, max_seq_length: int = 2048) -> float:
    import torch

    total, count = 0.0, 0
    model.eval()
    for sample in samples:
        encoded = encode(tokenizer, sample, max_seq_length)
        if encoded is None:
            continue
        ids = torch.tensor([encoded["input_ids"]], device=model.device)
        labels = torch.tensor([encoded["labels"]], device=model.device)
        with torch.no_grad():
            total += model(input_ids=ids, labels=labels).loss.item()
        count += 1
    return total / max(count, 1)


def main() -> None:
    from unsloth import FastLanguageModel

    samples = load_samples(DEFAULT_VAL)
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=DEFAULT_MODEL, max_seq_length=2048,
                                                         load_in_4bit=False)
    FastLanguageModel.for_inference(model)
    print(f"base {DEFAULT_MODEL}: val loss {mean_loss(model, tokenizer, samples):.4f}", flush=True)
    for checkpoint in sys.argv[1:]:
        from peft import PeftModel

        tuned = PeftModel.from_pretrained(model, checkpoint)
        print(f"{checkpoint}: val loss {mean_loss(tuned, tokenizer, samples):.4f}", flush=True)
        model = tuned.unload()  # back to the base weights for the next checkpoint


if __name__ == "__main__":
    main()
