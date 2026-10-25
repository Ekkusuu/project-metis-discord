"""
Step 2b: outside data mixed into the fine-tune, in the same example format as build_samples.py.

- when2call (nvidia/When2Call, CC-BY-4.0): deciding when a tool call is right and writing its arguments, or asking
  for what's missing. Taken from the preference set's *chosen* answers: its SFT set has no tool calls at all, only
  "I need X" questions and capability refusals ("Apologies, but I'm unable to ... perform web searches"). Refusals
  are left out entirely: they would teach Metis to say she can't look things up.
- general (HuggingFaceH4/no_robots, human-written, CC-BY-NC-4.0, fine for a personal project): ordinary chat,
  brainstorming and questions, so the fine-tune keeps the model's general behaviour instead of narrowing it to the
  Metis prompts (the usual guard against a fine-tune making a model rigid).

Neither has a persona, so they don't pull her voice anywhere; they're kept to roughly 15% of the mix.

    python training/external_data.py  ->  training/data/external.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "external"
OUT = HERE / "data" / "external.jsonl"
WHEN2CALL = "https://huggingface.co/datasets/nvidia/When2Call/resolve/main/train/when2call_train_pref.jsonl"
NO_ROBOTS = "https://huggingface.co/datasets/HuggingFaceH4/no_robots/resolve/main/data/train-00000-of-00001.parquet"

TOOL_SYSTEM = ("You can use these tools. To call one, reply with <TOOLCALL>[{{\"name\": ..., \"arguments\": "
               "{{...}}}}]</TOOLCALL>. If a tool needs something the user didn't give, ask for it. If none fits, "
               "answer yourself.\n\nTools:\n{tools}")
GENERAL_CATEGORIES = {"Chat", "Brainstorm", "Open QA", "Generation", "Classify", "Closed QA"}


def _download(url: str, name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def when2call(n_calls: int, n_questions: int, rng: random.Random) -> list:
    calls, questions = [], []
    for line in _download(WHEN2CALL, "when2call_train_pref.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        chosen = (row.get("chosen_response") or {}).get("content", "")
        if not chosen or any(w in chosen.lower() for w in ("unable", "apolog", "sorry", "i can't", "cannot")):
            continue  # capability refusals: never
        tools = "\n".join(t if isinstance(t, str) else json.dumps(t) for t in row.get("tools") or [])
        sample = {"messages": [{"role": "system", "content": TOOL_SYSTEM.format(tools=tools)}, *row["messages"]],
                  "answer": chosen}
        (calls if "<TOOLCALL>" in chosen else questions).append(sample)
    picked = rng.sample(calls, min(n_calls, len(calls))) + rng.sample(questions, min(n_questions, len(questions)))
    return [{**s, "task": "when2call", "id": f"when2call:{i}", "meta": {"source": "nvidia/When2Call", "split": "train"}}
            for i, s in enumerate(picked)]


def general(n: int, max_chars: int, rng: random.Random) -> list:
    import pandas as pd

    frame = pd.read_parquet(io.BytesIO(_download(NO_ROBOTS, "no_robots_train.parquet").read_bytes()))
    rows = []
    for _, row in frame.iterrows():
        messages = list(row["messages"])
        if row.get("category") not in GENERAL_CATEGORIES or messages[-1]["role"] != "assistant":
            continue
        if sum(len(m["content"]) for m in messages) > max_chars:
            continue  # short exchanges: Metis's own prompts are short-reply
        rows.append({"messages": [{"role": m["role"], "content": m["content"]} for m in messages[:-1]],
                     "answer": messages[-1]["content"]})
    picked = rng.sample(rows, min(n, len(rows)))
    return [{**s, "task": "general", "id": f"general:{i}", "meta": {"source": "HuggingFaceH4/no_robots", "split": "train"}}
            for i, s in enumerate(picked)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--when2call-calls", type=int, default=700)
    parser.add_argument("--when2call-questions", type=int, default=200)
    parser.add_argument("--general", type=int, default=500)
    parser.add_argument("--general-max-chars", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    samples = when2call(args.when2call_calls, args.when2call_questions, rng) + general(
        args.general, args.general_max_chars, rng)
    with OUT.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    counts = {}
    for s in samples:
        counts[s["task"]] = counts.get(s["task"], 0) + 1
    print(f"{len(samples)} examples -> {OUT}: {counts}")


if __name__ == "__main__":
    main()
