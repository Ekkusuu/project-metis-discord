"""
Step 1b: stretch a small set of conversations further, without a teacher model.

Every variant keeps the teacher's labels and says which tasks it may be used for (meta.tasks), because some
changes would make some labels wrong:

- names:   the speakers get other names (in their lines, the labels and the facts). All tasks. The model learns
           patterns instead of "ekkusu asks, okeida jokes".
- asr:     people's lines get speech-to-text noise (lowercase, no punctuation, an "uh" or "like"), as real voice
           input arrives. Not for facts: those quote people's exact words.
- persona: a different personality in the system prompt. Only for the planner, the addressee check and facts,
           whose answers don't depend on who Metis is; never for her replies, which must sound like the persona
           they were written for.

Only training conversations are augmented; the held-out personas stay as they were written.

    python training/augment.py --copies 2     ->  training/data/augmented/
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gen_conversations import NAMES  # noqa: E402
from personas import for_split  # noqa: E402

CONVERSATIONS = HERE / "data" / "conversations"
OUT = HERE / "data" / "augmented"
EXTRA_NAMES = ["Alex", "sam_", "Noah", "eli", "Marta", "jun", "Petra", "Oskar", "lia", "Rafa", "theo", "Nadia",
               "kenji", "Ilinca", "Bogdan", "mia.x", "Cosmin", "ava", "Hugo", "Tess"]
FILLERS = ["uh", "um", "like", "so", "wait"]


def _swap_names(text: str, mapping: dict) -> str:
    for old, new in mapping.items():
        text = re.sub(rf"(?<![\w]){re.escape(old)}(?![\w])", new, text, flags=re.IGNORECASE)
    return text


def rename(conversation: dict, rng: random.Random) -> dict:
    people = conversation["meta"]["people"]
    pool = [n for n in NAMES + EXTRA_NAMES if n.lower() not in {p.lower() for p in people}]
    mapping = dict(zip(people, rng.sample(pool, len(people))))
    out = json.loads(json.dumps(conversation))
    for line in out["lines"]:
        line["speaker"] = mapping.get(line["speaker"], line["speaker"])
        for key in ("text", "found"):
            if line.get(key):
                line[key] = _swap_names(line[key], mapping)
        if isinstance(line.get("plan"), dict):
            for key in ("standalone", "query"):
                line["plan"][key] = _swap_names(line["plan"].get(key, ""), mapping)
    out["facts"] = {mapping.get(name, name): [{**c, "quote": _swap_names(c.get("quote", ""), mapping),
                                               "fact": _swap_names(c.get("fact", ""), mapping)} for c in facts]
                    for name, facts in (out.get("facts") or {}).items()}
    out["meta"]["people"] = [mapping[p] for p in people]
    return out


def asr_noise(text: str, rng: random.Random) -> str:
    """What speech-to-text makes of a sentence: no capitals, little punctuation, a filler now and then."""
    text = text.lower()
    text = re.sub(r"[,.!;:]", "", text) if rng.random() < 0.7 else text
    words = text.split()
    if words and rng.random() < 0.5:
        words.insert(rng.randrange(len(words)), rng.choice(FILLERS))
    return " ".join(words)


def noisy(conversation: dict, rng: random.Random) -> dict:
    out = json.loads(json.dumps(conversation))
    for line in out["lines"]:
        if line.get("speaker") != "Metis":
            line["text"] = asr_noise(line["text"], rng)
    return out


def with_persona(conversation: dict, rng: random.Random) -> dict:
    out = json.loads(json.dumps(conversation))
    others = [p for p in for_split("train") if p.name != out["meta"].get("persona")]
    persona = rng.choice(others)
    out["meta"].update(persona=persona.name, persona_prompt=persona.voice, text_persona_prompt=persona.text,
                       persona_brief=persona.brief)
    return out


VARIANTS = {
    "names": (rename, ["reply", "planner", "addressee", "facts"]),
    "asr": (noisy, ["reply", "planner", "addressee"]),
    "persona": (with_persona, ["planner", "addressee", "facts"]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", type=int, default=2, help="variants of each kind per conversation")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("conv_*.json"):
        old.unlink()
    rng = random.Random(args.seed)
    made = 0
    for path in sorted(CONVERSATIONS.glob("conv_*.json")):
        conversation = json.loads(path.read_text(encoding="utf-8"))
        if conversation.get("meta", {}).get("split", "train") != "train":
            continue
        for kind, (make, tasks) in VARIANTS.items():
            for copy in range(args.copies):
                variant = make(rename(conversation, rng) if kind != "names" else conversation, rng)
                variant["meta"]["augment"] = kind
                variant["meta"]["tasks"] = tasks
                (OUT / f"{path.stem}_{kind}{copy}.json").write_text(
                    json.dumps(variant, ensure_ascii=False, indent=1), encoding="utf-8")
                made += 1
    print(f"{made} variants -> {OUT}")


if __name__ == "__main__":
    main()
