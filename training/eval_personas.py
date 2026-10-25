"""
Step 4: does the fine-tune actually become whoever the system prompt says?

The val personas in training/personas.py never appear in training. If replies for them are as in-character as for
the train personas, the model is reading the system prompt rather than reciting a memorised character. This asks
each persona the same questions through the app's own LLM service, so it tests whatever model is configured in
config.yaml (point model.path at the fine-tune's GGUF, restart, run this).

    python training/eval_personas.py                    # every held-out val persona
    python training/eval_personas.py --split train --personas cowboy,casual
    python training/eval_personas.py --dry-run          # print the prompts only

The app's LLM service must be running (it starts with `python metis.py`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from personas import BY_NAME, for_split  # noqa: E402

QUESTIONS = [
    "so what do you think of this call so far",
    "i had the worst day, cheer me up",
    "what are you doing after this",
]

# One speaker is enough: the point is the voice, not group dynamics.
SPEAKER = "Mira"


def render(persona_prompt: str, question: str, voice: bool = True) -> list[dict[str, str]]:
    """The exact prompt the chat pipeline builds for one user line, without people memory or tools."""
    from backend.routers.chat import build_prompt, reply_guidance

    messages = [{"role": "system", "content": persona_prompt},
                {"role": "user", "content": f"[{SPEAKER}]: {question}"}]
    guidance = reply_guidance(messages, voice, False, False, "", chiming_in=False)
    return build_prompt(messages, "", "", guidance, "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--personas", default=None, help="comma-separated names")
    parser.add_argument("--questions", default=None, help="comma-separated questions (default: built-in three)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true", help="print prompts, don't call the model")
    args = parser.parse_args()

    personas = for_split(args.split)
    if args.personas:
        names = [name.strip() for name in args.personas.split(",") if name.strip()]
        unknown = [name for name in names if name not in BY_NAME]
        if unknown:
            raise SystemExit(f"unknown persona(s): {', '.join(unknown)}")
        personas = [BY_NAME[name] for name in names]
    questions = [q.strip() for q in args.questions.split(",")] if args.questions else QUESTIONS

    if not args.dry_run:
        from backend.llama_engine import chat_completion, wait_until_ready

        if not wait_until_ready(timeout=5):
            raise SystemExit("the LLM service is not running (start the app, then retry)")

    for persona in personas:
        print(f"\n=== {persona.split}/{persona.name} ===")
        for question in questions:
            messages = render(persona.voice, question)
            if args.dry_run:
                print(f"\n--- {question}\nSYSTEM: {persona.voice}\n[{SPEAKER}]: {question}")
                continue
            reply = chat_completion(messages, temperature=args.temperature, top_p=args.top_p,
                                    max_tokens=args.max_tokens)
            print(f"\n[{SPEAKER}]: {question}\nMetis: {reply.strip()}")


if __name__ == "__main__":
    main()
