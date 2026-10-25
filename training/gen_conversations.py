"""
Step 1 of the training data: annotated Discord voice-call conversations written by the teacher model.

Each conversation has Metis's replies plus labels for what her model decides at runtime: who each line is for,
the tool plan for lines meant for her, what she found when she looked something up, and the lasting facts each
person revealed. build_samples.py turns them into training examples in exactly the prompts Metis uses.

Conversations are written per persona, from training/personas.py, and store the persona's full system prompt.
Val-split personas are never used for training, only to check that the fine-tune adapts to an unseen persona.

    python training/gen_conversations.py --count 40 --workers 4
    python training/gen_conversations.py --split val --count 25 --workers 5
    python training/gen_conversations.py --personas cowboy,radio_host --count 4
    python training/gen_conversations.py --repair        # keep rejects whose only problem was name punctuation
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from personas import Persona, schedule  # noqa: E402
from teacher import MODEL, ask_json  # noqa: E402

OUT = HERE / "data" / "conversations"

# What the calls are about, weighted towards what Metis gets wrong most often in the logs.
CATEGORIES = [
    (4, "game questions she should look up (best picks, builds, counters, tier lists, patch changes) for games "
        "like League of Legends, Valorant, CS2, Minecraft, Elden Ring, Fortnite or Overwatch, with one or two "
        "follow-ups that change a detail ('what about for Graves?', 'this time for Emerald Plus', 'then do it')"),
    (2, "other facts she should look up: weather, sports results, prices, release dates, who plays someone in a "
        "movie, news; sometimes they name a website ('check on rotten tomatoes'), sometimes speech-to-text "
        "mangles it ('u.g' for u.gg)"),
    (4, "casual banter and teasing between friends and Metis, where she has her own opinions and jokes back"),
    (3, "hypotheticals and 'what would you do' questions for Metis ('would you rather', 'if you had a "
        "superpower', 'what would you do with a million dollars', 'which game would you delete forever'): she "
        "answers from her own perspective and nobody looks anything up"),
    (3, "questions about Metis herself (who she is, what she likes, what she's up to, her opinions): she "
        "answers as herself, a girl in the call, using only the life details in her personality"),
    (3, "two or three friends talking among themselves about their day, games or plans: most lines are not for "
        "Metis, some talk ABOUT her ('she's actually right', 'her voice cracks'), and she joins in now and then"),
    (2, "someone tells Metis about their life (job, studies, where they live, pets, family, hobbies, plans) "
        "while chatting, mixed with small talk and opinions about the topic of the moment"),
    (2, "corrections and changes of mind: 'no, I meant...', 'don't look that up, just tell me what you think', "
        "'never mind', and thanks or acknowledgements that need no lookup"),
    (1, "someone asks Metis to remember something said earlier in another conversation ('what did I tell you "
        "about my exam?')"),
]

NAMES = ["ekkusu", "okeida", "Sava", "Mira", "jayden", "Kai", "lena_", "Tomas", "rin", "Ana", "max99", "Dorin",
         "sophie", "Vlad", "yuki", "Ben", "zara", "Nico", "Iris", "Luca"]
EMOTIONS = ["neutral", "happy", "excited", "teasing", "affectionate", "sleepy", "annoyed", "sad", "scared"]
ADDRESSEES = ["Metis", "talking about Metis", "another person in the chat", "everyone in the chat",
              "nobody in particular"]

PROMPT = """You write training data for Metis, a voice companion in a Discord voice call. Write ONE realistic \
call transcript of 8 to 16 lines and label it. Output only JSON, no prose, no code fences.

Metis's personality (she talks like this): {persona}
People in the call besides Metis: {people}. Their lines come from speech-to-text: mostly lowercase, fillers \
like "uh", sometimes a misheard word.
What the call is about: {category}

Metis's lines: spoken words only, usually 1-2 short sentences, casual and natural like a friend in a call. She \
reacts and gives her own take, asks a question only when she's curious, talks about herself as "I", and never \
says she is an AI or an assistant. When she looked something up, she says the facts in her own words (never \
"based on the search results"). She never says she checked or will check something unless she looked it up \
for that line.

JSON format:
{{"lines": [
  {{"speaker": "<a person's name>", "text": "...", "for": "<one of: {addressees}>",
    "plan": null or {{"action": "none" | "web_search" | "remember", "standalone": "<the line rewritten to make \
sense on its own>", "query": "<2-8 search keywords, with site:domain.com if they named a site; empty unless \
web_search>", "recent": <true if about news or very recent events>}}}},
  {{"speaker": "Metis", "mood": "<one of: {emotions}>", "text": "...", "found": "<only if she looked something \
up for this reply: short notes of what she found, with the site, e.g. 'u.gg, patch 26.21: Thresh 52.1% win \
rate'; otherwise empty>"}}
],
 "facts": {{"<person>": [{{"quote": "<their exact words>", "about": "themselves" | "someone else" | "Metis", \
"kind": "who they are" | "what they like or regularly do" | "a plan" | "an opinion about what's being talked \
about" | "something happening right now" | "a question or request", "importance": <1-10>, "fact": "<short \
phrase without their name>"}}]}}}}

Labels:
- "for": who the line is meant for. "plan" is set for every line meant for Metis (or talking about her) and \
null otherwise.
- action web_search for facts about games, products, people, events, prices, weather or news, or when asked to \
look something up or check again; none for chat, jokes, opinions, hypotheticals and questions about Metis \
herself; remember only for "do you remember what I told you".
- When a line gets web_search, Metis's next line has "found" and uses it.
- "facts": every candidate fact a person's lines contain, labelled honestly (most chatter has none)."""


def _weighted(options):
    return random.choices([o for _, o in options], weights=[w for w, _ in options])[0]


def _canonical(speaker: str, people: list) -> str | None:
    """The person's name, tolerating the punctuation the teacher adds ("Tomas." / "okeida.." / "Metis,")."""
    name = str(speaker or "").strip()
    if name == "Metis":
        return name
    for person in people:
        if name == person:
            return person
    trimmed = name.rstrip(".,!?:;")
    for person in people:
        if trimmed.casefold() == person.rstrip(".,!?:;").casefold():
            return person
    return None


def normalize(data: dict, people: list) -> dict:
    """Canonical speaker names in lines and fact keys, keeping the teacher's punctuation out of the data."""
    for line in data.get("lines") or []:
        name = _canonical(line.get("speaker"), people)
        if name:
            line["speaker"] = name
    if isinstance(data.get("facts"), dict):
        data["facts"] = {(_canonical(name, people) or name): facts for name, facts in data["facts"].items()}
    return data


def problems(data: dict, people: list) -> list:
    """Why a conversation can't be used (empty list = fine). The teacher sometimes drops the labels or invents
    someone who isn't in the call."""
    issues = []
    for line in data.get("lines") or []:
        speaker = line.get("speaker")
        if speaker == "Metis":
            if line.get("mood") not in EMOTIONS:
                issues.append(f"Metis line without a known mood: {line.get('mood')!r}")
            if any(p in line.get("text", "").lower() for p in ("as an ai", "i'm an ai", "language model", "assistant")):
                issues.append("Metis talks like an AI")
        elif speaker not in people:
            issues.append(f"unknown speaker {speaker!r}")
        else:
            if line.get("for") not in ADDRESSEES:
                issues.append(f"line without a valid 'for': {line.get('for')!r}")
            plan = line.get("plan")
            if line.get("for") in ("Metis", "talking about Metis") and (
                    not isinstance(plan, dict) or plan.get("action") not in ("none", "web_search", "remember")):
                issues.append("line for Metis without a plan")
    if not any(line.get("speaker") == "Metis" for line in data.get("lines") or []):
        issues.append("Metis never speaks")
    return issues


def one_conversation(index: int, seed: int, persona: Persona) -> Path | None:
    rng = random.Random(seed)
    people = rng.sample(NAMES, rng.choice([1, 1, 2, 2, 3]))
    category = _weighted(CATEGORIES)
    prompt = PROMPT.format(persona=persona.brief, people=", ".join(people), category=category,
                           addressees=" | ".join(ADDRESSEES), emotions=" | ".join(EMOTIONS))
    data = ask_json(prompt)
    if isinstance(data, dict) and isinstance(data.get("lines"), list):
        data = normalize(data, people)
    issues = (["not a conversation"] if not isinstance(data, dict) or not isinstance(data.get("lines"), list)
              or len(data["lines"]) < 4 else problems(data, people))
    if issues:
        (OUT.parent / "rejected").mkdir(parents=True, exist_ok=True)
        (OUT.parent / "rejected" / f"conv_{index:05d}.json").write_text(
            json.dumps({"issues": issues, "persona": persona.name, "split": persona.split, "people": people,
                        "category": category, "data": data}, ensure_ascii=False, indent=1), encoding="utf-8")
        return None
    data["meta"] = {
        "persona": persona.name,
        "persona_prompt": persona.voice,
        "text_persona_prompt": persona.text,
        "persona_brief": persona.brief,
        "split": persona.split,
        "people": people,
        "category": category,
        "seed": seed,
        "generator": MODEL,
    }
    path = OUT / f"conv_{index:05d}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _next_index() -> int:
    """First unused conversation number, so train and val runs never overwrite each other."""
    used = [int(path.stem.split("_")[1]) for path in OUT.glob("conv_*.json") if path.stem.split("_")[1].isdigit()]
    return max(used, default=-1) + 1


def repair() -> None:
    """Re-check rejects for name punctuation (the teacher writes "Tomas."/"okeida.."), keeping the good ones."""
    from personas import BY_NAME

    rejected = OUT.parent / "rejected"
    repaired = 0
    for path in sorted(rejected.glob("conv_*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        persona_name, data = blob.get("persona"), blob.get("data")
        if persona_name not in BY_NAME or not isinstance(data, dict) or not isinstance(data.get("lines"), list):
            continue
        people = blob.get("people") or list(dict.fromkeys(
            _canonical(line.get("speaker"), NAMES) or line.get("speaker")
            for line in data["lines"] if line.get("speaker") not in (None, "Metis")))
        data = normalize(data, people)
        issues = problems(data, people)
        if issues:
            print(f"{path.name}: still unusable ({'; '.join(issues[:2])})")
            continue
        persona = BY_NAME[persona_name]
        index = int(path.stem.split("_")[1])
        target = OUT / f"conv_{index:05d}.json"
        if target.exists():
            target = OUT / f"conv_{_next_index():05d}.json"
        data["meta"] = {"persona": persona.name, "persona_prompt": persona.voice,
                        "text_persona_prompt": persona.text, "persona_brief": persona.brief,
                        "split": persona.split, "people": people, "category": blob.get("category", ""),
                        "seed": 1000 + index, "generator": MODEL}
        target.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        path.unlink(missing_ok=True)
        repaired += 1
        print(f"{path.name} -> {target.name} ({persona.name})")
    print(f"repaired {repaired} conversation(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--personas", default=None,
                        help="comma-separated persona names (default: the whole split); e.g. cowboy,radio_host")
    parser.add_argument("--start", type=int, default=None, help="first index (default: after existing files)")
    parser.add_argument("--seed", type=int, default=7, help="persona schedule seed")
    parser.add_argument("--repair", action="store_true", help="re-check rejects and keep the recoverable ones")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.repair:
        repair()
        return

    only = [name.strip() for name in args.personas.split(",") if name.strip()] if args.personas else None
    personas = schedule(args.split, args.count, args.seed, only)
    OUT.mkdir(parents=True, exist_ok=True)
    start = args.start if args.start is not None else _next_index()
    t0, made = time.time(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(one_conversation, start + i, 1000 + start + i, persona): (start + i, persona)
                for i, persona in enumerate(personas)}
        for job in as_completed(jobs):
            index, persona = jobs[job]
            try:
                path = job.result()
            except Exception as e:
                print(f"#{index} {persona.name} failed: {e}", flush=True)
                continue
            made += bool(path)
            print(f"#{index} {persona.split}/{persona.name} {'ok' if path else 'unusable'} "
                  f"({made} made, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
