"""
Continuous learning: turn conversation into lasting facts about people.

Runs on the memory worker every few messages (and when a conversation is reset
or compacted). The model writes simple "Name | fact" lines (small local models
are far more reliable at this than at JSON), and only names that actually
spoke in the conversation are accepted, so facts always land on the right profile.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from backend.llama_engine import chat_completion
from backend.people import get_people

_SYSTEM = """You maintain the long-term memory of Metis, an AI companion who chats with people on Discord.
Go through the conversation message by message and write down every NEW, lasting fact about the people in it.

Worth remembering:
- Anything a person says or clearly shows about themselves, or that others say about them: job or school,
  where they live or are moving, pets, family and relationships, hobbies, games, music, food, plans and
  upcoming events, skills they are learning, likes AND dislikes, opinions, and how they like to be talked to.

Not worth remembering:
- Greetings, small talk, jokes, questions, one-off moods ("is tired right now"), and anything about Metis herself.

Output format (nothing else):
- One fact per line as: Name | fact
- The fact is a short phrase WITHOUT the person's name, e.g.
  Sava | has a cat named Pixel
  Sava | dislikes horror games
  Sava | is moving to Berlin next month
- Facts about a person always go under that person's name, never under NOTE.
- NOTE | text  is only for things that are not about one person (shared plans, server events, inside jokes).
- Skip facts already listed under "Already known". If something changed, write the new version.
- If nothing new was learned, output only: NONE"""

_LINE = re.compile(r"^\s*[-*•]?\s*\[?([^|\]\n]{1,60}?)\]?\s*\|\s*(.+?)\s*$")


def _transcript(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for m in messages:
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if m.get("role") == "assistant":
            lines.append(f"Metis: {content}")
        elif m.get("role") == "user":
            lines.append(content if content.startswith("[") else f"[{m.get('name') or 'User'}]: {content}")
    return "\n".join(lines)


def _chunks(messages: List[Dict[str, Any]], max_chars: int = 6000) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = [[]]
    size = 0
    for m in messages:
        length = len(str(m.get("content", "")))
        if chunks[-1] and size + length > max_chars:
            chunks.append([])
            size = 0
        chunks[-1].append(m)
        size += length
    return [c for c in chunks if c]


def _speakers(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """name -> person id for everyone who spoke (name-only ids for transcripts without ids)."""
    speakers: Dict[str, str] = {}
    for m in messages:
        if m.get("role") != "user":
            continue
        name = m.get("name")
        if not name:
            match = re.match(r"^\[([^\]]+)\]", str(m.get("content", "")))
            name = match.group(1) if match else None
        if not name:
            continue
        if m.get("user_id") is not None:
            speakers[name] = str(m["user_id"])
        else:
            speakers.setdefault(name, get_people().resolve(name) or f"name:{name.lower()}")
    return speakers


def _clean_fact(fact: str, name: str) -> str:
    fact = fact.strip().strip('"').rstrip(".")
    fact = re.sub(rf"^(?:user\s+)?(?:\[)?{re.escape(name)}(?:\])?(?:'s)?\s+", "", fact, flags=re.IGNORECASE)
    return fact


def parse_facts(response: str, speakers: Dict[str, str]) -> tuple:
    """Parse 'Name | fact' lines. Returns ({speaker name: [facts]}, [notes]); unknown names are dropped."""
    by_lower = {name.lower(): name for name in speakers}
    facts: Dict[str, List[str]] = {}
    notes: List[str] = []
    for line in response.splitlines():
        match = _LINE.match(line)
        if not match:
            continue
        who, what = match.group(1).strip(), match.group(2).strip().strip('"')
        if not what or what.upper() == "NONE":
            continue
        if who.upper() == "NOTE":
            notes.append(what)
            continue
        name = by_lower.get(who.lower())
        if name is None:
            # "Sava" vs "Sava (sava_x)": accept an unambiguous prefix match
            matches = [n for low, n in by_lower.items() if low.startswith(who.lower()) or who.lower().startswith(low)]
            name = matches[0] if len(matches) == 1 else None
        if name:
            facts.setdefault(name, []).append(_clean_fact(what, name))
    return facts, notes


def learn(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extract facts from `messages` and store them. Returns {"facts": n, "notes": n}."""
    people = get_people()
    totals = {"facts": 0, "notes": 0}
    touched: set = set()

    for chunk in _chunks(messages):
        speakers = _speakers(chunk)
        transcript = _transcript(chunk)
        if not speakers or not transcript:
            continue
        known = people.profile_block(speakers.values(), max_facts=40) or "(nothing yet)"
        try:
            response = chat_completion(
                [
                    {"role": "system", "content": _SYSTEM},
                    {
                        "role": "user",
                        "content": f"People in this conversation: {', '.join(sorted(speakers))}\n\n"
                        f"Already known:\n{known}\n\nConversation:\n{transcript}",
                    },
                ],
                temperature=0.2,
                top_p=0.9,
                max_tokens=700,
            )
        except Exception as e:
            print(f"Learning failed for one chunk: {e}")
            continue

        facts, notes = parse_facts(response, speakers)
        for name, items in facts.items():
            pid = speakers[name]
            added = people.add_facts(pid, items, name=name)
            if added:
                touched.add(pid)
                totals["facts"] += added
        totals["notes"] += people.add_notes(notes)

    if touched or totals["notes"]:
        written = people.export_markdown(touched)
        _reindex(written)
        print(f"🧠 Learned {totals['facts']} fact(s) about {len(touched)} person(s), {totals['notes']} note(s)")
    return totals


def _reindex(paths: List[Any]) -> None:
    try:
        from backend.rag_engine import reindex_path_if_tracked
        if paths:
            reindex_path_if_tracked(paths[0])
    except Exception as e:
        print(f"Warning: could not reindex learned memories: {e}")


def parse_legacy_transcript(text: str) -> List[Dict[str, Any]]:
    """Convert old temp_memory.txt lines ('User: [name]: ...' / 'AI: ...') into messages."""
    messages: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("User:"):
            messages.append({"role": "user", "content": line[5:].strip()})
        elif line.startswith("AI:"):
            messages.append({"role": "assistant", "content": line[3:].strip()})
    return messages

