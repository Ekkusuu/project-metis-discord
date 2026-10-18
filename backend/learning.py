"""
Continuous learning: turn conversation into lasting facts about people.

Runs on the memory worker every few messages (and when a conversation is reset
or compacted). Designed to work with small local models (2B-8B):

- One pass per speaker, showing only that person's own messages (each with the
  line before it as context). Asking a small model to attribute facts across a
  whole multi-person transcript mixes people up.
- A worked example as a prior chat turn instead of a long list of rules.
- Every fact must be grounded: its key words have to appear in what that
  person actually wrote, which drops invented or misattributed facts.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from backend.llama_engine import chat_completion
from backend.people import get_people

_SYSTEM = (
    "You help an AI companion remember people. Given the messages one person wrote, list the lasting facts "
    "they reveal about themselves, one short phrase per line, without their name. Only what they actually said. "
    "Skip greetings, moods and small talk. If there is nothing worth remembering, write NONE."
)

_EXAMPLE_IN = """Messages by Robin:
(context) Lena: anyone up for pizza later?
Robin: can't, i'm vegetarian and that place only does meat lol
Robin: also my exam in organic chemistry is tomorrow, wish me luck
(context) Metis: Good luck! You've got this.
Robin: thanks! after that i'm finally adopting a puppy, gonna call him Biscuit

Facts about Robin:"""

_EXAMPLE_OUT = """is vegetarian
studies organic chemistry and has an exam coming up
is adopting a puppy named Biscuit"""

# Words that carry no evidence on their own when checking that a fact is grounded.
_STOPWORDS = set(
    "the and but with for from that this have has had are was were been being their they them about very really "
    "just also into your you named name likes like loves enjoys hates dislikes plays playing wants going gonna "
    "would could should will some lots".split()
)
# One-off states that aren't worth remembering.
_TRANSIENT = re.compile(
    r"^(is|was|feels?|felt|seems?)\s+(so\s+|very\s+|really\s+|a bit\s+)?"
    r"(tired|sleepy|bored|hungry|sad|happy|sick|busy|fine|ok|okay|good|bad|annoyed|excited|stressed|back|here|online|away)\b",
    re.IGNORECASE,
)


def _speaker_name(m: Dict[str, Any]) -> Optional[str]:
    if m.get("name"):
        return m["name"]
    match = re.match(r"^\[([^\]]+)\]", str(m.get("content", "")))
    return match.group(1) if match else None


def _text(m: Dict[str, Any]) -> str:
    """Message text without the '[Name]:' prefix."""
    return re.sub(r"^\[[^\]]+\]:?\s*", "", str(m.get("content", ""))).strip()


def _speakers(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """name -> person id for everyone who spoke (name-only ids for transcripts without ids)."""
    speakers: Dict[str, str] = {}
    for m in messages:
        if m.get("role") != "user":
            continue
        name = _speaker_name(m)
        if not name:
            continue
        if m.get("user_id") is not None:
            speakers[name] = str(m["user_id"])
        else:
            speakers.setdefault(name, get_people().resolve(name) or f"name:{name.lower()}")
    return speakers


def person_view(messages: List[Dict[str, Any]], name: str) -> Tuple[str, str]:
    """(prompt view of `name`'s messages with one line of context each, their own text lowercased)."""
    lines: List[str] = []
    own: List[str] = []
    for i, m in enumerate(messages):
        if m.get("role") != "user" or _speaker_name(m) != name:
            continue
        if i > 0:
            prev = messages[i - 1]
            prev_name = "Metis" if prev.get("role") == "assistant" else _speaker_name(prev)
            if prev_name and prev_name != name:
                lines.append(f"(context) {prev_name}: {_text(prev)[:300]}")
        text = _text(m)
        lines.append(f"{name}: {text}")
        own.append(text.lower())
    return "\n".join(lines), " ".join(own)


def is_grounded(fact: str, own_text: str) -> bool:
    """True if enough of the fact's key words appear in what the person wrote."""
    words = [w for w in re.findall(r"[a-z0-9']+", fact.lower()) if len(w) >= 4 and w not in _STOPWORDS]
    if not words:
        return False
    hits = sum(1 for w in words if w[:5] in own_text)
    return hits >= max(1, len(words) // 2)


def _clean_fact(fact: str, name: str) -> str:
    fact = fact.strip().lstrip("-•*0123456789.) ").strip().strip('"').rstrip(".")
    fact = re.sub(rf"^(?:user\s+)?(?:\[)?{re.escape(name)}(?:\])?(?:'s)?\s+", "", fact, flags=re.IGNORECASE)
    return fact


def extract_facts(messages: List[Dict[str, Any]], name: str) -> List[str]:
    """Ask the model for facts about one speaker and keep the grounded ones."""
    view, own = person_view(messages, name)
    if not view or len(own) < 12:
        return []
    response = chat_completion(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _EXAMPLE_IN},
            {"role": "assistant", "content": _EXAMPLE_OUT},
            {"role": "user", "content": f"Messages by {name}:\n{view}\n\nFacts about {name}:"},
        ],
        temperature=0.1,
        top_p=0.9,
        max_tokens=220,
    )
    facts = []
    for line in response.splitlines():
        fact = _clean_fact(line, name)
        if not fact or fact.upper() == "NONE" or len(fact) > 160 or _TRANSIENT.match(fact):
            continue
        if is_grounded(fact, own):
            facts.append(fact)
    return facts


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


def learn(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extract facts from `messages` and store them per person. Returns {"facts": n, "notes": 0}."""
    people = get_people()
    totals = {"facts": 0, "notes": 0}
    touched: set = set()

    for chunk in _chunks(messages):
        for name, pid in _speakers(chunk).items():
            try:
                facts = extract_facts(chunk, name)
            except Exception as e:
                print(f"Learning failed for {name}: {e}")
                continue
            added = people.add_facts(pid, facts, name=name)
            if added:
                touched.add(pid)
                totals["facts"] += added

    if touched:
        _reindex(people.export_markdown(touched))
        print(f"🧠 Learned {totals['facts']} fact(s) about {len(touched)} person(s)")
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
