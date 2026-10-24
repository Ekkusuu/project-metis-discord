"""
Continuous learning: turn conversation into lasting facts about people.

Runs on the memory worker every few messages (and when a conversation is reset
or compacted). Designed to work with small local models (2B-8B):

- One pass per speaker, showing only that person's own messages (each with the
  line before it as context). Asking a small model to attribute facts across a
  whole multi-person transcript mixes people up.
- Like mem0: the model returns facts as JSON (`{"facts": [...]}`), taught by worked examples as earlier
  turns, including ones where the answer is empty (greetings, opinions about the topic of the moment,
  teasing Metis, reading out someone else's info). Lists of "passing remark" phrasings kept missing new ones.
- Every fact must be grounded: its key words have to appear in what that
  person actually wrote, which drops invented or misattributed facts.
- New facts are reconciled with what's stored (ADD / UPDATE / NONE, as in mem0), so "moved to Berlin"
  replaces "lives in Paris" instead of both being kept.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.llama_engine import json_completion
from backend.people import get_people

_SYSTEM = (
    "You help Metis, a girl in a Discord group, remember her friends. Given the messages one person wrote (lines "
    "marked (context) are other people, for context only), list the lasting facts that person reveals about "
    "themselves: things a friend would still remember next week (job or studies, where they live, family, "
    "pets, hobbies, games they play, what they like or dislike, plans). Use only what that person said about "
    "themselves. Not questions or requests, not opinions about the topic of the moment, not jokes or teasing, "
    "not things they say about other people or about Metis. Short phrases without their name. If there is "
    "nothing like that, return an empty list."
)

# Each candidate is labelled before it's written down: their exact words, who it's about and what kind of thing
# it is. Asked only for a list of facts, the 4B model turned "okeida trains voice models" (read out by someone
# else), "I'm alone" and the topic of the moment into facts about the speaker. A yes/no "lasting" label was too
# abstract for it (everything was "lasting"); picking a described kind works better.
_KEEP_KINDS = ["who they are (job, studies, home, family, pets)", "what they like or regularly do", "a plan"]
_OTHER_KINDS = ["an opinion about what's being talked about", "something happening right now", "a question or request"]
_FACTS_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "maxItems": 6, "items": {
        "type": "object",
        "properties": {
            "quote": {"type": "string", "maxLength": 160},
            "about": {"enum": ["themselves", "someone else", "Metis"]},
            "kind": {"enum": [*_KEEP_KINDS, *_OTHER_KINDS]},
            # Generative Agents' importance score: 1 = mundane, 10 = something a friend would never forget.
            "importance": {"enum": list(range(1, 11))},
            "fact": {"type": "string", "maxLength": 120},
        },
        "required": ["quote", "about", "kind", "importance", "fact"],
    }}},
    "required": ["candidates"],
}


_MIN_IMPORTANCE = 6
WHO, LIKES, PLAN = _KEEP_KINDS
OPINION, NOW, ASKS = _OTHER_KINDS


def _candidate(quote: str, about: str, kind: str, importance: int, fact: str) -> Dict[str, Any]:
    return {"quote": quote, "about": about, "kind": kind, "importance": importance, "fact": fact}


def _facts_example(lines: str, name: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"role": "user", "content": f"Messages by {name}:\n{lines}\n\nFacts about {name}:"},
            {"role": "assistant", "content": json.dumps({"candidates": candidates})}]


# Worked examples as earlier turns, with rejected candidates too (mem0 teaches "chatter isn't a fact" the same way).
_FACT_EXAMPLES = [
    *_facts_example("(context) Lena: anyone up for pizza later?\n"
                    "Robin: can't, i'm vegetarian and that place only does meat lol\n"
                    "Robin: metis what's the weather like tomorrow?\n"
                    "Robin: ugh i'm so tired today\n"
                    "Robin: my exam in organic chemistry is tomorrow, and after that i'm adopting a puppy, Biscuit",
                    "Robin", [
                        _candidate("i'm vegetarian", "themselves", WHO, 7, "is vegetarian"),
                        _candidate("metis what's the weather like tomorrow?", "themselves", ASKS, 1,
                                   "asks about the weather"),
                        _candidate("i'm so tired today", "themselves", NOW, 2, "is tired"),
                        _candidate("my exam in organic chemistry", "themselves", WHO, 7, "studies organic chemistry"),
                        _candidate("i'm adopting a puppy, Biscuit", "themselves", PLAN, 8,
                                   "is adopting a puppy named Biscuit")]),
    *_facts_example("Sam: hi\nSam: lol nice\nSam: what's the best champion right now?", "Sam", []),
    *_facts_example("(context) Ana: aatrox is so bad now\nKai: yeah he doesn't even heal, it's a joke\n"
                    "Kai: metis you totally like him though, admit it", "Kai", [
                        _candidate("he doesn't even heal, it's a joke", "themselves", OPINION, 2,
                                   "thinks Aatrox is bad"),
                        _candidate("metis you totally like him though", "Metis", OPINION, 1, "Metis likes Aatrox")]),
    *_facts_example("Jo: Lena trains voice models and owns a gaming laptop, by the way\n"
                    "Jo: and i main support, been playing since season 3", "Jo", [
                        _candidate("Lena trains voice models and owns a gaming laptop", "someone else", WHO, 5,
                                   "Lena trains voice models"),
                        _candidate("i main support, been playing since season 3", "themselves", LIKES, 6,
                                   "mains support and has played since season 3")]),
]

_RECONCILE_SYSTEM = (
    "You keep Metis's memory of a friend up to date. For each new fact, compare it with the stored facts:\n"
    "- ADD: it's new information.\n"
    "- UPDATE: it changes or corrects a stored fact (give that fact's id); the new fact replaces it.\n"
    "- NONE: a stored fact already says the same thing."
)


# Words that carry no evidence on their own when checking that a fact is grounded.
_STOPWORDS = set(
    "the and but with for from that this have has had are was were been being their they them about very really "
    "just also into your you named name likes like loves enjoys hates dislikes plays playing wants going gonna "
    "would could should will some lots".split()
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
    result = json_completion(
        [{"role": "system", "content": _SYSTEM}, *_FACT_EXAMPLES,
         {"role": "user", "content": f"Messages by {name}:\n{view}\n\nFacts about {name}:"}],
        _FACTS_SCHEMA, 0.1, 600, priority="background",
    )
    facts = []
    for candidate in result.get("candidates") or []:
        if (candidate.get("about") != "themselves" or candidate.get("kind") not in _KEEP_KINDS
                or int(candidate.get("importance") or 0) < _MIN_IMPORTANCE):
            continue
        fact = _clean_fact(str(candidate.get("fact", "")), name)
        # Evidence: the quote must be what they said, and the fact must come from it.
        if fact and len(fact) <= 160 and is_grounded(str(candidate.get("quote", "")), own) and is_grounded(fact, own):
            facts.append(fact)
    return facts


def reconcile(person_id: str, facts: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    (facts to add, {stored fact id: new text}) after comparing new facts with the person's stored ones (mem0's
    ADD / UPDATE / NONE). Ids are an enum of the real ones: a small model invents ids otherwise.
    """
    person = get_people().get(person_id)
    stored = (person or {}).get("facts", [])[-15:]
    if not facts or not stored:
        return facts, {}
    ids = {str(i): f["id"] for i, f in enumerate(stored, 1)}
    schema = {
        "type": "object",
        "properties": {"decisions": {"type": "array", "minItems": len(facts), "maxItems": len(facts), "items": {
            "type": "object",
            "properties": {"event": {"enum": ["ADD", "UPDATE", "NONE"]}, "id": {"enum": ["", *ids]}},
            "required": ["event", "id"]}}},
        "required": ["decisions"],
    }
    listing = "\n".join(f"{i}. {f['text']}" for i, f in zip(ids, stored))
    new = "\n".join(f"- {f}" for f in facts)
    ask = "\n\nOne decision per new fact, in the same order (id only for UPDATE)."
    result = json_completion(
        [{"role": "system", "content": _RECONCILE_SYSTEM},
         # A worked example: without it "moved to Berlin" was added next to "lives in Paris".
         {"role": "user", "content": "Stored facts:\n1. lives in Rome\n2. likes cats\n3. plays guitar\n\n"
                                     "New facts, in order:\n- moved to Madrid\n- loves cats\n- has a sister" + ask},
         {"role": "assistant", "content": json.dumps({"decisions": [
             {"event": "UPDATE", "id": "1"}, {"event": "NONE", "id": ""}, {"event": "ADD", "id": ""}]})},
         {"role": "user", "content": f"Stored facts:\n{listing}\n\nNew facts, in order:\n{new}" + ask}],
        schema, 0.0, 200, priority="background",
    )
    to_add, updates = [], {}
    for fact, decision in zip(facts, result.get("decisions") or []):
        event, ref = decision.get("event"), decision.get("id")
        if event == "UPDATE" and ref in ids:
            updates[ids[ref]] = fact
        elif event != "NONE":
            to_add.append(fact)
    return to_add, updates


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
            try:
                facts, updates = reconcile(pid, facts)
            except Exception as e:
                print(f"Couldn't compare new facts about {name} with what's known ({e}); adding them")
                updates = {}
            for fact_id, text in updates.items():
                if people.update_fact(pid, fact_id, text):
                    print(f"🧠 Updated a fact about {name}: {text}")
                    touched.add(pid)
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
