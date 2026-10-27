"""
Continuous learning: turn conversation into lasting memories about people.

Runs on the memory worker every few messages (facts) and when a conversation ends (an episode summary, like
sleep consolidating the day). Designed to work with small local models (2B-8B):

- One pass per speaker, showing only that person's own messages (each with the
  line before it as context). Asking a small model to attribute facts across a
  whole multi-person transcript mixes people up.
- Like mem0: the model returns facts as JSON (`{"facts": [...]}`), taught by worked examples as earlier
  turns, including ones where the answer is empty (greetings, opinions about the topic of the moment,
  teasing Metis, reading out someone else's info). Lists of "passing remark" phrasings kept missing new ones.
- Every fact must be grounded: its quote has to be (most of) one of the speaker's own statements and its key
  words have to appear in what that person wrote, which drops invented or misattributed facts. Nothing
  about Metis herself, and nothing only she said, is stored.
- New facts are reconciled with the most similar stored ones (ADD / UPDATE / NONE, as in mem0): a repeat
  confirms the old fact (it lasts longer), a change supersedes it ("moved to Berlin" replaces "lives in
  Paris", which is kept but no longer recalled).
- When a conversation ends, one call turns it into one to three episodes (who was there, what happened).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.llama_engine import json_completion
from backend.memory_store import content_stems
from backend.people import get_people, is_duplicate
from backend.recall import bot_names

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
    "- NONE: a stored fact already says the same thing (give that fact's id)."
)


# On top of memory_store.STOPWORDS: words a fact is phrased with, that say nothing about what it is about.
_STOPWORDS = set("named name likes loves enjoys hates dislikes plays playing wants lots been being into".split())


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
        own.append(statements(text).lower())
    return "\n".join(lines), " ".join(own)


def statements(text: str) -> str:
    """
    The text without its questions: facts are grounded only in what someone states. Asked "Who's Mochi?" about
    Metis's cat, "has a cat named Mochi" was filed under the person asking.
    """
    return " ".join(s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s and not s.endswith("?"))


def own_sentences(messages: List[Dict[str, Any]], name: str) -> List[str]:
    """The sentences `name` wrote that aren't questions, one by one: a quote has to be one of these."""
    sentences: List[str] = []
    for m in messages:
        if m.get("role") == "user" and _speaker_name(m) == name:
            sentences += [s for s in re.split(r"(?<=[.!?])\s+|\n+", statements(_text(m))) if s.strip()]
    return sentences


def _evidence(text: str) -> set:
    return content_stems(text, extra_stopwords=_STOPWORDS)


def is_grounded(fact: str, own_text: str) -> bool:
    """
    True if enough of the fact's key words appear in what the person wrote. Words are compared as stems, and
    3-letter ones count: with 5-character prefixes and 4+ letter words "has a dog" had no key words at all.
    """
    words = _evidence(fact)
    if not words:
        return False
    return len(words & _evidence(own_text)) >= max(1, len(words) // 2)


def quote_supported(quote: str, sentences: List[str]) -> bool:
    """True if the quote is substantially one of the person's own sentences (not a mix of several, not Metis's)."""
    words = _evidence(quote)
    return bool(words) and any(len(words & _evidence(s)) / len(words) >= 0.6 for s in sentences)


def is_about_metis(fact: str) -> bool:
    return any(re.search(rf"(?<![\w']){re.escape(n)}(?![\w'])", fact, re.IGNORECASE) for n in bot_names())


def _clean_fact(fact: str, name: str) -> str:
    fact = fact.strip().lstrip("-•*0123456789.) ").strip().strip('"').rstrip(".")
    fact = re.sub(rf"^(?:user\s+)?(?:\[)?{re.escape(name)}(?:\])?(?:'s)?\s+", "", fact, flags=re.IGNORECASE)
    return fact


def extract_candidates(messages: List[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    """Ask the model for facts about one speaker; the grounded ones as {"fact", "quote", "importance"}."""
    view, own = person_view(messages, name)
    if not view or len(own) < 12:
        return []
    result = json_completion(
        [{"role": "system", "content": _SYSTEM}, *_FACT_EXAMPLES,
         {"role": "user", "content": f"Messages by {name}:\n{view}\n\nFacts about {name}:"}],
        _FACTS_SCHEMA, 0.1, 600, priority="background",
    )
    sentences = own_sentences(messages, name)
    facts = []
    for candidate in result.get("candidates") or []:
        if (candidate.get("about") != "themselves" or candidate.get("kind") not in _KEEP_KINDS
                or int(candidate.get("importance") or 0) < _MIN_IMPORTANCE):
            continue
        fact, quote = _clean_fact(str(candidate.get("fact", "")), name), str(candidate.get("quote", "")).strip()
        # Evidence: the quote must be what they said, and the fact must come from it. Nothing about Metis.
        if (fact and len(fact) <= 160 and not is_about_metis(fact) and quote_supported(quote, sentences)
                and is_grounded(fact, own)):
            facts.append({"fact": fact, "quote": quote, "importance": int(candidate["importance"])})
    return facts


def extract_facts(messages: List[Dict[str, Any]], name: str) -> List[str]:
    return [c["fact"] for c in extract_candidates(messages, name)]


def reconcile(person_id: str, facts: List[Dict[str, Any]]
              ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    (facts to add, {stored fact id: fact replacing it}, {stored fact id: fact repeating it}) after comparing new
    facts with the stored ones most like them (mem0's ADD / UPDATE / NONE). Ids are an enum of the real ones:
    a small model invents ids otherwise. Compared with only the last 15 stored facts, a paraphrase of an older
    one was added again.
    """
    stored = get_people().memories.nearest(person_id, [f["fact"] for f in facts], limit=8)
    if not facts or not stored:
        return facts, {}, {}
    # Nearly the same words: no model needed.
    same: Dict[str, Dict[str, Any]] = {}
    open_facts = []
    for fact in facts:
        twin = next((s for s in stored if is_duplicate(fact["fact"], [s["text"]])), None)
        if twin:
            same[twin["id"]] = fact
        else:
            open_facts.append(fact)
    if not open_facts:
        return [], {}, same
    ids = {str(i): s["id"] for i, s in enumerate(stored, 1)}
    schema = {
        "type": "object",
        "properties": {"decisions": {"type": "array", "minItems": len(open_facts), "maxItems": len(open_facts),
                                     "items": {
            "type": "object",
            "properties": {"event": {"enum": ["ADD", "UPDATE", "NONE"]}, "id": {"enum": ["", *ids]}},
            "required": ["event", "id"]}}},
        "required": ["decisions"],
    }
    listing = "\n".join(f"{i}. {s['text']}" for i, s in zip(ids, stored))
    new = "\n".join(f"- {f['fact']}" for f in open_facts)
    ask = "\n\nOne decision per new fact, in the same order (the stored fact's id for UPDATE and NONE)."
    result = json_completion(
        [{"role": "system", "content": _RECONCILE_SYSTEM},
         # A worked example: without it "moved to Berlin" was added next to "lives in Paris".
         {"role": "user", "content": "Stored facts:\n1. lives in Rome\n2. likes cats\n3. plays guitar\n\n"
                                     "New facts, in order:\n- moved to Madrid\n- loves cats\n- has a sister" + ask},
         {"role": "assistant", "content": json.dumps({"decisions": [
             {"event": "UPDATE", "id": "1"}, {"event": "NONE", "id": "2"}, {"event": "ADD", "id": ""}]})},
         {"role": "user", "content": f"Stored facts:\n{listing}\n\nNew facts, in order:\n{new}" + ask}],
        schema, 0.0, 200, priority="background",
    )
    to_add, updates = [], {}
    for fact, decision in zip(open_facts, result.get("decisions") or []):
        event, ref = decision.get("event"), decision.get("id")
        if event == "UPDATE" and ref in ids:
            updates[ids[ref]] = fact
        elif event == "NONE":
            if ref in ids:
                same[ids[ref]] = fact
        else:
            to_add.append(fact)
    return to_add, updates, same


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


def _session(session: str) -> str:
    return session or datetime.now(timezone.utc).strftime("%Y-%m-%d")


def learn(messages: List[Dict[str, Any]], session: str = "") -> Dict[str, int]:
    """Extract facts from `messages` and store them per person. Returns {"facts": n, "notes": 0}."""
    people = get_people()
    session = _session(session)
    totals = {"facts": 0, "notes": 0}
    touched: set = set()

    for chunk in _chunks(messages):
        for name, pid in _speakers(chunk).items():
            try:
                facts = extract_candidates(chunk, name)
            except Exception as e:
                print(f"Learning failed for {name}: {e}")
                continue
            try:
                facts, updates, same = reconcile(pid, facts)
            except Exception as e:
                print(f"Couldn't compare new facts about {name} with what's known ({e}); adding them")
                updates, same = {}, {}
            for fact_id, fact in same.items():
                if people.reinforce_fact(pid, fact_id, fact["importance"]):
                    print(f"🧠 Heard again about {name}: {fact['fact']}")
            for fact_id, fact in updates.items():
                if people.supersede_fact(pid, fact_id, fact["fact"], fact["importance"], fact["quote"], session):
                    print(f"🧠 Updated a fact about {name}: {fact['fact']}")
                    touched.add(pid)
                    totals["facts"] += 1
            for fact in facts:
                if people.add_fact(pid, fact["fact"], fact["importance"], fact["quote"], name=name, session=session):
                    touched.add(pid)
                    totals["facts"] += 1

    if touched:
        print(f"🧠 Learned {totals['facts']} fact(s) about {len(touched)} person(s)")
    return totals


# ---------------------------------------------------------------------------
# Episodes: what happened in a conversation that ended
# ---------------------------------------------------------------------------
_EPISODE_SYSTEM = (
    "You help Metis, a girl in a Discord group, remember conversations the way a friend does. Given a conversation "
    "that just ended, write one to three short summaries (one or two sentences each, third person, naming who "
    "was there) of what happened that a friend would still remember in a week: what they talked about, plans, "
    "feelings, news, how it turned out. Say only what the people said or did. Never record something only Metis "
    "said or made up. Greetings, small talk and banter that goes nowhere are not worth remembering: then return "
    "an empty list. Most important first."
)
_EPISODES_SCHEMA = {
    "type": "object",
    "properties": {"episodes": {"type": "array", "maxItems": 3, "items": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "maxLength": 240},
            # 1 = mundane, 10 = something a friend would never forget.
            "importance": {"enum": list(range(1, 11))},
        },
        "required": ["summary", "importance"],
    }}},
    "required": ["episodes"],
}


def _episode_example(transcript: str, episodes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"role": "user", "content": f"Conversation:\n{transcript}\n\nWhat happened:"},
            {"role": "assistant", "content": json.dumps({"episodes": episodes})}]


_EPISODE_EXAMPLES = [
    *_episode_example("Robin: ugh my chemistry exam is tomorrow and i haven't slept\n"
                      "Lena: you always do well, you've got this\n"
                      "Robin: thanks. after it i'm adopting a puppy, Biscuit\n"
                      "Metis: Good luck tomorrow! Biscuit is a great name.\n"
                      "Lena: send pictures!",
                      [{"summary": "Robin was stressed about an organic chemistry exam the next day and said they "
                                   "would adopt a puppy named Biscuit afterwards. Lena and Metis cheered them on.",
                        "importance": 7}]),
    *_episode_example("Sam: hi\nMetis: Hey Sam!\nSam: what's up\nMetis: Not much, you?\nSam: lol ok\nSam: brb", []),
    *_episode_example("Kai: aatrox is so bad now\nAna: his heal got nerfed, it's a joke\n"
                      "Metis: I watched a horror movie about him last night\nKai: lol what are you talking about\n"
                      "Ana: no idea", []),
]

_MIN_USER_MESSAGES = 3
_MIN_USER_CHARS = 60


def transcript(messages: List[Dict[str, Any]], max_chars: int = 7000) -> str:
    """'Name: text' lines of the conversation, Metis's included; the oldest lines go if it is too long."""
    lines = [f"{'Metis' if m.get('role') == 'assistant' else _speaker_name(m) or '?'}: {_text(m)}" for m in messages]
    while len(lines) > 1 and sum(len(line) + 1 for line in lines) > max_chars:
        lines.pop(0)
    return "\n".join(lines)


def episode_is_sound(summary: str, speakers: List[str], user_text: str) -> bool:
    """
    A summary has to name someone who was there and be made of what people said. Summaries that wandered off
    (or took Metis's inventions for events) are the memories that make no sense, so they aren't kept.
    """
    first_names = [re.split(r"\s*\(", n)[0] for n in speakers]
    if not any(re.search(rf"(?<![\w']){re.escape(n)}(?![\w'])", summary, re.IGNORECASE) for n in first_names if n):
        return False
    words = content_stems(summary, ignore=[*first_names, *bot_names()])
    return bool(words) and len(words & content_stems(user_text)) / len(words) >= 0.4


def consolidate(messages: List[Dict[str, Any]], session: str = "") -> int:
    """
    One call over a conversation that ended: store up to three episodes for everyone who spoke in it. Like sleep,
    this keeps what mattered and drops the rest (most conversations yield nothing). Returns how many were stored.
    """
    users = [m for m in messages if m.get("role") == "user" and m.get("content")]
    speakers = _speakers(messages)
    if (len(users) < _MIN_USER_MESSAGES or not speakers
            or sum(len(_text(m)) for m in users) < _MIN_USER_CHARS):
        return 0  # nothing happened worth a call
    result = json_completion(
        [{"role": "system", "content": _EPISODE_SYSTEM}, *_EPISODE_EXAMPLES,
         {"role": "user", "content": f"Conversation:\n{transcript(messages)}\n\nWhat happened:"}],
        _EPISODES_SCHEMA, 0.1, 500, priority="background",
    )
    people = get_people()
    ids = list(dict.fromkeys(speakers.values()))
    known = [e["text"] for e in people.memories.active("episode", ids)]
    user_text = " ".join(_text(m) for m in users)
    stored = 0
    for episode in result.get("episodes") or []:
        summary = str(episode.get("summary", "")).strip()
        if (not summary or not episode_is_sound(summary, list(speakers), user_text)
                or is_duplicate(summary, known)):
            continue
        people.add_episode(ids, summary, int(episode.get("importance") or 5), _session(session),
                           names={pid: name for name, pid in speakers.items()})
        known.append(summary)
        stored += 1
    if stored:
        print(f"🧠 Remembered {stored} episode(s) with {', '.join(speakers)}")
    return stored


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
