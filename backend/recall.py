"""
Remembering is cued: a memory comes to mind when the moment is about it, not because it is stored.

Before, everything known about the people in the conversation was pasted into every prompt. On "Hello Metis, how
are you?" she answered "How's your game since the parents won't let you have a cat?". Now the cue (the latest
message and the user lines before it) is compared with each memory of the people there; the ones that are similar
enough, and haven't faded (backend/memory_store.py), are given to the model as background, at most three and
one of them an episode. A greeting recalls nothing. Memories brought to mind get stronger.

Asked directly ("what do you know about Okeida?", "do you remember...", the `remember` tool) she digs:
lower thresholds, faded memories too, episodes and facts, anyone including people who aren't here.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from backend.config import get_config
from backend.memory_store import MemoryStore, parse_time, retention, setting

_CUE_LINES = 3  # the latest message and the two user lines before it

_ASKS_TO_RECALL = re.compile(
    r"\b(remember|recall|remind me|know about|know anything|tell me about|told you|told me|did i (say|mention|tell)|"
    r"what do you know)\b", re.IGNORECASE)
_FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|i'm|i've)\b", re.IGNORECASE)


def user_lines(messages: Sequence[Dict[str, str]], count: int = _CUE_LINES) -> List[str]:
    """What people said, oldest first: the last `count` user messages without their '[Name]:' prefix."""
    lines = [re.sub(r"^\[[^\]]+\]:?\s*", "", m["content"]).strip() for m in messages if m["role"] == "user"]
    return [line for line in lines if line][-count:]


def asks_to_recall(text: str) -> bool:
    """Is the message asking her to think back or about what she knows of someone?"""
    return bool(_ASKS_TO_RECALL.search(text))


def mentioned_people(text: str, people: Dict[str, List[str]]) -> List[str]:
    """Ids of the known people (by name or other name; `people`: id -> names) the text names."""
    return [pid for pid, names in people.items()
            if any(len(n) >= 3 and re.search(rf"(?<![\w']){re.escape(n)}(?![\w'])", text, re.IGNORECASE) for n in names)]


def recall(store: MemoryStore, cues: Sequence[str], person_ids: Optional[Iterable[str]], *, explicit: bool = False,
           favoured: Iterable[str] = (), ignore: Iterable[str] = (), now: Optional[datetime] = None,
           mark: bool = True) -> List[Dict[str, Any]]:
    """
    The memories the cue calls to mind, best first. `person_ids`: whose memories (None = anyone). `explicit`:
    she was asked to think back; thresholds drop, faded memories count, and the memories of the `favoured`
    people (named in the question) are all candidates, however little their words overlap with it.
    """
    candidates = store.active(person_ids=person_ids)
    if not candidates or not cues:
        return []
    favoured = {str(p) for p in favoured}
    factor = setting("search_threshold_factor") if explicit else 1.0
    floor = setting("search_retention_floor" if explicit else "retention_floor")
    bonus = setting("importance_bonus")
    scored = []
    for memory, (sim, mode) in zip(candidates, store.similarities(cues, candidates, ignore)):
        strength = retention(memory, now)
        if strength < floor:
            continue
        threshold = setting("recall_threshold" if mode == "vec" else "recall_threshold_words") * factor
        nudge = bonus * (memory["importance"] - 5) / 5
        score = sim * strength + nudge
        if explicit and favoured & set(memory["person_ids"]):
            score = max(sim, 0.5) * strength + nudge  # asked about this person: whatever they remember of them
        elif sim <= 0 or (sim if explicit else sim * strength) + nudge < threshold:
            # A passing cue must beat the threshold with the memory's strength; asked, only the similarity counts.
            continue
        scored.append({**memory, "sim": round(sim, 3), "score": round(score, 3), "mode": mode})
    scored.sort(key=lambda m: -m["score"])

    limit = int(setting("recall_max")) + (2 if explicit else 0)
    episodes_allowed = 2 if explicit else 1
    chosen: List[Dict[str, Any]] = []
    episodes = 0
    for memory in scored:
        if len(chosen) >= limit:
            break
        if memory["kind"] == "episode":
            if episodes >= episodes_allowed:
                continue
            episodes += 1
        chosen.append(memory)
    if mark and chosen:
        store.mark_recalled([m["id"] for m in chosen])  # the testing effect: remembering strengthens
    return chosen


def age_text(created: str, now: Optional[datetime] = None) -> str:
    days = max(0, ((now or datetime.now(timezone.utc)) - parse_time(created)).days)
    if days == 0:
        return "earlier today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    return f"{days // 30} months ago"


def format_memories(memories: List[Dict[str, Any]], names: Dict[str, str], now: Optional[datetime] = None) -> str:
    """One line each: '- Ekkusu: plays League of Legends' / '- (3 days ago) Ekkusu and Okeida planned ...'."""
    lines = []
    for m in memories:
        if m["kind"] == "episode":
            lines.append(f"- ({age_text(m['created'], now)}) {m['text']}")
        else:
            who = ", ".join(names.get(p, p) for p in m["person_ids"])
            lines.append(f"- {who}: {m['text']}")
    return "\n".join(lines)


def recall_block(store: MemoryStore, people: Any, messages: Sequence[Dict[str, str]], participant_ids: List[str],
                 present_names: Sequence[str] = (), bot_names: Iterable[str] = ("Metis",)) -> str:
    """
    The memory section of the prompt for the latest message ('' when nothing comes to mind). `participant_ids`:
    recent speakers, the last one to speak first. People in the voice channel who haven't spoken count too.
    """
    cues = user_lines(messages)
    if not cues:
        return ""
    ids = list(dict.fromkeys([*participant_ids, *filter(None, (people.resolve(n) for n in present_names))]))
    names = people.names_of(ids)
    everyone = people.names_of(people.people)
    explicit = asks_to_recall(cues[-1])
    favoured: List[str] = []
    if explicit:
        favoured = mentioned_people(cues[-1], everyone)
        if participant_ids and _FIRST_PERSON.search(cues[-1]):
            favoured.append(participant_ids[0])  # "what do you know about me?"
    ignore = list(bot_names) if explicit else [*bot_names, *(n for ns in names.values() for n in ns)]
    memories = recall(store, cues, None if explicit else ids, explicit=explicit, favoured=favoured, ignore=ignore)
    if not memories:
        return ""
    label = {pid: ns[0] for pid, ns in everyone.items()}
    print(f"🧠 Brought to mind{' (asked)' if explicit else ''}: "
          + "; ".join(f"{m['text'][:60]!r} {m['mode']} {m['sim']:.2f}" for m in memories))
    body = format_memories(memories, label)
    if explicit:
        return ("What you remember (use only this when asked about them, and say so if you don't know "
                "something):\n" + body)
    return ("Something you remember that this brought to mind (background: use it only if it fits what they're "
            "saying right now, and never bring it up on your own):\n" + body)


def think_back(store: MemoryStore, people: Any, query: str, context: str = "",
               bot_names: Iterable[str] = ("Metis",)) -> str:
    """
    What she remembers that bears on `query` (the `remember` tool): facts and episodes of anyone, absent people
    too, with the lower thresholds of being asked. `context`: what people said just before, one line each.
    """
    everyone = people.names_of(people.people)
    memories = recall(store, [*context.splitlines()[-2:], query], None, explicit=True,
                      favoured=mentioned_people(query, everyone), ignore=bot_names)
    if not memories:
        return ""
    print("🧠 Thought back: " + "; ".join(f"{m['text'][:60]!r} {m['mode']} {m['sim']:.2f}" for m in memories))
    return ("What you remember (about your friends; use only this when asked about them):\n"
            + format_memories(memories, {pid: ns[0] for pid, ns in everyone.items()}))


def bot_names() -> List[str]:
    return list(dict.fromkeys(["Metis", *(get_config().get("chat", {}).get("bot_names") or [])]))
