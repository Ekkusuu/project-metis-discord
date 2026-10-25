"""
Step 2 of the fine-tuning data: turn annotated conversations into SFT examples.

Every example is rendered with Metis's own prompt functions, not a copy of them, so the model trains on exactly
what it sees at runtime:

- reply:     the full chat prompt (persona, people memory, search results, reply guidance) -> her next line
- planner:   planner_messages() -> the JSON plan (standalone rewrite, action, query, recent)
- addressee: addressee_messages() -> who the line was for ("Metis", "another person in the chat", ...)
- facts:     the fact-extraction prompt from learning.py -> labelled candidates for one speaker

Personas are data, not baked in. Every conversation carries the exact system prompt it was written for
(`meta.persona_prompt`, from training/personas.py), and that prompt is what the reply/planner examples use. The
model therefore learns "be whoever the system prompt describes", not "be the current casual persona".

One JSON object per line:

    {"task": "reply", "id": "conv_00000:reply:2", "messages": [...], "answer": "...",
     "meta": {"persona": "shy_bookworm", "split": "train", ...}}

`messages` ends with the user turn; `answer` is the assistant target. training/train_unsloth.py appends it and
masks the prompt, so the loss is only on what Metis should say.

    python training/build_samples.py
    python training/build_samples.py --tasks reply,planner --mode both
    python training/build_samples.py --split val --out training/data/samples.val.jsonl
    python training/build_samples.py --conversation conv_00000 --limit 20 --preview
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import re
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

CONVERSATIONS = HERE / "data" / "conversations"
DEFAULT_OUT = HERE / "data" / "samples.jsonl"

TASKS = ("reply", "planner", "addressee", "facts")

# The tag the TTS voice understands when a line has no mood. "neutral" is always allowed.
_EMOTION_FALLBACK = "neutral"


# ---------------------------------------------------------------------------
# Runtime-shaped conversation
# ---------------------------------------------------------------------------
class Entry:
    """One line of a conversation, in the shape Conversation.prompt()/turns() use."""

    __slots__ = ("role", "name", "text", "mood", "found", "line", "uid", "raw")

    def __init__(self, role: str, text: str, name: str = "", mood: str = "", found: str = "",
                 line: int = 0, uid: Optional[int] = None, raw: Optional[Dict[str, Any]] = None) -> None:
        self.role = role          # "user" | "assistant"
        self.name = name          # speaker name for user lines; "Metis" for assistant lines
        self.text = text
        self.mood = mood
        self.found = found
        self.line = line          # index in conversation["lines"]
        self.uid = uid
        self.raw = raw or {}

    @property
    def content(self) -> str:
        return f"[{self.name}]: {self.text}" if self.role == "user" else self.text


def _entries(conversation: Dict[str, Any]) -> List[Entry]:
    """The call as runtime messages: user lines labelled, Metis lines optionally carrying her mood tag."""
    people = list(conversation["meta"]["people"])
    uids = {name: 1000 + i for i, name in enumerate(people)}
    entries: List[Entry] = []
    for i, line in enumerate(conversation.get("lines") or []):
        if line.get("speaker") == "Metis":
            text = str(line.get("text") or "").strip()
            mood = str(line.get("mood") or _EMOTION_FALLBACK).lower()
            entries.append(Entry("assistant", text, "Metis", mood=mood, found=str(line.get("found") or ""),
                                 line=i, raw=line))
        else:
            name = str(line.get("speaker") or "")
            entries.append(Entry("user", str(line.get("text") or "").strip(), name,
                                 line=i, uid=uids.get(name), raw=line))
    return entries


def _visible(entries: Sequence[Entry], voice: bool) -> List[Entry]:
    """
    What Conversation.prompt() shows the model. In voice, a bare "yeah"/"mm-hmm" between people is noise and is
    dropped, except right after Metis spoke or as the newest line.
    """
    if not voice:
        return list(entries)
    from backend.turn_taking import is_acknowledgement

    kept: List[Entry] = []
    for i, entry in enumerate(entries):
        if entry.role != "user" or i == len(entries) - 1:
            kept.append(entry)
        elif (not is_acknowledgement(entry.text)
              or (kept and kept[-1].role == "assistant")):
            kept.append(entry)
    return kept


def _assistant_text(entry: Entry, mood_tags: bool) -> str:
    if mood_tags and entry.mood:
        return f"[{entry.mood}] {entry.text}"
    return entry.text


def _system_prompt(conversation: Dict[str, Any], voice: bool) -> str:
    """
    The system message this conversation was written for. New conversations store the full prompt in meta, so
    any persona works; older ones resolve the built-in personality by name.
    """
    from backend.prompt_loader import resolve_prompt

    meta = conversation.get("meta") or {}
    key = "persona_prompt" if voice else "text_persona_prompt"
    prompt = str(meta.get(key) or "").strip()
    if prompt:
        return prompt
    persona = str(meta.get("persona") or "casual")
    return resolve_prompt(f"@{'voice_chat' if voice else 'text_chat'}.{persona}")


def _messages(entries: Sequence[Entry], system: str, voice: bool, mood_tags: bool) -> List[Dict[str, str]]:
    """The exact message list the chat pipeline builds for a voice/text conversation."""
    out = [{"role": "system", "content": system}]
    for entry in _visible(entries, voice):
        if entry.role == "assistant":
            out.append({"role": "assistant", "content": _assistant_text(entry, mood_tags)})
        else:
            out.append({"role": "user", "content": entry.content})
    return out


# ---------------------------------------------------------------------------
# Tools (the planner's menu, exactly as decide_tools() builds it)
# ---------------------------------------------------------------------------
def _decidable_tools() -> Dict[str, Any]:
    from backend.tool_manager import _first_string_param, _knowledge_always, get_tool_registry

    enabled = get_tool_registry().get_all_enabled()
    return {
        name: tool for name, tool in enabled.items()
        if name != "fetch_webpage" and not (name == "knowledge_search" and _knowledge_always())
        and _first_string_param(tool)
    }


def _plan_json(standalone: str, action: str, query: str = "", recent: bool = False) -> str:
    return json.dumps({"standalone": standalone, "action": action, "query": query or "", "recent": bool(recent)},
                      ensure_ascii=False)


# ---------------------------------------------------------------------------
# People memory (what Metis knew before a given line)
# ---------------------------------------------------------------------------
_FULL_KIND = {
    "who they are": "who they are (job, studies, home, family, pets)",
    "what they like or regularly do": "what they like or regularly do",
    "a plan": "a plan",
    "an opinion about what's being talked about": "an opinion about what's being talked about",
    "opinion about what's being talked about": "an opinion about what's being talked about",
    "something happening right now": "something happening right now",
    "a question or request": "a question or request",
}


def _kind(value: str) -> Optional[str]:
    from backend.learning import _KEEP_KINDS, _OTHER_KINDS

    known = {*_KEEP_KINDS, *_OTHER_KINDS}
    value = str(value or "").strip()
    if value in known:
        return value
    return _FULL_KIND.get(value)


def _fact_candidates(conversation: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    The labelled candidates per speaker, made usable by the runtime: a real quote, a kind the schema knows, and
    the line that revealed it. The teacher occasionally invents a quote; those would be dropped by
    extract_facts() anyway, so they must not teach the model to invent evidence.
    """
    from backend.learning import is_grounded

    lines = conversation.get("lines") or []
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name, facts in (conversation.get("facts") or {}).items():
        own = " ".join(str(line.get("text") or "").lower() for line in lines
                       if line.get("speaker") == name)
        clean: List[Dict[str, Any]] = []
        for fact in facts or []:
            quote = str(fact.get("quote") or "").strip()
            kind = _kind(fact.get("kind"))
            about = str(fact.get("about") or "")
            try:
                importance = int(fact.get("importance") or 0)
            except (TypeError, ValueError):
                importance = 0
            text = str(fact.get("fact") or "").strip().rstrip(".")
            if (not quote or not text or kind is None or about not in ("themselves", "someone else", "Metis")
                    or not 1 <= importance <= 10 or not is_grounded(quote, own)):
                continue
            revealed_at = _revealing_line(lines, name, quote)
            if revealed_at is None:
                continue
            clean.append({"quote": quote[:160], "about": about, "kind": kind,
                          "importance": importance, "fact": text[:120], "_line": revealed_at})
        out[name] = clean
    return out


def _revealing_line(lines: Sequence[Dict[str, Any]], name: str, quote: str) -> Optional[int]:
    """Index of the line that contains the quote (exact first, then learning.py's keyword grounding)."""
    from backend.learning import is_grounded

    lowered = quote.lower()
    for i, line in enumerate(lines):
        if line.get("speaker") != name:
            continue
        text = str(line.get("text") or "").lower()
        if lowered in text or is_grounded(quote, text):
            return i
    return None


def _known_facts(conversation: Dict[str, Any], candidates: Dict[str, List[Dict[str, Any]]],
                 before_line: int) -> Dict[str, List[str]]:
    """
    Facts the people store would hold before `before_line`: only what learning keeps (about themselves, a
    lasting kind, importance >= 6). A fact is known after the line that revealed it, never before it.
    """
    from backend.learning import _KEEP_KINDS, _MIN_IMPORTANCE, _clean_fact

    known: Dict[str, List[str]] = {}
    for name, facts in candidates.items():
        for fact in facts:
            if fact["_line"] >= before_line:
                continue
            if fact["about"] != "themselves" or fact["kind"] not in _KEEP_KINDS:
                continue
            if fact["importance"] < _MIN_IMPORTANCE:
                continue
            text = _clean_fact(fact["fact"], name)
            if text and text not in known.setdefault(name, []):
                known[name].append(text)
    return known


def _people_context(entries: Sequence[Entry], line: int, voice: bool,
                    known: Dict[str, List[str]]) -> str:
    """people_context(req), with a throwaway people store holding what was learned before this line."""
    import backend.people as people_module
    from backend.people import PeopleStore
    from backend.routers.chat import ChatRequest, Participant, people_context

    store = PeopleStore(Path(tempfile.gettempdir()) / "metis_sample_people.json")
    store.people = {}
    names = [entry.name for entry in entries[:line + 1] if entry.role == "user"]
    # Most recent speaker first, like Conversation.participants().
    ordered = list(dict.fromkeys(reversed(names)))
    participants = []
    for name in ordered:
        pid = f"sample-{name.lower()}"
        participants.append(Participant(id=pid, name=name))
        facts = known.get(name, [])
        store.people[pid] = {"id": pid, "name": name, "aliases": [], "facts": [
            {"id": str(i), "text": text} for i, text in enumerate(facts)
        ]}
    present = list(dict.fromkeys(entry.name for entry in entries if entry.role == "user")) if voice else []
    request = ChatRequest(messages=[], participants=participants, present=present, voice_mode=voice)

    previous = people_module._store
    people_module._store = store
    try:
        return people_context(request)
    finally:
        people_module._store = previous


# ---------------------------------------------------------------------------
# Search context and guidance
# ---------------------------------------------------------------------------
_SITE = re.compile(r"^(?P<site>[a-z0-9][a-z0-9.-]*\.[a-z]{2,})(?:\s*[,:]\s*|\s+)(?P<answer>.+)$", re.IGNORECASE)


def _search_context(found: str) -> Tuple[str, bool, str]:
    """(context_text, searched_web, found_answer) for a web look-up, in research.format_brief's shape."""
    found = str(found or "").strip()
    if not found:
        return ("[web_search]\nYou looked this up but couldn't find a clear answer. Say so; don't guess.",
                True, "")
    match = _SITE.match(found)
    if match:
        site, answer = match.group("site"), match.group("answer").strip()
    else:
        site, answer = "the web", found
    return f"[web_search]\nYou looked this up ({site}). What you found: {answer}", True, answer


@contextlib.contextmanager
def _frozen_now(iso: str):
    """A stable date/time in build_prompt(), so rebuilding the dataset gives byte-identical output."""
    import backend.routers.chat as chat

    frozen = datetime.datetime.fromisoformat(iso)

    class _Now(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen

    original = chat.datetime
    chat.datetime = types.SimpleNamespace(datetime=_Now)
    try:
        yield
    finally:
        chat.datetime = original


def _mode_tag(voice: bool) -> str:
    return "voice" if voice else "text"


def _reply_example(entries: Sequence[Entry], index: int, conversation: Dict[str, Any], voice: bool,
                   mood_tags: bool, candidates: Dict[str, List[Dict[str, Any]]], now: str) -> Optional[Dict[str, Any]]:
    """The prompt and target for Metis's line at `index`."""
    from backend.routers.chat import build_prompt, remember_lookup, reply_guidance
    import backend.routers.chat as chat

    reply = entries[index]
    trigger = entries[index - 1] if index else None
    if trigger is None or trigger.role != "user":
        return None

    # The prompt stops at the line she answers; her own line is the target, not part of the input.
    history = entries[:index]
    system = _system_prompt(conversation, voice)
    messages = _messages(history, system, voice, mood_tags)
    known = _known_facts(conversation, candidates, trigger.line)
    people = _people_context(history, len(history) - 1, voice, known)

    plan = _plan_for_trigger(trigger)
    context_text, searched_web, found_answer = "", False, ""
    if plan and plan.get("action") == "web_search":
        context_text, searched_web, found_answer = _search_context(reply.found)

    # Earlier lookups are in the prompt for a few minutes ("only bring it up if they ask again").
    conversation_id = f"sample:{conversation.get('meta', {}).get('seed', 0)}"
    chat._recent_lookups.pop(conversation_id, None)
    for prior in entries[: index - 1]:
        if prior.role == "user" and prior.line + 1 < len(entries):
            next_line = entries[prior.line + 1]
            if next_line.role == "assistant" and next_line.found:
                prior_plan = _plan_for_trigger(prior)
                if prior_plan and prior_plan.get("action") == "web_search":
                    remember_lookup(prior.text[:150], next_line.found, conversation_id)

    chiming_in = trigger.role == "user" and str(trigger.raw.get("for") or "") == "another person in the chat"
    guidance = reply_guidance(messages, voice, bool(context_text), searched_web, found_answer,
                              chiming_in=chiming_in)
    with _frozen_now(now):
        final = build_prompt(messages, context_text, people, guidance, conversation_id)

    answer = _assistant_text(reply, mood_tags)
    meta = conversation.get("meta") or {}
    return {
        "task": "reply",
        "id": f"{conversation['meta']['seed']}:reply:{_mode_tag(voice)}:{index}",
        "messages": final,
        "answer": answer,
        "meta": {"conversation": meta.get("source", ""), "line": index,
                 "persona": meta.get("persona", ""), "split": meta.get("split", "train"),
                 "voice": voice, "chiming_in": chiming_in, "web_search": searched_web},
    }


def _plan_for_trigger(trigger: Entry) -> Optional[Dict[str, Any]]:
    """The plan the model recorded for the line Metis is answering (synthesised as 'none' for chime-ins)."""
    plan = trigger.raw.get("plan")
    if isinstance(plan, dict):
        return plan
    if trigger.role == "user":
        # Not addressed to Metis: she is joining in, so nothing is looked up.
        return {"action": "none", "standalone": trigger.text, "query": "", "recent": False}
    return None


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _planner_examples(conversation: Dict[str, Any], entries: Sequence[Entry], voice: bool, mood_tags: bool,
                      decidable: Dict[str, Any], context_count: int) -> Iterable[Dict[str, Any]]:
    from backend.turn_taking import is_acknowledgement
    from backend.tool_manager import planner_messages

    system = _system_prompt(conversation, voice)
    last_query = ""
    for i, entry in enumerate(entries):
        if entry.role != "user":
            continue
        plan = _plan_for_trigger(entry)
        answered = i + 1 < len(entries) and entries[i + 1].role == "assistant"
        if not answered or plan is None:
            if plan and plan.get("action") == "web_search":
                last_query = str(plan.get("query") or last_query)
            continue
        if len(entry.text) < 3 or is_acknowledgement(entry.text):
            if plan.get("action") == "web_search":
                last_query = str(plan.get("query") or last_query)
            continue

        messages = _messages(entries[: i + 1], system, voice, mood_tags)
        prompt = planner_messages(messages, messages[-1]["content"], decidable, last_query, context_count)
        standalone = str(plan.get("standalone") or entry.text).strip()
        answer = _plan_json(standalone, str(plan.get("action") or "none"),
                            str(plan.get("query") or ""), bool(plan.get("recent")))
        meta = conversation.get("meta") or {}
        yield {
            "task": "planner",
            "id": f"{conversation['meta']['seed']}:planner:{_mode_tag(voice)}:{i}",
            "messages": prompt,
            "answer": answer,
            "meta": {"conversation": meta.get("source", ""), "line": entry.line,
                     "persona": meta.get("persona", ""), "split": meta.get("split", "train"),
                     "voice": voice, "last_query": last_query},
        }
        if plan.get("action") == "web_search":
            last_query = str(plan.get("query") or "")


def _addressee_examples(conversation: Dict[str, Any], entries: Sequence[Entry], voice: bool,
                        mood_tags: bool) -> Iterable[Dict[str, Any]]:
    from backend.turn_taking import ADDRESSEE_CHOICES, Turn, addressee_messages

    turns: List[Turn] = []
    names_by_id: Dict[int, str] = {}
    for i, entry in enumerate(entries):
        if entry.role == "assistant":
            turns.append(Turn(None, "Metis", _assistant_text(entry, mood_tags), i * 5.0,
                              to=turns[-1].speaker_id if turns and not turns[-1].is_metis else None))
            continue
        uid = entry.uid if entry.uid is not None else 1000 + i
        names_by_id[uid] = entry.name
        turns.append(Turn(uid, entry.name, entry.text, i * 5.0))
        target = str(entry.raw.get("for") or "")
        if target not in ADDRESSEE_CHOICES:
            continue
        meta = conversation.get("meta") or {}
        yield {
            "task": "addressee",
            "id": f"{conversation['meta']['seed']}:addressee:{_mode_tag(voice)}:{i}",
            "messages": addressee_messages(turns, names_by_id),
            "answer": target,
            "meta": {"conversation": meta.get("source", ""), "line": entry.line,
                     "persona": meta.get("persona", ""), "split": meta.get("split", "train"),
                     "voice": voice},
        }


def _facts_examples(conversation: Dict[str, Any], entries: Sequence[Entry], voice: bool,
                    mood_tags: bool, candidates: Dict[str, List[Dict[str, Any]]]) -> Iterable[Dict[str, Any]]:
    from backend.learning import _FACT_EXAMPLES, _SYSTEM, person_view

    raw: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.role == "assistant":
            raw.append({"role": "assistant", "content": _assistant_text(entry, mood_tags)})
        else:
            raw.append({"role": "user", "name": entry.name, "user_id": entry.uid,
                        "content": f"[{entry.name}]: {entry.text}"})

    for name in conversation["meta"]["people"]:
        view, own = person_view(raw, name)
        if not view or len(own) < 12:
            continue
        target = [{key: value for key, value in fact.items() if not key.startswith("_")}
                  for fact in candidates.get(name, [])[:6]]
        answer = json.dumps({"candidates": target}, ensure_ascii=False)
        meta = conversation.get("meta") or {}
        yield {
            "task": "facts",
            "id": f"{conversation['meta']['seed']}:facts:{_mode_tag(voice)}:{name}",
            "messages": [{"role": "system", "content": _SYSTEM}, *_FACT_EXAMPLES,
                         {"role": "user", "content": f"Messages by {name}:\n{view}\n\nFacts about {name}:"}],
            "answer": answer,
            "meta": {"conversation": meta.get("source", ""), "person": name,
                     "persona": meta.get("persona", ""), "split": meta.get("split", "train"),
                     "voice": voice},
        }


# ---------------------------------------------------------------------------
# Validation and CLI
# ---------------------------------------------------------------------------
def _validate(sample: Dict[str, Any]) -> List[str]:
    issues = []
    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["no messages"]
    for message in messages:
        if message.get("role") not in ("system", "user", "assistant") or not isinstance(message.get("content"), str):
            issues.append("bad message")
    if messages[-1].get("role") != "user":
        issues.append("last message is not from the user")
    answer = sample.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        issues.append("empty answer")
    if sample["task"] in ("planner", "facts"):
        try:
            parsed = json.loads(answer)
            if sample["task"] == "planner":
                if not {"standalone", "action", "query", "recent"} <= set(parsed):
                    issues.append("planner answer is missing fields")
            elif not isinstance(parsed.get("candidates"), list):
                issues.append("facts answer has no candidates list")
        except (TypeError, json.JSONDecodeError):
            issues.append("structured answer is not JSON")
    return issues


def build(conversations: Sequence[Path], tasks: Sequence[str], mode: str, mood_tags: str = "auto",
          now: str = "2026-10-24T20:00:00", limit: Optional[int] = None,
          split: str = "all", persona: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from backend.routers.chat import voice_emotions
    from backend.config import get_config

    voices = ["voice", "text"] if mode == "both" else [mode]
    default_voice = voices[0] == "voice"
    if mood_tags == "auto":
        tags = default_voice and bool(voice_emotions())
    else:
        tags = mood_tags == "on" and default_voice

    decidable = _decidable_tools() if "planner" in tasks else {}
    context_count = int(get_config().get("tools", {}).get("context_messages", 6))
    samples: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {"conversations": 0, "skipped": 0, "tasks": {task: 0 for task in tasks},
                             "splits": {}, "personas": {}, "examples_by_split": {},
                             "examples_by_persona": {}, "rejected": [],
                             "voice": default_voice, "mood_tags": tags}
    for path in conversations:
        try:
            conversation = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            stats["rejected"].append({"file": path.name, "error": str(e)})
            continue
        conversation.setdefault("meta", {})
        conversation["meta"]["source"] = path.stem
        conversation["meta"].setdefault("split", "train")
        if split != "all" and conversation["meta"]["split"] != split:
            continue
        if persona and conversation["meta"].get("persona") != persona:
            continue
        meta = conversation["meta"]
        entries = _entries(conversation)
        candidates = _fact_candidates(conversation)
        made = 0
        for voice_mode in [v == "voice" for v in voices]:
            if "reply" in tasks:
                for i, entry in enumerate(entries):
                    if entry.role != "assistant":
                        continue
                    sample = _reply_example(entries, i, conversation, voice_mode, tags, candidates, now)
                    if sample:
                        samples.append(sample)
                        made += 1
            if "planner" in tasks:
                for sample in _planner_examples(conversation, entries, voice_mode, tags, decidable, context_count):
                    samples.append(sample)
                    made += 1
            if "addressee" in tasks:
                for sample in _addressee_examples(conversation, entries, voice_mode, tags):
                    samples.append(sample)
                    made += 1
            if "facts" in tasks:
                for sample in _facts_examples(conversation, entries, voice_mode, tags, candidates):
                    samples.append(sample)
                    made += 1
        if made:
            stats["conversations"] += 1
            stats["splits"][meta["split"]] = stats["splits"].get(meta["split"], 0) + 1
            stats["personas"][meta.get("persona", "")] = stats["personas"].get(meta.get("persona", ""), 0) + 1
        else:
            stats["skipped"] += 1
        if limit and len(samples) >= limit:
            break

    for sample in samples:
        issues = _validate(sample)
        if issues:
            stats["rejected"].append({"id": sample.get("id"), "issues": issues})
    samples = [sample for sample in samples if not _validate(sample)]
    for sample in samples:
        stats["tasks"][sample["task"]] += 1
        meta = sample.get("meta") or {}
        split_name = meta.get("split", "train")
        persona = meta.get("persona", "")
        stats["examples_by_split"][split_name] = stats["examples_by_split"].get(split_name, 0) + 1
        stats["examples_by_persona"][persona] = stats["examples_by_persona"].get(persona, 0) + 1
    return samples, stats


def _preview(sample: Dict[str, Any]) -> str:
    role = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT"}
    lines = [f"--- {sample['task']} {sample['id']} ---"]
    lines += [f"[{role[m['role']]}] {m['content']}" for m in sample["messages"]]
    lines.append(f"[TARGET] {sample['answer']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation", default=None, help="only this stem (e.g. conv_00000)")
    parser.add_argument("--tasks", default=",".join(TASKS), help=f"comma-separated: {', '.join(TASKS)}")
    parser.add_argument("--mode", choices=["voice", "text", "both"], default="voice")
    parser.add_argument("--mood-tags", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--persona", default=None, help="only conversations written for this persona")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many examples")
    parser.add_argument("--now", default="2026-10-24T20:00:00", help="fixed prompt date/time")
    parser.add_argument("--preview", action="store_true", help="print the first example of each task")
    args = parser.parse_args()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        parser.error(f"unknown task(s): {', '.join(unknown)}")

    paths = sorted(CONVERSATIONS.glob("conv_*.json"))
    if args.conversation:
        paths = [path for path in paths if path.stem == args.conversation]
        if not paths:
            parser.error(f"no conversation named {args.conversation}")
    if not paths:
        parser.error(f"no conversations in {CONVERSATIONS}")

    samples, stats = build(paths, tasks, args.mode, args.mood_tags, args.now, args.limit, args.split,
                           args.persona)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    stats_path = args.out.parent / f"{args.out.stem}.stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{len(samples)} examples from {stats['conversations']} conversation(s) -> {args.out}")
    print("  " + ", ".join(f"{task}: {count}" for task, count in stats["tasks"].items()))
    if stats["examples_by_split"]:
        print("  " + ", ".join(f"{name}: {count} examples"
                               for name, count in sorted(stats["examples_by_split"].items())))
    if stats["rejected"]:
        print(f"  rejected: {len(stats['rejected'])}")
        for item in stats["rejected"][:5]:
            print(f"    {item}")
    if args.preview:
        seen = set()
        for sample in samples:
            if sample["task"] not in seen:
                seen.add(sample["task"])
                print("\n" + _preview(sample))


if __name__ == "__main__":
    main()
