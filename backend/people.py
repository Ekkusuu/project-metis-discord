"""
Who Metis knows: one profile per Discord user, keyed by their stable user id.

Display names change and collide; user ids don't. Every message (text or
voice) updates the speaker's profile, and facts learned from conversations
(see backend/learning.py) are attached to the right person here. Profiles are
also written out as Markdown under memory/long_term/people/ so the knowledge
base can retrieve facts about people who aren't in the current conversation.

People only known by name (e.g. learned from an old transcript without ids)
get a pseudo id "name:<lowercase name>" and are merged into the real profile
the first time someone with that name shows up on Discord.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from backend.config import MEMORY_DIR, get_config

PEOPLE_FILE = MEMORY_DIR / "people.json"
PROFILE_DIR = MEMORY_DIR / "long_term" / "people"
NOTES_FILE = MEMORY_DIR / "long_term" / "learned_notes.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def is_duplicate(fact: str, existing: Iterable[str], threshold: float = 0.82) -> bool:
    """True if `fact` says (nearly) the same thing as an existing fact."""
    a = _norm(fact)
    if not a:
        return True
    for other in existing:
        b = _norm(other)
        if a == b or (a in b and len(a) > 12) or SequenceMatcher(None, a, b).ratio() >= threshold:
            return True
    return False


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "person"


class PeopleStore:
    def __init__(self, path: Path = PEOPLE_FILE) -> None:
        self.path = path
        self._lock = threading.RLock()
        self.people: Dict[str, Dict[str, Any]] = {}
        self.notes: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.people = data.get("people", {})
            self.notes = data.get("notes", [])
        except Exception as e:
            print(f"Warning: could not read {self.path.name}: {e}")

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"people": self.people, "notes": self.notes}, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self.path)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def seen(self, user_id: int | str, name: str, aliases: Iterable[str] = ()) -> Dict[str, Any]:
        """Record that a user spoke. Returns their profile."""
        pid = str(user_id)
        with self._lock:
            person = self.people.get(pid)
            if person is None:
                person = {"id": pid, "name": name, "aliases": [], "first_seen": _now(), "last_seen": _now(),
                          "message_count": 0, "facts": []}
                self.people[pid] = person
                self._merge_pseudo(pid, [name, *aliases])
            if person["name"] != name:
                if person["name"] not in person["aliases"]:
                    person["aliases"].append(person["name"])
                person["name"] = name
            for alias in aliases:
                if alias and alias != name and alias not in person["aliases"]:
                    person["aliases"].append(alias)
            person["last_seen"] = _now()
            person["message_count"] = person.get("message_count", 0) + 1
            self.save()
        return person

    def _merge_pseudo(self, pid: str, names: List[str]) -> None:
        for name in names:
            pseudo = self.people.pop(f"name:{name.lower()}", None)
            if pseudo:
                target = self.people[pid]
                for fact in pseudo["facts"]:
                    if not is_duplicate(fact["text"], (f["text"] for f in target["facts"])):
                        target["facts"].append(fact)
                (PROFILE_DIR / f"{_slug(pseudo['name'])}-{_slug(pseudo['id'])}.md").unlink(missing_ok=True)
                print(f"Merged name-only profile '{pseudo['name']}' into user {pid}")

    def resolve(self, name: str, candidates: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Map a name the model used to a person id. `candidates` (name -> id, from the
        conversation the facts came from) wins over the global registry.
        """
        key = name.strip().lstrip("@").lower()
        if not key:
            return None
        for cname, cid in (candidates or {}).items():
            if cname.lower() == key:
                return cid
        with self._lock:
            for pid, person in self.people.items():
                names = [person["name"], *person.get("aliases", [])]
                if any(n.lower() == key for n in names):
                    return pid
        # Loose match: "Sava" for "sava_x", "Ekkusu" for "ekkusu_x"
        for cname, cid in (candidates or {}).items():
            if _norm(cname).startswith(_norm(key)) or _norm(key).startswith(_norm(cname)):
                return cid
        return None

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------
    def add_facts(self, person_id: str, facts: Iterable[str], name: Optional[str] = None) -> int:
        """Attach new facts to a person (creating a name-only profile if needed). Returns how many were new."""
        limit = int(get_config().get("memory", {}).get("max_facts_per_person", 80))
        added = 0
        with self._lock:
            person = self.people.get(person_id)
            if person is None:
                person = {"id": person_id, "name": name or person_id, "aliases": [], "first_seen": _now(),
                          "last_seen": _now(), "message_count": 0, "facts": []}
                self.people[person_id] = person
            for fact in facts:
                fact = fact.strip().rstrip(".")
                if len(fact) < 3 or is_duplicate(fact, (f["text"] for f in person["facts"])):
                    continue
                person["facts"].append({"id": uuid.uuid4().hex[:10], "text": fact, "added": _now()})
                added += 1
            if len(person["facts"]) > limit:
                person["facts"] = person["facts"][-limit:]
            if added:
                self.save()
        return added

    def add_notes(self, notes: Iterable[str]) -> int:
        added = 0
        with self._lock:
            for note in notes:
                note = note.strip()
                if len(note) < 5 or is_duplicate(note, (n["text"] for n in self.notes)):
                    continue
                self.notes.append({"id": uuid.uuid4().hex[:10], "text": note, "added": _now()})
                added += 1
            self.notes = self.notes[-500:]
            if added:
                self.save()
        return added

    def remove_fact(self, person_id: str, fact_id: str) -> bool:
        with self._lock:
            person = self.people.get(person_id)
            if not person:
                return False
            before = len(person["facts"])
            person["facts"] = [f for f in person["facts"] if f["id"] != fact_id]
            self.save()
            return len(person["facts"]) != before

    def remove_note(self, note_id: str) -> bool:
        with self._lock:
            before = len(self.notes)
            self.notes = [n for n in self.notes if n["id"] != note_id]
            self.save()
            return len(self.notes) != before

    def forget(self, person_id: str) -> bool:
        with self._lock:
            person = self.people.pop(person_id, None)
            self.save()
        if person:
            (PROFILE_DIR / f"{_slug(person['name'])}-{_slug(person_id)}.md").unlink(missing_ok=True)
        return person is not None

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    def get(self, person_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            person = self.people.get(str(person_id))
            return json.loads(json.dumps(person)) if person else None

    def all(self) -> List[Dict[str, Any]]:
        with self._lock:
            people = json.loads(json.dumps(list(self.people.values())))
        return sorted(people, key=lambda p: p.get("last_seen", ""), reverse=True)

    def profile_block(self, person_ids: Iterable[str], max_facts: int = 15) -> str:
        """Prompt text describing the given people (most recent facts first)."""
        lines = []
        with self._lock:
            for pid in dict.fromkeys(str(p) for p in person_ids):
                person = self.people.get(pid)
                if not person:
                    continue
                aka = f" (also known as {', '.join(person['aliases'][-3:])})" if person.get("aliases") else ""
                facts = [f["text"] for f in person["facts"][-max_facts:]]
                body = "; ".join(facts) if facts else "nothing known yet"
                lines.append(f"- {person['name']}{aka}: {body}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Markdown export for the knowledge base
    # ------------------------------------------------------------------
    def export_markdown(self, person_ids: Optional[Iterable[str]] = None) -> List[Path]:
        written: List[Path] = []
        with self._lock:
            ids = list(person_ids) if person_ids is not None else list(self.people)
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            for pid in ids:
                person = self.people.get(pid)
                if not person:
                    continue
                path = PROFILE_DIR / f"{_slug(person['name'])}-{_slug(pid)}.md"
                for stale in PROFILE_DIR.glob(f"*-{_slug(pid)}.md"):
                    if stale != path:
                        stale.unlink(missing_ok=True)
                names = ", ".join([person["name"], *person.get("aliases", [])])
                facts = "\n".join(f"- user personal information: {person['name']} {f['text']}" for f in person["facts"])
                path.write_text(f"# {person['name']}\nNames: {names}\n\n{facts}\n", encoding="utf-8")
                written.append(path)
            if self.notes:
                NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
                NOTES_FILE.write_text("# Things Metis learned\n\n" + "\n".join(f"- {n['text']}" for n in self.notes) + "\n",
                                      encoding="utf-8")
                written.append(NOTES_FILE)
        return written


_store: Optional[PeopleStore] = None
_store_lock = threading.Lock()


def get_people() -> PeopleStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = PeopleStore()
        return _store
