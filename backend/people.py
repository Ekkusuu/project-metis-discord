"""
Who Metis knows: one profile per Discord user, keyed by their stable user id.

Display names change and collide; user ids don't. Every message (text or
voice) updates the speaker's profile (names, aliases, last seen). What she has
learned about people lives in the memory store (backend/memory_store.py, memory/memories.db):
facts about one person and episodes of conversations; this class is the registry
in front of it, so a person's facts and episodes are read and written through it.
General notes that aren't about one person stay here and are written out as Markdown
for the knowledge base.

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
from backend.memory_store import MemoryStore, retention

PEOPLE_FILE = MEMORY_DIR / "people.json"
PROFILE_DIR = MEMORY_DIR / "long_term" / "people"  # the old Markdown profiles, removed on load
NOTES_FILE = MEMORY_DIR / "long_term" / "learned_notes.md"

# Importance given to facts that were stored before facts were rated (the migration).
MIGRATED_IMPORTANCE = 6


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def _same(a: str, b: str, threshold: float) -> bool:
    return a == b or (a in b and len(a) > 12) or SequenceMatcher(None, a, b).ratio() >= threshold


def is_duplicate(fact: str, existing: Iterable[str], threshold: float = 0.82) -> bool:
    """True if `fact` says (nearly) the same thing as an existing fact."""
    a = _norm(fact)
    if not a:
        return True
    return any(_same(a, _norm(other), threshold) for other in existing)


class PeopleStore:
    def __init__(self, path: Path = PEOPLE_FILE, memories: Optional[MemoryStore] = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self.people: Dict[str, Dict[str, Any]] = {}
        self.notes: List[Dict[str, Any]] = []
        self.memories = memories or MemoryStore(path.with_name("memories.db"))
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.people = data.get("people", {})
                self.notes = data.get("notes", [])
            except Exception as e:
                print(f"Warning: could not read {self.path.name}: {e}")
                return
        self._migrate_facts()
        for old_profile in PROFILE_DIR.glob("*.md"):  # facts are in the store now; don't also index them as notes
            old_profile.unlink(missing_ok=True)

    def _migrate_facts(self) -> None:
        """
        Facts used to be a list inside each profile. They move into the memory store once (same ids, so a crash
        between the two writes can be repeated), as importance 6 facts created when they were added.
        """
        moved = 0
        with self._lock:
            for pid, person in self.people.items():
                for fact in person.pop("facts", None) or []:
                    self.memories.add("fact", [pid], fact["text"], importance=MIGRATED_IMPORTANCE,
                                      created=fact.get("added"), memory_id=fact.get("id"), embed=False)
                    moved += 1
            if moved:
                self.save()
                print(f"🧠 Moved {moved} fact(s) from {self.path.name} into the memory store")

    def save_soon(self, delay: float = 5.0) -> None:
        """Coalesce frequent, low-value writes (last seen / message counts) into one save."""
        with self._lock:
            if getattr(self, "_save_timer", None) is None:
                self._save_timer = threading.Timer(delay, self.save)
                self._save_timer.daemon = True
                self._save_timer.start()

    def save(self) -> None:
        with self._lock:
            timer = getattr(self, "_save_timer", None)
            if timer is not None:
                timer.cancel()
                self._save_timer = None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"people": self.people, "notes": self.notes}, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self.path)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def _new_person(self, pid: str, name: str) -> Dict[str, Any]:
        person = {"id": pid, "name": name, "aliases": [], "first_seen": _now(), "last_seen": _now(), "message_count": 0}
        self.people[pid] = person
        return person

    def seen(self, user_id: int | str, name: str, aliases: Iterable[str] = ()) -> Dict[str, Any]:
        """Record that a user spoke. Returns their profile."""
        pid = str(user_id)
        with self._lock:
            person = self.people.get(pid)
            if person is None:
                person = self._new_person(pid, name)
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
            self.save_soon()
        return person

    def _merge_pseudo(self, pid: str, names: List[str]) -> None:
        for name in names:
            pseudo = self.people.pop(f"name:{name.lower()}", None)
            if pseudo:
                self.memories.reassign_person(pseudo["id"], pid)
                kept: List[str] = []
                for fact in self.memories.active("fact", [pid]):
                    if is_duplicate(fact["text"], kept):
                        self.memories.delete(fact["id"])
                    else:
                        kept.append(fact["text"])
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

    def names_of(self, person_ids: Iterable[str]) -> Dict[str, List[str]]:
        """person id -> every name they go by (current name first), for the ones we know."""
        with self._lock:
            return {pid: [p["name"], *p.get("aliases", [])] for pid in (str(i) for i in person_ids)
                    if (p := self.people.get(pid))}

    # ------------------------------------------------------------------
    # Facts (in the memory store)
    # ------------------------------------------------------------------
    def _ensure_person(self, person_id: str, name: Optional[str]) -> None:
        if person_id not in self.people:
            self._new_person(person_id, name or person_id)

    def add_fact(self, person_id: str, text: str, importance: int = MIGRATED_IMPORTANCE, quote: str = "",
                 name: Optional[str] = None, session: str = "") -> Optional[Dict[str, Any]]:
        """
        Store a new fact about a person (creating a name-only profile if needed). If it only repeats a stored
        fact, that one is confirmed instead (it lasts longer) and None is returned.
        """
        text = text.strip().rstrip(".")
        if len(text) < 3:
            return None
        with self._lock:
            self._ensure_person(person_id, name)
            same = next((f for f in self.memories.active("fact", [person_id]) if is_duplicate(text, [f["text"]])), None)
            if same:
                self.memories.reinforce(same["id"], importance)
                return None
            fact = self.memories.add("fact", [person_id], text, quote=quote, importance=importance, session=session)
            limit = int(get_config().get("memory", {}).get("max_facts_per_person", 80))
            self.memories.forget_weakest(person_id, limit)
            self.save()
            return fact

    def add_facts(self, person_id: str, facts: Iterable[str], name: Optional[str] = None) -> int:
        """Attach new facts to a person. Returns how many were new."""
        return sum(1 for fact in facts if self.add_fact(person_id, fact, name=name))

    def supersede_fact(self, person_id: str, old_id: str, text: str, importance: int = MIGRATED_IMPORTANCE,
                       quote: str = "", session: str = "") -> bool:
        """The fact changed (moved city, new main): the old one is kept as superseded, pointing at the new one."""
        old = self.memories.get(old_id)
        if not old or old["kind"] != "fact" or person_id not in old["person_ids"]:
            return False
        new = self.memories.add("fact", [person_id], text.strip().rstrip("."), quote=quote,
                                importance=max(importance, old["importance"]), session=session)
        return self.memories.supersede(old_id, new["id"])

    def reinforce_fact(self, person_id: str, fact_id: str, importance: int = 0) -> bool:
        fact = self.memories.get(fact_id)
        return bool(fact and person_id in fact["person_ids"] and self.memories.reinforce(fact_id, importance))

    def update_fact(self, person_id: str, fact_id: str, text: str) -> bool:
        """Replace what a stored fact says (a correction by hand)."""
        fact = self.memories.get(fact_id)
        return bool(fact and fact["kind"] == "fact" and person_id in fact["person_ids"]
                    and self.memories.set_text(fact_id, text.rstrip(".")))

    def remove_fact(self, person_id: str, fact_id: str) -> bool:
        fact = self.memories.get(fact_id)
        return bool(fact and fact["kind"] == "fact" and person_id in fact["person_ids"] and self.memories.delete(fact_id))

    def add_episode(self, person_ids: Iterable[str], text: str, importance: int = 5, session: str = "",
                    names: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """An episode shared by everyone in `person_ids` (`names`: id -> name, for ones without a profile yet)."""
        ids = [str(p) for p in person_ids]
        with self._lock:
            for pid in ids:
                self._ensure_person(pid, (names or {}).get(pid))
            self.save()
        return self.memories.add("episode", ids, text, importance=importance, session=session)

    def remove_episode(self, person_id: str, episode_id: str) -> bool:
        episode = self.memories.get(episode_id)
        return bool(episode and episode["kind"] == "episode" and person_id in episode["person_ids"]
                    and self.memories.delete(episode_id))

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
        self.memories.delete_person(person_id)
        return person is not None

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    def _with_memories(self, person: Dict[str, Any], active: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The profile plus what's remembered about the person: `facts`, and `episodes` they were part of."""
        mine = [m for m in active if person["id"] in m["person_ids"]]
        names = {pid: ns[0] for pid, ns in self.names_of({p for m in mine for p in m["person_ids"]}).items()}

        def view(m: Dict[str, Any]) -> Dict[str, Any]:
            return {"id": m["id"], "text": m["text"], "added": m["created"], "importance": m["importance"],
                    "confirmations": m["confirmations"], "recalls": m["recalls"], "retention": round(retention(m), 2)}

        return {**json.loads(json.dumps(person)),
                "facts": [view(m) for m in mine if m["kind"] == "fact"],
                "episodes": [{**view(m), "people": [names.get(p, p) for p in m["person_ids"]]}
                             for m in mine if m["kind"] == "episode"]}

    def get(self, person_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            person = self.people.get(str(person_id))
            return self._with_memories(person, self.memories.active()) if person else None

    def all(self) -> List[Dict[str, Any]]:
        with self._lock:
            active = self.memories.active()
            people = [self._with_memories(p, active) for p in self.people.values()]
        return sorted(people, key=lambda p: p.get("last_seen", ""), reverse=True)

    def who_is_who(self, person_ids: Iterable[str]) -> str:
        """The people by name (and the other names they go by): what's always in the prompt, with no facts."""
        with self._lock:
            people = [self.people[pid] for pid in dict.fromkeys(str(p) for p in person_ids) if pid in self.people]
        return ", ".join(p["name"] + (f" (also known as {', '.join(p['aliases'][-3:])})" if p.get("aliases") else "")
                         for p in people)

    # ------------------------------------------------------------------
    # Markdown export of the general notes for the knowledge base
    # ------------------------------------------------------------------
    def export_notes(self) -> List[Path]:
        with self._lock:
            if not self.notes:
                return []
            NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
            NOTES_FILE.write_text("# Things Metis learned\n\n" + "\n".join(f"- {n['text']}" for n in self.notes) + "\n",
                                  encoding="utf-8")
            return [NOTES_FILE]


_store: Optional[PeopleStore] = None
_store_lock = threading.Lock()


def get_people() -> PeopleStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = PeopleStore()
        return _store


def get_memories() -> MemoryStore:
    return get_people().memories
