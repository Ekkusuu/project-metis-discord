"""
What Metis remembers, in the way people remember: a SQLite table of memories that fade unless they come back.

Two kinds, both about people (person ids are Discord user ids, see backend/people.py):
- fact: one lasting thing about one person ("plays League of Legends"), with the quote it came from.
- episode: a short third-person summary of a conversation that ended ("Ekkusu and Okeida planned a ranked night").

Nothing is ever deleted by time. Each memory has a retention computed when it is read (Ebbinghaus):
`exp(-days since it was last confirmed or recalled / stability)`, where stability grows with importance and with
every confirmation or recall. Below a floor a memory is simply not brought to mind (backend/recall.py), until
something reminds her of it again. Similarity to the moment is embedding cosine when the knowledge base's
embedder is up, else word overlap on stems.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.config import MEMORY_DIR, get_config

DB_FILE = MEMORY_DIR / "memories.db"

# Defaults for the `memory:` keys in config.yaml.
# recall_threshold was calibrated with the real embeddinggemma-300m on ~25 cues: greetings, "what's up", "lol" and
# small talk peaked at 0.36 cosine to any memory, topical cues ("what should I play tonight" vs "plays League of
# Legends") scored 0.38-0.56. 0.40 recalls the topical ones and leaves a 0.04 margin below it for noise; as the
# score is cosine x retention, a faded memory needs a closer match. The word-overlap fallback is a share of
# stems (see lexical_similarity): one shared word with the cue is 0.33 or more, a greeting shares none.
DEFAULTS: Dict[str, float] = {
    "fact_stability_days": 60,      # how long an unremarkable fact (importance 5, never confirmed) lasts, e-folding
    "episode_stability_days": 14,   # episodes fade faster: what happened one night matters less than who someone is
    "min_stability_days": 3,
    "retention_floor": 0.25,        # below this a memory isn't brought to mind by a passing cue
    "search_retention_floor": 0.05,  # ...but she can still dig for it when asked
    "recall_threshold": 0.40,       # embedding cosine x retention a memory must reach to be brought to mind
    "recall_threshold_words": 0.3,  # the same on word overlap, when no embedder is available
    "recall_max": 3,                # memories brought to mind per reply, at most one of them an episode
    "importance_bonus": 0.05,       # ranking nudge: +-this at importance 10 / 0
    "search_threshold_factor": 0.75,  # asked about a person or told to think back: the thresholds above times this
}


def setting(name: str) -> float:
    value = (get_config().get("memory") or {}).get(name, DEFAULTS[name])
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(DEFAULTS[name])


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_time(iso: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Strength
# ---------------------------------------------------------------------------
def stability(kind: str, importance: int, confirmations: int, recalls: int) -> float:
    """Days over which a memory fades (e-folding). Important and rehearsed memories last longer."""
    base = setting("episode_stability_days" if kind == "episode" else "fact_stability_days")
    days = base * (1 + 0.35 * (importance - 5)) * (1 + 0.5 * math.log(1 + confirmations + recalls))
    return max(days, setting("min_stability_days"))


def retention(memory: Dict[str, Any], now: Optional[datetime] = None) -> float:
    """1.0 right after it was last confirmed or recalled, falling towards 0 (never reaching it)."""
    now = now or datetime.now(timezone.utc)
    last = max(parse_time(memory["last_confirmed"]), parse_time(memory["last_recalled"] or memory["last_confirmed"]))
    days = max(0.0, (now - last).total_seconds() / 86400)
    return math.exp(-days / stability(memory["kind"], memory["importance"], memory["confirmations"], memory["recalls"]))


# ---------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------
# Words that say nothing about a topic: function words, greetings, chat filler, and the contractions the
# tokenizer leaves behind. A cue made only of these ("Hello, how are you?") cues no memory.
STOPWORDS = frozenset(
    """a about above after again all also am an and any are aren as at be because been before being below between both
    but by can cannot could did do does doing dont down during each few for from further get got gets had has have
    having he her here hers herself him himself his how if in into is it its itself just let may me might more most
    must my myself no nor not now of off on once only or other our ours ourselves out over own same shall she should so
    some such than that the their theirs them themselves then there these they this those through to too under until up
    us very was we were what when where which while who whom why will with would you your yours yourself yourselves
    hello hi hey hiya yo sup howdy bye goodbye thanks thank please sorry okay ok yeah yes yep yup nope nah well oh ah
    hmm uh um lol haha right sure maybe actually really pretty quite kind thing things something anything everything
    one two like love want wanna gonna going go goes come know think say said tell tells talk talking see look let make
    need dont doesnt didnt cant couldnt wont wouldnt isnt arent wasnt werent im ive ill id youre youve youll thats
    whats hows lets theyre theres its ones lot lots bit much many give take
    """.split()
)


def stem(word: str) -> str:
    """A light suffix stripper: plays/played/playing -> play, nurses -> nurs, moving/moved -> mov."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    stripped = False
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and not word.endswith("ss") and len(word) - len(suffix) >= 3:
            word, stripped = word[: -len(suffix)], suffix in ("ing", "ed")
            break
    if stripped and len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiouls":
        word = word[:-1]  # running -> run, stopped -> stop
    return word[:-1] if len(word) > 3 and word.endswith("e") else word


def tokens(text: str) -> List[str]:
    text = text.lower().replace("’", "'")
    return re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", text).replace("'", ""))


def content_stems(text: str, ignore: Iterable[str] = (), min_length: int = 3, extra_stopwords: Iterable[str] = ()) -> set:
    """Stems of the words that carry meaning (3 letters and up: 'dog', 'cat', 'gym'), minus `ignore` (names)."""
    skip = {w for name in ignore for w in tokens(name)}
    stop = STOPWORDS.union(extra_stopwords)
    return {stem(w) for w in tokens(text) if len(w) >= min_length and w not in stop and w not in skip}


def lexical_similarity(cue: set, memory: set) -> float:
    """
    Shared stems over the smaller side, with the memory capped at 3 words: an episode of fifteen words isn't
    "less similar" because the cue only touched the one thing it was about.
    """
    if not cue or not memory:
        return 0.0
    return min(1.0, len(cue & memory) / min(len(cue), min(len(memory), 3)))


# ---------------------------------------------------------------------------
# Embeddings (the knowledge base's model; None when it isn't available)
# ---------------------------------------------------------------------------
def _embed(texts: List[str], query: bool = False) -> Optional[List[List[float]]]:
    if not (get_config().get("rag") or {}).get("enabled", False):
        return None
    from backend.rag_engine import embed_texts

    return embed_texts(texts, query=query)


def _pack(vector: Sequence[float]) -> bytes:
    import numpy as np

    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return (array / norm if norm else array).tobytes()


def _unpack(blob: Optional[bytes], dim: int = 0):
    import numpy as np

    if not blob:
        return None
    array = np.frombuffer(blob, dtype=np.float32)
    return array if not dim or array.size == dim else None


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('fact', 'episode')),
    person_ids TEXT NOT NULL,
    text TEXT NOT NULL,
    quote TEXT NOT NULL DEFAULT '',
    importance INTEGER NOT NULL DEFAULT 5,
    created TEXT NOT NULL,
    last_confirmed TEXT NOT NULL,
    last_recalled TEXT,
    confirmations INTEGER NOT NULL DEFAULT 0,
    recalls INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded', 'forgotten')),
    superseded_by TEXT,
    session TEXT NOT NULL DEFAULT '',
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS memories_status ON memories (status, kind);
"""


class MemoryStore:
    def __init__(self, path: Path = DB_FILE) -> None:
        self.path = path
        self._lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    # -- rows -----------------------------------------------------------------
    @staticmethod
    def _row(row: sqlite3.Row) -> Dict[str, Any]:
        memory = dict(row)
        memory["person_ids"] = json.loads(memory["person_ids"])
        memory["last_recalled"] = memory["last_recalled"] or ""
        return memory

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row(row) if row else None

    def active(self, kind: Optional[str] = None, person_ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """Active memories, oldest first; of `kind`, and about anyone in `person_ids` (all if None)."""
        sql, args = "SELECT * FROM memories WHERE status = 'active'", []
        if kind:
            sql, args = sql + " AND kind = ?", [kind]
        with self._lock:
            rows = [self._row(r) for r in self._db.execute(sql + " ORDER BY created, rowid", args)]
        if person_ids is None:
            return rows
        wanted = {str(p) for p in person_ids}
        return [m for m in rows if wanted & set(m["person_ids"])]

    def count(self, kind: Optional[str] = None) -> int:
        return len(self.active(kind))

    def add(self, kind: str, person_ids: Iterable[str], text: str, quote: str = "", importance: int = 5,
            created: Optional[str] = None, session: str = "", memory_id: Optional[str] = None,
            embed: bool = True) -> Dict[str, Any]:
        """Store a memory. An id that already exists is left alone (the migration can run twice)."""
        created = created or now_iso()
        memory_id = memory_id or uuid.uuid4().hex[:10]
        vector = None
        if embed:
            try:
                vector = (_embed([text]) or [None])[0]
            except Exception as e:
                print(f"Warning: could not embed a memory ({e}); it will be matched by words")
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO memories (id, kind, person_ids, text, quote, importance, created, "
                "last_confirmed, session, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (memory_id, kind, json.dumps([str(p) for p in person_ids]), text.strip(), quote.strip(),
                 max(1, min(10, int(importance))), created, created, session,
                 _pack(vector) if vector is not None else None))
            self._db.commit()
        return self.get(memory_id) or {}

    def _update(self, memory_id: str, **fields: Any) -> bool:
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            cursor = self._db.execute(f"UPDATE memories SET {assignments} WHERE id = ?", (*fields.values(), memory_id))
            self._db.commit()
            return cursor.rowcount > 0

    def reinforce(self, memory_id: str, importance: Optional[int] = None, now: Optional[str] = None) -> bool:
        """Someone said it again: it was confirmed just now, and lasts longer. Importance only ever rises."""
        memory = self.get(memory_id)
        if not memory or memory["status"] != "active":
            return False
        return self._update(memory_id, confirmations=memory["confirmations"] + 1, last_confirmed=now or now_iso(),
                            importance=max(memory["importance"], min(10, int(importance or 0))))

    def mark_recalled(self, memory_ids: Iterable[str], now: Optional[str] = None) -> None:
        """Brought to mind: remembering something strengthens it (the testing effect)."""
        stamp = now or now_iso()
        with self._lock:
            self._db.executemany("UPDATE memories SET recalls = recalls + 1, last_recalled = ? WHERE id = ?",
                                 [(stamp, i) for i in memory_ids])
            self._db.commit()

    def supersede(self, old_id: str, new_id: str) -> bool:
        """The old fact no longer holds: kept (it was true once), pointing at what replaced it."""
        return self._update(old_id, status="superseded", superseded_by=new_id)

    def set_text(self, memory_id: str, text: str) -> bool:
        """A correction typed by hand: the same memory, said differently (confirmed now, embedding redone)."""
        return self._update(memory_id, text=text.strip(), last_confirmed=now_iso(), embedding=None)

    def set_status(self, memory_id: str, status: str) -> bool:
        return self._update(memory_id, status=status)

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            cursor = self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._db.commit()
            return cursor.rowcount > 0

    def delete_person(self, person_id: str) -> None:
        """Forget a person: their facts go, and so do episodes nobody else was part of."""
        person_id = str(person_id)
        with self._lock:
            for memory in self.active() + self._inactive():
                if person_id not in memory["person_ids"]:
                    continue
                rest = [p for p in memory["person_ids"] if p != person_id]
                if memory["kind"] == "fact" or not rest:
                    self._db.execute("DELETE FROM memories WHERE id = ?", (memory["id"],))
                else:
                    self._db.execute("UPDATE memories SET person_ids = ? WHERE id = ?", (json.dumps(rest), memory["id"]))
            self._db.commit()

    def _inactive(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._row(r) for r in self._db.execute("SELECT * FROM memories WHERE status != 'active'")]

    def reassign_person(self, old_id: str, new_id: str) -> None:
        """A name-only profile turned out to be a real user: its memories become theirs."""
        with self._lock:
            for memory in self.active() + self._inactive():
                if old_id in memory["person_ids"]:
                    ids = list(dict.fromkeys(new_id if p == old_id else p for p in memory["person_ids"]))
                    self._db.execute("UPDATE memories SET person_ids = ? WHERE id = ?", (json.dumps(ids), memory["id"]))
            self._db.commit()

    def forget_weakest(self, person_id: str, keep: int, now: Optional[datetime] = None) -> int:
        """Over the cap (memory.max_facts_per_person): the faintest facts are let go. Returns how many."""
        facts = self.active("fact", [person_id])
        if len(facts) <= keep:
            return 0
        facts.sort(key=lambda m: retention(m, now) * (1 + m["importance"] / 10))
        for memory in facts[: len(facts) - keep]:
            self.set_status(memory["id"], "forgotten")
        return len(facts) - keep

    # -- similarity -------------------------------------------------------------
    def ensure_embeddings(self, memories: List[Dict[str, Any]], dim: int) -> None:
        """Embed memories written while no embedder was up (or by the migration). Quiet if it still isn't."""
        missing = [m for m in memories if _unpack(m["embedding"], dim) is None]
        if not missing:
            return
        try:
            vectors = _embed([m["text"] for m in missing])
        except Exception as e:
            print(f"Warning: could not embed memories ({e})")
            return
        if not vectors:
            return
        for memory, vector in zip(missing, vectors):
            memory["embedding"] = _pack(vector)
            self._update(memory["id"], embedding=memory["embedding"])

    def similarities(self, cues: Sequence[str], memories: List[Dict[str, Any]],
                     ignore: Iterable[str] = ()) -> List[Tuple[float, str]]:
        """
        (similarity, "vec" | "words") of each memory to the cue: the latest line, in chronological `cues`, and the
        few before it. Vectors where both sides have one, else stems. `ignore`: names, which every episode
        contains and every greeting may say, so they would cue anything.
        """
        cues = [c for c in cues if c.strip()]
        if not cues or not memories:
            return [(0.0, "words")] * len(memories)
        cue_vectors = None
        try:
            # The latest line alone and with the lines before it; the better match counts.
            cue_vectors = _embed([cues[-1], " ".join(cues)], query=True)
        except Exception as e:
            print(f"Warning: could not embed the cue ({e}); matching by words")
        dim = len(cue_vectors[0]) if cue_vectors else 0
        if cue_vectors:
            self.ensure_embeddings(memories, dim)
        import numpy as np

        cue_words = content_stems(" ".join(cues), ignore)
        out: List[Tuple[float, str]] = []
        for memory in memories:
            vector = _unpack(memory["embedding"], dim) if cue_vectors else None
            if vector is not None:
                out.append((max(float(np.dot(vector, np.asarray(c, dtype=np.float32)
                                              / (np.linalg.norm(c) or 1))) for c in cue_vectors), "vec"))
            else:
                out.append((lexical_similarity(cue_words, content_stems(memory["text"], ignore)), "words"))
        return out

    def nearest(self, person_id: str, texts: Sequence[str], limit: int = 8) -> List[Dict[str, Any]]:
        """The person's active facts most like any of `texts`, closest first (what a new fact must be compared with)."""
        facts = self.active("fact", [person_id])
        if len(facts) <= limit:
            return facts
        best = [0.0] * len(facts)
        for text in texts:
            for i, (sim, _) in enumerate(self.similarities([text], facts)):
                best[i] = max(best[i], sim)
        ranked = sorted(range(len(facts)), key=lambda i: -best[i])
        return [facts[i] for i in ranked[:limit]]
