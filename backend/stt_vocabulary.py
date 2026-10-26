"""
Domain vocabularies for speech-to-text (voice.stt_vocabulary in config.yaml): names the recogniser has never
seen, like game characters ("Kha'Zix" came out as "Kazakhs", "Cogma" for Kog'Maw).

Two layers, as in production voice assistants:
1. Phrase boosting while decoding (NeMo GPU-PB): the recogniser is nudged toward the listed names. Kept mild,
   because a strong boost turns ordinary words into names.
2. Phonetic correction afterwards: words are converted to phonemes (g2p_en) and compared with the names'
   phonemes. Made-up words ("Kaziks") are corrected readily; real English words ("Kazakhs") only when the
   conversation is about that topic, so "I went to Kazakhstan" stays as it is.

Each vocabulary comes from a URL (JSON) and/or a list, is cached in memory/stt_vocabulary/ and refreshed daily,
so new names arrive without code changes and an outage falls back to the cached copy.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.config import resolve_path

_CACHE_DIR = "memory/stt_vocabulary"


@dataclass
class Vocabulary:
    topic: str
    terms: List[str]
    cues: List[str] = field(default_factory=list)  # words that show the conversation is about this topic


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "vocabulary"


def _extract(data: Any, key: str) -> List[str]:
    """Names from a JSON list of objects (or an object of objects), by `key`; a plain list of strings as is."""
    items = data.values() if isinstance(data, dict) else data
    out = []
    for item in items or []:
        value = item.get(key) if isinstance(item, dict) else item
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    return out


def _fetch(entry: Dict[str, Any]) -> List[str]:
    """The vocabulary's names from its URL, cached; the cached copy when the URL can't be reached."""
    url = entry.get("url")
    if not url:
        return []
    cache = resolve_path(_CACHE_DIR) / f"{_slug(entry.get('topic') or url)}.json"
    max_age = float(entry.get("refresh_hours", 24)) * 3600
    if cache.exists() and time.time() - cache.stat().st_mtime < max_age:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Metis"})
        with urllib.request.urlopen(request, timeout=10) as response:
            names = _extract(json.load(response), entry.get("field", "name"))
        exclude = {str(x).lower() for x in entry.get("exclude") or []}
        names = sorted({n for n in names if n.lower() not in exclude})
        if names:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
            return names
    except Exception as e:
        print(f"⚠️ Couldn't download the {entry.get('topic') or url} vocabulary ({e}); using the saved copy")
    try:
        return json.loads(cache.read_text(encoding="utf-8"))
    except Exception:
        return []


def load_vocabularies(entries: Sequence[Dict[str, Any]]) -> List[Vocabulary]:
    vocabularies = []
    for entry in entries or []:
        terms = list(dict.fromkeys([*_fetch(entry), *[str(t) for t in entry.get("terms") or []]]))
        if terms:
            vocabularies.append(Vocabulary(str(entry.get("topic") or "names"), terms,
                                           [str(c).lower() for c in entry.get("cues") or []]))
    return vocabularies


# ---------------------------------------------------------------------------
# Phonetic matching
# ---------------------------------------------------------------------------
_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
# Consonants that speech-to-text and accents swap for each other.
_CLOSE = [{"S", "Z"}, {"K", "G"}, {"T", "D"}, {"P", "B"}, {"F", "V"}, {"TH", "DH"}, {"CH", "JH", "SH", "ZH"},
          {"M", "N", "NG"}]
_WEAK = {"HH", "Y", "W"}  # glides that get dropped or added


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if a in _VOWELS and b in _VOWELS:
        return 0.4
    if any(a in group and b in group for group in _CLOSE):
        return 0.5
    return 1.0


def _skeleton(phonemes: Sequence[str]) -> Tuple[str, ...]:
    """The consonants, with ones that are easily confused merged and glides dropped."""
    out = []
    for p in phonemes:
        if p in _VOWELS or p in _WEAK:
            continue
        out.append(next((min(group) for group in _CLOSE if p in group), p))
    return tuple(out)


def phonetic_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """1 = sounds the same. Weighted edit distance over phonemes (vowels and similar consonants are cheap)."""
    if not a or not b:
        return 0.0
    previous = [0.0]
    for p in b:
        previous.append(previous[-1] + (0.5 if p in _WEAK else 1.0))
    for x in a:
        current = [previous[0] + (0.5 if x in _WEAK else 1.0)]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + (0.5 if x in _WEAK else 1.0),
                               current[j - 1] + (0.5 if y in _WEAK else 1.0),
                               previous[j - 1] + _sub_cost(x, y)))
        previous = current
    return max(0.0, 1.0 - previous[-1] / max(len(a), len(b)))


_WORD = re.compile(r"[A-Za-z][A-Za-z'.\-]*[A-Za-z]|[A-Za-z]")


class PhoneticCorrector:
    """Snaps misheard words onto vocabulary names by how they sound (see the module docstring)."""

    def __init__(self, vocabularies: Sequence[Vocabulary], protected: Sequence[str] = (), *,
                 threshold: float = 0.78, threshold_on_topic: float = 0.72, threshold_english: float = 0.82,
                 topic_seconds: float = 600.0) -> None:
        from g2p_en import G2p

        self._g2p = G2p()
        self.threshold, self.threshold_on_topic, self.threshold_english = threshold, threshold_on_topic, threshold_english
        self.topic_seconds = topic_seconds
        self.vocabularies = list(vocabularies)
        self._protected = {p.lower() for p in protected}
        self._last_on_topic: Dict[str, float] = {}
        self._entries: List[Tuple[str, Vocabulary, List[str]]] = []  # (term, vocabulary, phonemes)
        self._exact: Dict[str, str] = {}
        for vocab in self.vocabularies:
            for term in vocab.terms:
                self._exact[self._key(term)] = term
                for spoken in self._spoken_forms(term):
                    self._entries.append((term, vocab, self.phonemes(spoken)))

    def protect(self, names: Sequence[str]) -> None:
        """Words that must never be changed (the names of people in the call)."""
        self._protected.update(n.lower() for n in names)

    @staticmethod
    def _key(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    @staticmethod
    def _spoken_forms(term: str) -> List[str]:
        """How a written name is said: "Kha'Zix" as one word and as parts, "Nunu & Willump" as "Nunu"."""
        term = re.sub(r"\bDr\.", "Doctor", term)
        forms = {term.replace("'", ""), term.replace("'", " ")}
        if "&" in term:
            forms.add(term.split("&")[0])
        return [f.strip() for f in forms if f.strip()]

    def _english(self, word: str) -> bool:
        return word.lower().strip("'.-") in self._g2p.cmu

    def phonemes(self, text: str) -> List[str]:
        out: List[str] = []
        for word in re.findall(r"[a-z]+", text.lower().replace("'", "")):
            pron = self._g2p.cmu[word][0] if word in self._g2p.cmu else self._g2p.predict(word)
            for p in pron:
                p = p.rstrip("012")
                # "ER" is "uh" + "r": "Trendomere" (EH R) vs "Tryndamere" (ER) differed by a whole phoneme.
                out.extend(["AH", "R"] if p == "ER" else [p])
        return out

    def _mentions(self, vocab: Vocabulary, text: str) -> bool:
        """The text names something from this vocabulary or one of its topic cues."""
        lowered = f" {text.lower()} "
        if any(re.search(rf"(?<![a-z]){re.escape(c)}(?![a-z])", lowered) for c in vocab.cues):
            return True
        keys = [self._key(re.sub(r"'s$", "", w, flags=re.IGNORECASE)) for w in _WORD.findall(text)]
        if any(k in self._exact and self._exact[k] in vocab.terms for k in keys):
            return True
        return any(t.lower() in lowered for t in vocab.terms if " " in t)

    def _on_topic(self, vocab: Vocabulary, text: str, now: float) -> bool:
        """This utterance, or one in the last few minutes, was about the vocabulary's topic."""
        return self._mentions(vocab, text) or now - self._last_on_topic.get(vocab.topic, 0.0) < self.topic_seconds

    def correct(self, text: str) -> str:
        if not self._entries or not text:
            return text
        now = time.time()
        on_topic = {v.topic: self._on_topic(v, text, now) for v in self.vocabularies}
        words = list(_WORD.finditer(text))
        candidates = []  # (score, start word, end word, replacement)
        for size in (3, 2, 1):
            for i in range(len(words) - size + 1):
                span = words[i:i + size]
                raw = text[span[0].start():span[-1].end()]
                possessive = ""
                if size == 1 and re.search(r"'s$", raw, re.IGNORECASE):
                    raw, possessive = raw[:-2], raw[-2:]
                key = self._key(raw)
                if not key or raw.lower() in self._protected:
                    continue
                if key in self._exact:
                    # Spelled out differently ("Kai Sa", "Bel Veth", "RekSai"): the name's own spelling. Already
                    # right, the words are taken, so "Aurelion Sol is" can't become "Aurelion Sol".
                    candidates.append((1.0 if raw != self._exact[key] else 2.0, i, i + size,
                                       self._exact[key] + possessive, raw))
                    continue
                if any(w.group(0).lower() in self._protected for w in span):
                    continue
                english = all(self._english(w.group(0)) for w in span)
                if english and size > 1:
                    continue  # "is real" -> "Ezreal": ordinary phrases are left alone
                sounds = self.phonemes(raw)
                if len(sounds) < (5 if english else 3):
                    continue
                best = max(((phonetic_similarity(sounds, p), term, vocab) for term, vocab, p in self._entries),
                           key=lambda x: x[0])
                score, term, vocab = best
                if english and _skeleton(sounds) != _skeleton(next(p for t, v, p in self._entries if t == term)):
                    # A real word ("counter") is only replaced if its consonants are the name's ("Kazakhs").
                    best = max(((phonetic_similarity(sounds, p), t, v) for t, v, p in self._entries
                                if _skeleton(p) == _skeleton(sounds)), key=lambda x: x[0], default=(0.0, term, vocab))
                    score, term, vocab = best
                topical = on_topic[vocab.topic]
                needed = (self.threshold_english if topical else 2.0) if english else (
                    self.threshold_on_topic if topical else self.threshold)
                if score >= needed:
                    candidates.append((score, i, i + size, term + possessive, raw))
        chosen, used = [], set()
        for score, start, end, replacement, raw in sorted(candidates, key=lambda c: (-c[0], c[1] - c[2])):
            if used.isdisjoint(range(start, end)):
                chosen.append((start, end, replacement, raw, score))
                used.update(range(start, end))
        chosen = [c for c in chosen if c[4] < 2.0]  # names that were already right
        for start, end, replacement, raw, score in sorted(chosen, reverse=True):
            print(f"🔧 Vocabulary ({score:.0%}): '{raw}' → '{replacement}'")
            text = text[:words[start].start()] + replacement + text[words[end - 1].end():]
        for vocab in self.vocabularies:
            if self._mentions(vocab, text):
                self._last_on_topic[vocab.topic] = now
        return text


def boost_phrases(vocabularies: Sequence[Vocabulary], extra: Sequence[str] = ()) -> List[str]:
    """Phrases for decoder boosting, written the way the recogniser writes them (capitalised, no '&')."""
    phrases = []
    for vocab in vocabularies:
        for term in vocab.terms:
            phrases.extend(PhoneticCorrector._spoken_forms(term)[:1] if "&" in term else [term])
    return list(dict.fromkeys([*phrases, *extra]))


def enable_boosting(model: Any, phrases: List[str], alpha: float) -> bool:
    """NeMo GPU-PB phrase boosting (NeMo >= 2.5) on a loaded transducer model. False if unsupported."""
    if not phrases or alpha <= 0:
        return False
    try:
        from omegaconf import open_dict

        with open_dict(model.cfg.decoding):
            model.cfg.decoding.strategy = "greedy_batch"
            # Triton isn't available on Windows; the plain PyTorch path is fast enough for a few hundred names.
            model.cfg.decoding.greedy.boosting_tree = {"key_phrases_list": phrases, "context_score": 1.0,
                                                       "depth_scaling": 2.0, "use_triton": False}
            model.cfg.decoding.greedy.boosting_tree_alpha = alpha
        model.change_decoding_strategy(model.cfg.decoding, verbose=False)
        return True
    except Exception as e:
        print(f"⚠️ Phrase boosting unavailable ({e}); names are still corrected after transcription")
        return False
