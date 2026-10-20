"""
Pure text helpers shared by the bot, the chat pipeline and the memory system.
No heavy imports here so it stays cheap to import and easy to test.
"""
from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# Reasoning ("<think>") handling
# ---------------------------------------------------------------------------
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNTIL_CLOSE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_TO_END = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks emitted by reasoning models.

    Handles complete blocks, a stray closing tag (everything before it is
    reasoning) and an unclosed opening tag (everything after it is reasoning).
    """
    if not text:
        return ""
    text = _THINK_BLOCK.sub("", text)
    if re.search(r"</think>", text, re.IGNORECASE):
        text = _THINK_UNTIL_CLOSE.sub("", text)
    text = _THINK_OPEN_TO_END.sub("", text)
    return _THINK_TAG.sub("", text).strip()


class ThinkFilter:
    """Incremental version of strip_thinking for token streams.

    feed() returns only the text that is safe to show/speak. A partial tag at
    the end of a chunk is held back until the next chunk disambiguates it.

    Limitation: text already returned can't be taken back, so reasoning that
    starts *without* an opening tag and ends with a stray "</think>" in a later
    chunk leaks. In practice the LLM service resolves the model's chat format,
    and node-llama-cpp then routes reasoning into thought segments that never
    reach the text stream; this filter is a safety net for tagged output.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        out: List[str] = []
        while self._buffer:
            lower = self._buffer.lower()
            tag = self._CLOSE if self._inside else self._OPEN
            idx = lower.find(tag)
            if idx != -1:
                if not self._inside:
                    out.append(self._buffer[:idx])
                self._buffer = self._buffer[idx + len(tag):]
                self._inside = not self._inside
                continue
            # A stray closing tag while outside means everything so far was reasoning.
            if not self._inside:
                close_idx = lower.find(self._CLOSE)
                if close_idx != -1:
                    out.clear()
                    self._buffer = self._buffer[close_idx + len(self._CLOSE):]
                    continue
            keep = _partial_tag_suffix(lower, tag)
            if not self._inside:
                keep = max(keep, _partial_tag_suffix(lower, self._CLOSE))
                out.append(self._buffer[: len(self._buffer) - keep])
            self._buffer = self._buffer[len(self._buffer) - keep:] if keep else ""
            break
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._inside else self._buffer
        self._buffer = ""
        return rest


def _partial_tag_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `tag`."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


# ---------------------------------------------------------------------------
# Reply cleanup
# ---------------------------------------------------------------------------
_NAME_PREFIX = re.compile(r"^\s*(?:\[?metis\]?\s*:|@metis\b[,:]?)\s*", re.IGNORECASE)


def strip_name_prefix(text: str) -> str:
    """Drop a leading "Metis:" / "[Metis]:" / "@Metis" the model sometimes adds to its own replies."""
    return _NAME_PREFIX.sub("", text, count=1)


# ---------------------------------------------------------------------------
# Sentence splitting for streaming TTS
# ---------------------------------------------------------------------------
ABBREVIATIONS = {
    # Titles
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "rev", "hon", "gov", "pres",
    # Academic/Professional
    "phd", "md", "esq", "inc", "ltd", "corp", "llc", "co",
    # Common abbreviations
    "vs", "etc", "eg", "ie", "al", "cf", "approx", "avg", "dept", "est",
    "fig", "min", "max", "misc", "no", "nos", "vol", "vols", "ed", "eds",
    "ft", "in", "oz", "lb", "lbs", "pt", "qt", "gal", "cm", "mm", "km", "kg",
    # Time/Date
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "am", "pm", "hr", "hrs", "sec", "secs",
    # Places
    "st", "ave", "blvd", "rd", "mt",
}


def is_sentence_boundary(text: str, pos: int) -> bool:
    """True if the punctuation at `pos` really ends a sentence (not "Dr.", "12.5", "U.S.")."""
    if pos < 0 or pos >= len(text):
        return False
    char = text[pos]
    if char in "!?":
        # Wait for the full run ("?!", "!!!") before cutting.
        return pos + 1 < len(text) and text[pos + 1] not in "!?."
    if char != ".":
        return False
    # Can't decide at the very end of the buffer (might be a decimal or "...").
    if pos + 1 >= len(text):
        return False
    nxt = text[pos + 1]
    if nxt.isdigit() or nxt == ".":
        return False
    if nxt not in " \n\t\"')":
        return False

    start = pos - 1
    while start >= 0 and text[start].isalnum():
        start -= 1
    word_before = text[start + 1:pos].lower()

    if word_before.isdigit():
        # "in 2024. Then" ends a sentence; "item 3. of" doesn't.
        next_pos = pos + 1
        while next_pos < len(text) and text[next_pos] in " \n\t":
            next_pos += 1
        if next_pos < len(text) and (text[next_pos].islower() or text[next_pos].isdigit()):
            return False
    if word_before in ABBREVIATIONS:
        return False
    # Initialisms like "U.S." / "A.I."
    if len(word_before) == 1 and word_before.isalpha() and start >= 1 and text[start] == ".":
        return False
    return True


def find_sentence_boundary(text: str) -> int:
    """Index of the first real sentence boundary in `text`, or -1."""
    for i, char in enumerate(text):
        if char in ".!?" and is_sentence_boundary(text, i):
            return i
        if char == "\n" and i > 0 and text[:i].strip():
            return i
    return -1


def pop_sentences(buffer: str) -> tuple[List[str], str]:
    """Split complete sentences off the front of `buffer`; returns (sentences, remainder)."""
    sentences: List[str] = []
    while True:
        end = find_sentence_boundary(buffer)
        if end == -1:
            return sentences, buffer
        sentence = buffer[: end + 1].strip()
        buffer = buffer[end + 1:].lstrip()
        if sentence:
            sentences.append(sentence)


_MARKDOWN_NOISE = re.compile(r"(\*\*|__|\*|`+|^#+\s*|^>\s*|^\s*[-•]\s+)", re.MULTILINE)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F000-\U0001F2FF\U00002600-\U000026FF️]+"
)
_ACTION = re.compile(r"\*[^*\n]{1,60}\*")


def clean_for_speech(text: str) -> str:
    """Remove things TTS would read out literally: *actions*, markdown, emoji, URLs."""
    text = _ACTION.sub("", text)
    text = re.sub(r"https?://\S+", "", text)
    text = _MARKDOWN_NOISE.sub("", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# ---------------------------------------------------------------------------
# Discord message splitting
# ---------------------------------------------------------------------------
def split_message(text: str, limit: int = 2000) -> List[str]:
    """Split text into Discord-sized chunks on natural boundaries.

    Prefers paragraph breaks, then line breaks, then spaces. Code fences that
    get cut are closed at the end of one chunk and reopened in the next.
    """
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    open_fence = ""  # e.g. "```python" if we're inside a code block
    while text:
        prefix = f"{open_fence}\n" if open_fence else ""
        budget = limit - len(prefix)
        if len(text) <= budget:
            chunks.append(prefix + text)
            break
        reserve = 4  # room for a closing "\n```"
        window = text[: budget - reserve]
        cut = -1
        for sep in ("\n\n", "\n", " "):
            idx = window.rfind(sep)
            if idx > budget // 3:
                cut = idx
                break
        if cut == -1:
            cut = len(window)
        piece, text = text[:cut].rstrip(), text[cut:].lstrip("\n ")
        fence = open_fence
        for match in re.finditer(r"^```(\S*)", piece, re.MULTILINE):
            fence = "" if fence else f"```{match.group(1)}"
        body = prefix + piece
        if fence:
            body += "\n```"
        chunks.append(body)
        open_fence = fence
    return chunks
