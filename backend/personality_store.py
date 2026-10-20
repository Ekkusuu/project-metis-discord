"""
Edit personalities in system_prompts.yaml from the dashboard.

A personality is a key under both `text_chat:` and `voice_chat:`. The file is hand-written, with
comments and section banners, so it is edited as text: only the block of the personality being
changed is replaced. Every write is checked by parsing the result before it replaces the file,
and the previous version is kept as system_prompts.yaml.bak.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

SECTIONS = ("text_chat", "voice_chat")
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class PersonalityError(ValueError):
    pass


def _prompts_file() -> Path:
    from backend.prompt_loader import _prompts_file as path
    return path


def _section_range(lines: List[str], section: str) -> Optional[Tuple[int, int]]:
    """(first line inside the section, line after its last block); comments before the next section excluded."""
    start = next((i for i, line in enumerate(lines) if re.match(rf"^{section}:\s*(#.*)?$", line)), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^[A-Za-z_][\w-]*:", lines[i])), len(lines))
    while end > start + 1 and (not lines[end - 1].strip() or lines[end - 1].startswith("#")):
        end -= 1
    return start + 1, end


def _block_range(lines: List[str], section: str, name: str) -> Optional[Tuple[int, int]]:
    """Lines of `  name: ...` and its indented body, without trailing blank lines."""
    span = _section_range(lines, section)
    if not span:
        return None
    first = next((i for i in range(*span) if re.match(rf"^  {re.escape(name)}:", lines[i])), None)
    if first is None:
        return None
    end = first + 1
    while end < span[1] and (not lines[end].strip() or lines[end].startswith("    ")):
        end += 1
    while end > first + 1 and not lines[end - 1].strip():
        end -= 1
    return first, end


def _render(name: str, prompt: str) -> List[str]:
    """A literal block (`|2`), so line breaks are kept exactly as typed."""
    body = [line.rstrip() for line in prompt.strip("\n").splitlines()]
    return [f"  {name}: |2"] + [f"    {line}" if line else "" for line in body]


def raw_prompt(name: str, section: str) -> str:
    """The prompt as written in the file (line breaks kept), for editing."""
    lines = _prompts_file().read_text(encoding="utf-8").splitlines()
    span = _block_range(lines, section, name)
    if not span:
        return ""
    head = lines[span[0]].split(":", 1)[1].strip()
    if head and not head.startswith((">", "|")):
        return str(yaml.safe_load(head) or "")  # one-line `name: "prompt"`
    return "\n".join(line[4:] if line.startswith("    ") else line.strip() for line in lines[span[0] + 1:span[1]])


def _write(text: str, expected: Dict[str, Dict[str, Optional[str]]]) -> None:
    """Parse-check `text`, confirm it holds what was intended, then replace the file (keeping a .bak)."""
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise PersonalityError(f"The edit would break system_prompts.yaml: {e}")
    for section, prompts in expected.items():
        for name, prompt in prompts.items():
            got = (data.get(section) or {}).get(name)
            if (got.strip() if isinstance(got, str) else got) != (prompt.strip() if prompt is not None else None):
                raise PersonalityError(f"Saving {section}.{name} didn't round-trip; nothing was changed.")
    path = _prompts_file()
    path.with_suffix(".yaml.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    from backend.prompt_loader import reload_prompts
    reload_prompts()


def save(name: str, prompts: Dict[str, str]) -> None:
    """Create or replace a personality. `prompts` maps "text_chat"/"voice_chat" to the prompt text."""
    if not NAME.match(name):
        raise PersonalityError("Use 1-32 lowercase letters, digits, - or _ for the name (it's what >personality takes).")
    prompts = {s: "\n".join(line.rstrip() for line in (prompts.get(s) or "").strip("\n").splitlines())
               for s in SECTIONS}
    for section in SECTIONS:
        if not prompts[section].strip():
            raise PersonalityError(f"The {section.replace('_', ' ')} prompt is empty.")
    text = _prompts_file().read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")
    for section in SECTIONS:
        block = _render(name, prompts[section])
        span = _block_range(lines, section, name)
        if span:
            lines[span[0]:span[1]] = block
            continue
        section_span = _section_range(lines, section)
        if section_span is None:  # no such section yet: add it at the end
            lines += ["", f"{section}:"] + block
        else:
            lines[section_span[1]:section_span[1]] = [""] + block
    _write(newline.join(lines), {s: {name: prompts[s]} for s in SECTIONS})


def delete(name: str) -> None:
    text = _prompts_file().read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")
    found = False
    for section in SECTIONS:
        span = _block_range(lines, section, name)
        if span:
            found = True
            end = span[1]
            while end < len(lines) and not lines[end].strip() and end - span[1] < 1:
                end += 1  # and the blank line that separated it from the next one
            del lines[span[0]:end]
    if not found:
        raise PersonalityError(f"No personality called {name}.")
    _write(newline.join(lines), {s: {name: None} for s in SECTIONS})
