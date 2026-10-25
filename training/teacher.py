"""
The teacher model for making training data: DeepSeek through the OpenCode CLI (`opencode run`).

Claude can't be used for this (Anthropic's terms don't allow training chatbots on its outputs); DeepSeek's
outputs can be used for training.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

MODEL = "opencode/deepseek-v4.1-flash-free"


def ask(prompt: str, model: str = MODEL, timeout: float = 240) -> str:
    """One prompt in, the model's text out. Runs in an empty folder so the agent has no project to look at."""
    empty = tempfile.mkdtemp(prefix="metis_teacher_")
    try:
        # The prompt goes in an attached file: passed as an argument through the Windows command shim, a
        # multi-line prompt arrived cut off after its first line.
        prompt_file = Path(empty) / "instructions.md"
        prompt_file.write_text(prompt, encoding="utf-8")
        # No stdin: with an inherited one the CLI waited for input forever.
        result = subprocess.run([shutil.which("opencode") or "opencode", "run", "-m", model, "-f", str(prompt_file),
                                 "Follow the instructions in the attached file exactly. Reply with only what they "
                                 "ask for."],
                                cwd=empty, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                encoding="utf-8", timeout=timeout)
    finally:
        # TemporaryDirectory's cleanup recurses forever on Windows when the CLI still holds a file; leftovers in
        # %TEMP% are harmless, a lost teacher answer is not.
        shutil.rmtree(empty, ignore_errors=True)
    if result.returncode != 0:
        raise RuntimeError(f"opencode failed ({result.returncode}): {result.stderr[-400:]}")
    # The CLI prints a header line ("> build · model") before the answer.
    lines = [line for line in result.stdout.splitlines() if not re.match(r"^\s*(\x1b\[[0-9;]*m)*\s*>\s*\w+ ·", line)]
    return re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(lines)).strip()


def ask_json(prompt: str, model: str = MODEL, tries: int = 2) -> Optional[Any]:
    """Like ask(), parsed as JSON (code fences and text around the JSON are ignored). None if it never parses."""
    for _ in range(tries):
        text = ask(prompt, model)
        match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


if __name__ == "__main__":
    print(ask_json('Output only JSON: {"ok": true}'))
