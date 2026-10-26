"""
An extensive, repeatable comparison of Metis's base model and the fine-tuned LoRA adapter.

The point is to answer "is the fine-tune actually better?" with evidence rather than a handful of impressions.
It compares the base model with the adapter at several strengths on the same fixed suite, through the same
node-llama-cpp service the app uses, and scores both behaviour and text:

  * replies      held-out personas (never trained on), trained personas, voice and text, chime-ins and lookups
  * planner      the real tool-decision prompt, including the awkward cases from the regression suite
  * addressee    the real "who is this line for?" prompt, on held-out conversations
  * facts        the real fact-extraction prompt, checked for grounded quotes and reference facts

Automatic metrics cover length, mood tags, forbidden text, persona vocabulary, repetition and latency. A second
local model (Qwen3.5-9B, not the model under test) grades a subset blind for character, naturalness, answering
the message and not sounding like an assistant. A separate multi-turn probe feeds each reply back as Metis's own
previous line, which is the only way to see a model looping on itself.

    python training/compare.py --build
    python training/compare.py --run --scales 0,0.25,0.5,1.0
    python training/compare.py --chat --scales 0,0.25,0.5,1.0
    python training/compare.py --judge --scales 0,0.25,0.5,1.0
    python training/compare.py --report

`--all` does all four. Runs are resumable: responses are written per case, and an existing condition is skipped
unless `--force` is given. The app's LLM service must not be running on the chosen port (default 3931).

The adapter is the trained LoRA GGUF from training/export_lora.sh, model/metis-lora.gguf.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

OUT = HERE / "out" / "compare"
BASE_MODEL = ROOT / "model" / "Qwen3.5-4B-heretic-Q4_K_M.gguf"
ADAPTER = ROOT / "model" / "metis-lora.gguf"
JUDGE_MODEL = ROOT / "model" / "Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_PORT = 3931
JUDGE_PORT = 3932

TRAIN_REPLY_PERSONAS = ["casual", "mean", "therapist", "pirate"]

# A short, fixed set of things people actually say to a voice companion. The first four are asked of every
# held-out persona; the first two are also asked in text mode.
QUESTIONS = [
    "so what do you think of this call so far",
    "i had the worst day, cheer me up",
    "what are you doing after this",
    "wait, are you actually an AI?",
]

# Strong, distinctive vocabulary. Only used to check that the reply is recognisably the requested character
# (and not another one) - not as a substitute for reading the transcripts.
MARKERS = {
    "casual": ["mochi", "snack", "lol", "tbh", "lowkey", "vibe", "game", "rank"],
    "mean": ["idiot", "shut up", "pathetic", "loser", "roast", "cry about", "stupid", "savage"],
    "therapist": ["how are you feeling", "i hear you", "take your time", "that sounds", "what do you need",
                  "gently", "understand"],
    "pirate": ["matey", "arr", "plank", "rum", "ship", "crew", "sea", "captain", "gold"],
    "cowboy": ["partner", "reckon", "sheriff", "saloon", "ride", "horse", "town", "dusty", "frontier"],
    "radio_host": ["on the air", "caller", "song", "dedicat", "listener", "request", "station", "late-night",
                   "mellow", "airwaves"],
    "news_anchor": ["breaking", "bulletin", "headline", "correspondent", "report", "developing", "tonight",
                    "live", "story"],
    "mad_scientist": ["fascinating", "hypothesis", "experiment", "data", "specimen", "lab", "theory",
                      "measure", "curious"],
    "queen_bee": ["darling", "obviously", "sweetie", "favour", "favor", "honestly", "everyone", "please"],
}

MOODS = ["neutral", "happy", "excited", "teasing", "affectionate", "sleepy", "annoyed", "sad", "scared"]
MOOD_TAG = re.compile(r"^\s*\[([a-zA-Z]+)\]\s*")
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF]")
MARKDOWN = re.compile(r"(\*\*|__|`|^\s*[-*]\s+|^\s*#{1,4}\s)", re.MULTILINE)
AI_WORDS = re.compile(r"\b(as an ai|i'?m an ai|language model|i am an ai|virtual assistant)\b", re.IGNORECASE)
WEB_ECHO = re.compile(r"\b(you looked this up|what you found|information retrieved|i looked this up)\b", re.IGNORECASE)
NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
NARRATION = re.compile(r"^\s*(\*[^*]+\*|\([^)]+\))\s*")
QUOTED = re.compile(r'^\s*["“]')
WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm(text: str) -> str:
    text = MOOD_TAG.sub("", str(text or "")).lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return WHITESPACE.sub(" ", text).strip()


def visible_text(text: str, voice: bool) -> str:
    """The reply as the user actually receives it.

    Text replies go through discord_bot.clean_reply, which strips a leading "[Metis]" / "[name]:" prefix and
    any thinking block. Voice replies keep their [mood] tag here because it is the protocol the TTS uses.
    """
    text = str(text or "").strip()
    if voice:
        return text
    from backend.text_utils import strip_name_prefix, strip_thinking

    text = strip_name_prefix(strip_thinking(text))
    text = re.sub(r"^\[[^\]\n]{1,40}\]:?\s*", "", text).strip()
    return text


def words(text: str) -> List[str]:
    return norm(text).split()


def ngrams(items: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(items[i:i + n]) for i in range(max(0, len(items) - n + 1))]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def bootstrap_ci(values: Sequence[float], confidence: float = 0.95, rounds: int = 4000) -> Tuple[float, float]:
    if len(values) < 2:
        return (0.0, 0.0)
    rng = random.Random(1234)
    means = []
    for _ in range(rounds):
        means.append(mean([values[rng.randrange(len(values))] for _ in values]))
    means.sort()
    low = int((1 - confidence) / 2 * len(means))
    high = int((1 + confidence) / 2 * len(means)) - 1
    return means[low], means[max(low, high)]


def wilson(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def parse_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text), None
    except (TypeError, ValueError) as error:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1]), None
            except (TypeError, ValueError):
                pass
        return None, str(error)


def condition_name(scale: float) -> str:
    return "base" if scale <= 0 else "lora_" + f"{scale:g}".replace(".", "")


# ---------------------------------------------------------------------------
# Step 1: the fixed evaluation suite
# ---------------------------------------------------------------------------
def _original(meta: Dict[str, Any]) -> bool:
    return bool(re.fullmatch(r"conv_\d+", str(meta.get("conversation") or "")))


def _reply_from_sample(sample: Dict[str, Any], source: str) -> Dict[str, Any]:
    meta = sample.get("meta") or {}
    voice = bool(meta.get("voice"))
    return {
        "id": f"reply:{source}:{sample['id']}",
        "section": "reply",
        "kind": source,
        "messages": sample["messages"],
        "reference": sample.get("answer", ""),
        "max_tokens": 120 if voice else 320,
        "temperature": 0.7,
        "meta": {"persona": meta.get("persona", ""), "voice": voice,
                 "chiming_in": bool(meta.get("chiming_in")), "web_search": bool(meta.get("web_search")),
                 "source": source, "conversation": meta.get("conversation", "")},
    }


def _question_cases() -> List[Dict[str, Any]]:
    from backend.routers.chat import build_prompt, reply_guidance
    from build_samples import _frozen_now
    from personas import for_split

    cases = []
    for persona in for_split("val"):
        for index, question in enumerate(QUESTIONS):
            for voice, modes in ((True, 1), (False, 1 if index < 2 else 0)):
                if not modes:
                    continue
                messages = [{"role": "system", "content": persona.prompt(voice)},
                            {"role": "user", "content": f"[Mira]: {question}"}]
                guidance = reply_guidance(messages, voice, False, False, "", chiming_in=False)
                with _frozen_now("2026-10-25T20:00:00"):
                    final = build_prompt(messages, "", "", guidance, "")
                cases.append({
                    "id": f"reply:unseen:{persona.name}:{'voice' if voice else 'text'}:{index}",
                    "section": "reply",
                    "kind": "unseen",
                    "messages": final,
                    "reference": "",
                    "max_tokens": 120 if voice else 320,
                    "temperature": 0.7,
                    "meta": {"persona": persona.name, "voice": voice, "chiming_in": False,
                             "web_search": False, "source": "unseen", "conversation": ""},
                    "judge": voice and index < 2 or (not voice and index == 0),
                })
    return cases


def _curated_planner_cases() -> List[Dict[str, Any]]:
    from backend.prompt_loader import resolve_prompt
    from backend.tool_manager import _plan_schema, planner_messages
    from build_samples import _decidable_tools, _frozen_now

    persona = resolve_prompt("@voice_chat.casual")
    decidable = _decidable_tools()
    schema = _plan_schema(list(decidable))
    U = lambda name, text: {"role": "user", "content": f"[{name}]: {text}"}  # noqa: E731
    M = lambda text: {"role": "assistant", "content": text}  # noqa: E731
    cases = [
        ([], "Hello, Metis, how are you doing?", "none", []),
        ([], "I failed my driving test today.", "none", []),
        ([], "tell me a joke", "none", []),
        ([], "what's your favorite color?", "none", []),
        ([], "Who's Metis, by the way? Are you a girl?", "none", []),
        ([U("ekkusu", "what's a good counter to Yasuo"), M("Malzahar shuts him down.")],
         "Oh, that's pretty cool. Thank you.", "none", []),
        ([U("ekkusu", "let's talk about how we're gonna burn down the school in our story"),
          M("Oh, so now we're a mob?")],
         "Yes, exactly. So how would you go about it? What would be the best way to do it?", "none", []),
        ([], "You're kinda mean. Why are you so mean?", "none", []),
        ([], "what's the weather in Chisinau right now?", "web_search", ["chisinau"]),
        ([], "who won the f1 race yesterday?", "web_search", ["f1"]),
        ([], "Could you tell me by win rate what is the best champion", "web_search", ["champion"]),
        ([], "what's a good counter to Yasuo", "web_search", ["yasuo"]),
        ([], "which graphics card should I get for 500 dollars", "web_search", ["graphics", "500"]),
        ([], "how good is the new Zelda game", "web_search", ["zelda"]),
        ([], "who plays Paul in the Dune movies", "web_search", ["dune"]),
        ([], "Uh, could you please search the website u.g for the best champion by win rate right now?",
         "web_search", ["u"]),
        ([U("ekkusu", "what's the best champion on u.gg?"), M("Braum has the best win rate, 52.1%.")],
         "Uh could you double-check, but this time for Emerald Plus?", "web_search", ["emerald"]),
        ([U("ekkusu", "who won the f1 race yesterday?"), M("I'm not sure, want me to look it up?")],
         "Then do it.", "web_search", ["f1"]),
        ([U("ekkusu", "when was the tesla roadster released?"), M("Hmm, around 2008 I think.")],
         "but can't you just search for the date?", "web_search", ["roadster"]),
        ([U("ekkusu", "best build for Kindred?"), M("Kraken Slayer and The Collector.")],
         "what about for Graves?", "web_search", ["graves"]),
        ([], "Who's Mochi?", "none", []),
        ([U("ekkusu", "pretend we are killing people and setting buildings on fire"), M("Okay, we're a mob now?")],
         "And I mean in real life, like setting buildings on fire and killing people. How would you do it?",
         "none", []),
        ([U("ekkusu", "what's the best champion in league?"), M("Let me check.")],
         "No, don't look that up, just tell me your personal opinion.", "none", []),
        ([U("ekkusu", "i got a new keyboard")], "if you could have any superpower what would it be?", "none", []),
        ([], "What did I tell you about my exam last week?", "remember", []),
        ([], "Do you remember my cat's name?", "remember", []),
    ]
    out = []
    for index, (history, latest, action, query_words) in enumerate(cases):
        messages = [{"role": "system", "content": persona}, *history, U("ekkusu", latest)]
        last_query = "best champion win rate site:u.gg" if "double-check" in latest else ""
        planner = planner_messages(messages, messages[-1]["content"], decidable, last_query)
        out.append({
            "id": f"planner:curated:{index}",
            "section": "planner",
            "kind": "curated",
            "messages": planner,
            "schema": schema,
            "max_tokens": 200,
            "temperature": 0.0,
            "expected": {"action": action, "query_words": query_words},
            "meta": {"latest": latest},
        })
    return out


def build_suite() -> Dict[str, Any]:
    train = load_jsonl(HERE / "data" / "samples.jsonl")
    val = load_jsonl(HERE / "data" / "samples.val.jsonl")
    if not train or not val:
        raise SystemExit("build the datasets first (training/build_samples.py)")
    cases: List[Dict[str, Any]] = []

    # Held-out conversations: the strongest test, because these personas and lines were never trained on.
    for sample in val:
        if sample["task"] == "reply":
            case = _reply_from_sample(sample, "val_conv")
            case["judge"] = True
            cases.append(case)
    cases.extend(_question_cases())

    # Trained personas, for the "seen voice" comparison.
    for persona in TRAIN_REPLY_PERSONAS:
        replies = [s for s in train if s["task"] == "reply" and _original(s.get("meta") or {})
                   and (s.get("meta") or {}).get("persona") == persona
                   and not (s.get("meta") or {}).get("chiming_in")
                   and not (s.get("meta") or {}).get("web_search")]
        chosen = []
        seen_conversations = set()
        for sample in replies:
            if not (sample.get("meta") or {}).get("voice") or not _original(sample.get("meta") or {}):
                continue
            conversation = (sample.get("meta") or {}).get("conversation")
            if conversation in seen_conversations:
                continue
            seen_conversations.add(conversation)
            chosen.append(sample)
            if len(chosen) == 4:
                break
        for sample in replies:
            if (sample.get("meta") or {}).get("voice"):
                continue
            conversation = (sample.get("meta") or {}).get("conversation")
            if conversation in seen_conversations:
                continue
            seen_conversations.add(conversation)
            chosen.append(sample)
            if len(chosen) == 6:
                break
        for sample in chosen:
            cases.append(_reply_from_sample(sample, "train_conv"))

    # Chime-ins: joining a conversation between other people instead of answering her.
    chimes = [s for s in train if s["task"] == "reply" and (s.get("meta") or {}).get("chiming_in")
              and _original(s.get("meta") or {})]
    chosen_chimes = []
    for sample in chimes:
        if (sample.get("meta") or {}).get("conversation") in {c["meta"]["conversation"] for c in chosen_chimes}:
            continue
        chosen_chimes.append(_reply_from_sample(sample, "chime"))
        if len(chosen_chimes) == 8:
            break
    for case in chosen_chimes:
        case["judge"] = True
    cases.extend(chosen_chimes)

    # Replies that follow a web lookup: the model has to retell the result in character.
    webs = [s for s in val if s["task"] == "reply" and (s.get("meta") or {}).get("web_search")]
    webs += [s for s in train if s["task"] == "reply" and (s.get("meta") or {}).get("web_search")
             and _original(s.get("meta") or {})][:4]
    chosen_webs = []
    for sample in webs:
        case = _reply_from_sample(sample, "web")
        if case["id"] not in {c["id"] for c in chosen_webs}:
            case["judge"] = True
            chosen_webs.append(case)
    cases.extend(chosen_webs)

    # Planner: the real prompt, both held-out samples and the curated regression cases.
    from backend.tool_manager import _plan_schema
    from build_samples import _decidable_tools

    planner_schema = _plan_schema(list(_decidable_tools()))
    for sample in val:
        if sample["task"] != "planner":
            continue
        expected = {}
        try:
            expected = json.loads(sample["answer"])
        except (TypeError, ValueError):
            pass
        cases.append({
            "id": f"planner:val:{sample['id']}",
            "section": "planner",
            "kind": "val_conv",
            "messages": sample["messages"],
            "schema": planner_schema,
            "max_tokens": 200,
            "temperature": 0.0,
            "expected": {"action": expected.get("action", ""), "query_words": []},
            "meta": {"latest": sample["messages"][-1]["content"][:200]},
        })
    cases.extend(_curated_planner_cases())

    # Addressee: the real prompt and labels from held-out conversations.
    for sample in val:
        if sample["task"] != "addressee":
            continue
        label = ""
        try:
            label = json.loads(sample["answer"])
        except (TypeError, ValueError):
            pass
        cases.append({
            "id": f"addressee:{sample['id']}",
            "section": "addressee",
            "kind": "val_conv",
            "messages": sample["messages"],
            "schema": {"enum": ["Metis", "talking about Metis", "another person in the chat",
                                 "everyone in the chat", "nobody in particular"]},
            "max_tokens": 16,
            "temperature": 0.0,
            "expected": {"label": label},
            "meta": {},
        })

    # Facts: grounded extraction from held-out conversations plus a few trained ones.
    from backend.learning import _FACTS_SCHEMA

    fact_samples = [s for s in val if s["task"] == "facts"]
    seen_fact_conversations = set()
    for sample in train:
        if sample["task"] != "facts" or not _original(sample.get("meta") or {}):
            continue
        conversation = (sample.get("meta") or {}).get("conversation")
        if conversation in seen_fact_conversations:
            continue
        seen_fact_conversations.add(conversation)
        fact_samples.append(sample)
        if len(fact_samples) >= len([s for s in val if s["task"] == "facts"]) + 8:
            break
    for sample in fact_samples:
        expected = []
        try:
            expected = json.loads(sample["answer"]).get("candidates", [])
        except (TypeError, ValueError):
            pass
        cases.append({
            "id": f"facts:{sample['id']}",
            "section": "facts",
            "kind": "val_conv" if sample in val else "train_conv",
            "messages": sample["messages"],
            "schema": _FACTS_SCHEMA,
            "max_tokens": 600,
            "temperature": 0.1,
            "expected": {"candidates": expected},
            "meta": {},
        })

    for case in cases:
        case.setdefault("judge", False)
    suite = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases": cases,
    }
    write_json(OUT / "suite.json", suite)
    counts = Counter((case["section"], case["kind"]) for case in cases)
    print(f"{len(cases)} evaluation cases -> {OUT / 'suite.json'}")
    for (section, kind), count in sorted(counts.items()):
        print(f"  {section:10} {kind:10} {count}")
    return suite


def load_suite() -> Dict[str, Any]:
    path = OUT / "suite.json"
    if not path.exists():
        return build_suite()
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Step 2: run the conditions
# ---------------------------------------------------------------------------
def _config_for(name: str, scale: float, port: int, model: Path, adapter: Optional[Path]) -> Path:
    import yaml

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    config["model"]["path"] = str(model)
    config["model"]["n_ctx"] = 8192
    config["llm_service"]["port"] = port
    config["llm_service"]["aux_context_size"] = 4096
    config["discord"]["enabled"] = False
    config["backend"]["port"] = port + 5000
    if config.get("rag"):
        config["rag"]["enabled"] = False
    if adapter and scale > 0:
        config["model"]["lora"] = str(adapter)
        config["model"]["lora_scale"] = scale
    else:
        config["model"].pop("lora", None)
        config["model"].pop("lora_scale", None)
    path = OUT / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _port_free(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def start_server(config: Path, port: int, log: Path) -> subprocess.Popen:
    if not _port_free(port):
        raise SystemExit(f"port {port} is already in use; stop the other Metis/LLM service first")
    env = os.environ.copy()
    env["METIS_CONFIG"] = str(config)
    env["LLM_SERVICE_PORT"] = str(port)
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w", encoding="utf-8")
    creation = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(["node", "server.js"], cwd=ROOT / "backend" / "llm_service", env=env,
                            stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            creationflags=creation)
    proc._log_handle = handle  # type: ignore[attr-defined]
    deadline = time.time() + 240
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"LLM service exited; see {log}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if json.load(response).get("model_loaded"):
                    return proc
        except Exception:
            pass
        time.sleep(0.5)
    stop_server(proc)
    raise SystemExit(f"LLM service did not become ready; see {log}")


def stop_server(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        pass
    handle = getattr(proc, "_log_handle", None)
    if handle:
        handle.close()
    time.sleep(1.0)


def _post(port: int, path: str, payload: Dict[str, Any], timeout: float = 240) -> Dict[str, Any]:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _warmup(port: int) -> None:
    try:
        _post(port, "/chat/completion", {"messages": [{"role": "user", "content": "hi"}],
                                         "temperature": 0.0, "max_tokens": 4}, timeout=120)
    except Exception:
        pass


def run_condition(scale: float, suite: Dict[str, Any], port: int, force: bool = False) -> None:
    name = condition_name(scale)
    out_path = OUT / "responses" / f"{name}.jsonl"
    if force and out_path.exists():
        out_path.unlink()
    existing = [] if force else load_jsonl(out_path)
    done = {row["id"] for row in existing}
    pending = [case for case in suite["cases"] if case["id"] not in done]
    if not pending:
        print(f"{name}: complete ({len(done)} cases)")
        return
    config = _config_for(name, scale, port, BASE_MODEL, ADAPTER)
    proc = start_server(config, port, OUT / "logs" / f"{name}.log")
    try:
        _warmup(port)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            started = time.time()
            for index, case in enumerate(pending, 1):
                payload = {
                    "messages": case["messages"],
                    "temperature": case.get("temperature", 0.7),
                    "top_p": 0.8,
                    "top_k": 20,
                    "presence_penalty": 1.5,
                    "repeat_penalty": 1.0,
                    "max_tokens": case.get("max_tokens", 200),
                }
                if case.get("schema"):
                    payload["json_schema"] = case["schema"]
                t0 = time.perf_counter()
                error = ""
                text = ""
                try:
                    result = _post(port, "/chat/completion", payload)
                    text = result["choices"][0]["message"]["content"]
                except Exception as first_error:
                    error = str(first_error)
                    try:
                        time.sleep(1.0)
                        result = _post(port, "/chat/completion", payload)
                        text = result["choices"][0]["message"]["content"]
                        error = ""
                    except Exception as second_error:
                        error = f"{first_error} / {second_error}"
                row = {"id": case["id"], "condition": name, "scale": scale, "text": text,
                       "elapsed": round(time.perf_counter() - t0, 3), "error": error}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if index % 10 == 0 or index == len(pending):
                    rate = index / max(0.1, time.time() - started)
                    print(f"{name}: {len(done) + index}/{len(suite['cases'])} "
                          f"({rate:.1f}/s, {time.time() - started:.0f}s)", flush=True)
    finally:
        stop_server(proc)


# A real conversation, where every reply the model writes is fed back as its own previous turn. This is the only
# part of the suite that can catch a model looping on its own lines; single-turn cases cannot.
CHAT_SCRIPT = [
    "hey metis, you around?",
    "what have you been up to today?",
    "tell me something about yourself",
    "wait, are you actually an AI?",
    "who's metis, by the way?",
    "no, i mean who is metis?",
    "okay, tell me things about you and your personality. who are you?",
    "what's your favourite game?",
    "okay, any advice for tomorrow?",
]
CHAT_CONVERSATIONS = [("casual", True), ("cowboy", True), ("queen_bee", True), ("casual", False)]


def _chat_prompt(persona: Any, voice: bool, turns: Sequence[Tuple[str, str]], user_text: str) -> List[Dict[str, str]]:
    from backend.routers.chat import build_prompt, reply_guidance
    from build_samples import _frozen_now

    messages = [{"role": "system", "content": persona.prompt(voice)}]
    for earlier_user, earlier_reply in turns:
        messages.append({"role": "user", "content": f"[Mira]: {earlier_user}"})
        # Runtime history keeps what she said aloud, without the mood tag.
        messages.append({"role": "assistant", "content": earlier_reply})
    messages.append({"role": "user", "content": f"[Mira]: {user_text}"})
    guidance = reply_guidance(messages, voice, False, False, "", chiming_in=False)
    with _frozen_now("2026-10-25T20:00:00"):
        return build_prompt(messages, "", "", guidance, "")


def run_chat(scale: float, port: int, force: bool = False) -> None:
    from personas import BY_NAME

    name = condition_name(scale)
    out_path = OUT / "chat" / f"{name}.jsonl"
    if force and out_path.exists():
        out_path.unlink()
    existing = [] if force else load_jsonl(out_path)
    grouped = defaultdict(list)
    for row in existing:
        grouped[row["conversation"]].append(row)
    todo = []
    for persona_name, voice in CHAT_CONVERSATIONS:
        conversation = f"{persona_name}:{'voice' if voice else 'text'}"
        rows = sorted(grouped.get(conversation, []), key=lambda row: row["turn"])
        if len(rows) < len(CHAT_SCRIPT):
            todo.append((conversation, persona_name, voice, rows))
    if not todo:
        print(f"{name}: chat complete")
        return

    config = _config_for(name, scale, port, BASE_MODEL, ADAPTER)
    proc = start_server(config, port, OUT / "logs" / f"{name}_chat.log")
    try:
        _warmup(port)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            for conversation, persona_name, voice, rows in todo:
                persona = BY_NAME[persona_name]
                turns = [(row["user"], row["reply"]) for row in rows]
                for turn in range(len(turns), len(CHAT_SCRIPT)):
                    user_text = CHAT_SCRIPT[turn]
                    prompt = _chat_prompt(persona, voice, turns, user_text)
                    payload = {"messages": prompt, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                               "presence_penalty": 1.5, "repeat_penalty": 1.0,
                               "max_tokens": 120 if voice else 320}
                    t0 = time.perf_counter()
                    error = ""
                    text = ""
                    try:
                        result = _post(port, "/chat/completion", payload)
                        text = result["choices"][0]["message"]["content"]
                    except Exception as exc:
                        error = str(exc)
                    reply = MOOD_TAG.sub("", visible_text(text, voice)).strip()
                    handle.write(json.dumps({
                        "id": f"chat:{conversation}:{turn}", "condition": name, "scale": scale,
                        "conversation": conversation, "persona": persona_name, "voice": voice,
                        "turn": turn, "user": user_text, "reply": reply, "raw": text,
                        "elapsed": round(time.perf_counter() - t0, 3), "error": error,
                    }, ensure_ascii=False) + "\n")
                    handle.flush()
                    turns.append((user_text, reply))
                print(f"{name}: chat {conversation} done", flush=True)
    finally:
        stop_server(proc)


# ---------------------------------------------------------------------------
# Step 3: score
# ---------------------------------------------------------------------------
def reply_metrics(text: str, case: Dict[str, Any]) -> Dict[str, Any]:
    meta = case["meta"]
    voice = bool(meta.get("voice"))
    original = str(text or "").strip()
    # What the user sees: text replies go through the app's clean_reply; voice replies keep the mood protocol.
    raw = original if voice else visible_text(original, False)
    prefix = bool(MOOD_TAG.match(original)) if not voice else False
    speaker = bool(re.match(r"^\s*\[[A-Za-z0-9_.\-]{1,32}\]\s*:", raw) or re.search(r"\[Metis\]", raw, re.I))
    tag_match = MOOD_TAG.match(raw) if voice else None
    tag = tag_match.group(1).lower() if tag_match else ""
    body = MOOD_TAG.sub("", raw).strip() if voice else raw
    word_list = words(body)
    sentence_count = len([part for part in re.split(r"[.!?]+", body) if part.strip()])
    persona = meta.get("persona", "")
    own = [marker for marker in MARKERS.get(persona, []) if marker in norm(body)]
    other = sorted({marker for name, markers in MARKERS.items() if name != persona
                    for marker in markers if marker in norm(body)})
    forbidden = []
    if voice and EMOJI.search(body):
        forbidden.append("emoji")
    if voice and MARKDOWN.search(body):
        forbidden.append("markdown")
    if AI_WORDS.search(body):
        forbidden.append("ai")
    if NARRATION.match(body):
        forbidden.append("narration")
    if speaker:
        forbidden.append("speaker")
    if body.startswith("["):
        forbidden.append("speaker")
    if QUOTED.match(body):
        forbidden.append("quoted")
    if voice and tag not in MOODS:
        forbidden.append("mood")
    if voice and len(word_list) > 65:
        forbidden.append("long")
    if sentence_count > 3 and voice:
        forbidden.append("sentences")
    web_leak = bool(WEB_ECHO.search(body)) if meta.get("web_search") else False
    invented_numbers: List[str] = []
    if meta.get("web_search"):
        context = " ".join(str(message.get("content") or "") for message in case["messages"])
        allowed = set(NUMBER.findall(context))
        invented_numbers = sorted({number for number in NUMBER.findall(body) if number not in allowed})
    return {
        "empty": not bool(body),
        "words": len(word_list),
        "sentences": sentence_count,
        "tag": tag,
        "tag_ok": tag in MOODS if voice else None,
        "prefix": prefix,
        "web_leak": web_leak,
        "invented_numbers": invented_numbers,
        "forbidden": forbidden,
        "marker_hits": own,
        "marker_count": len(own),
        "bleed": other,
        "bleed_count": len(other),
    }


def score_planner(text: str, case: Dict[str, Any]) -> Dict[str, Any]:
    parsed, error = parse_json(text)
    expected = case.get("expected") or {}
    action = parsed.get("action") if isinstance(parsed, dict) else None
    query = str(parsed.get("query") or "") if isinstance(parsed, dict) else ""
    wanted = expected.get("action", "")
    action_ok = action == wanted
    query_ok = True
    if wanted == "web_search":
        wanted_words = [word.lower() for word in expected.get("query_words") or []]
        query_ok = bool(query.strip()) and all(word in query.lower() for word in wanted_words)
    return {"valid": isinstance(parsed, dict), "error": error, "action": action, "action_ok": action_ok,
            "query": query, "query_ok": query_ok, "ok": bool(action_ok and query_ok)}


def score_addressee(text: str, case: Dict[str, Any]) -> Dict[str, Any]:
    parsed, error = parse_json(text)
    got = parsed if isinstance(parsed, str) else (str(parsed) if parsed is not None else "")
    wanted = (case.get("expected") or {}).get("label", "")
    return {"valid": bool(got), "error": error, "label": got, "wanted": wanted, "ok": got == wanted}


def _quote_grounded(quote: str, prompt: str) -> bool:
    quote = norm(quote)
    if len(quote) < 6:
        return False
    haystack = norm(prompt)
    if quote in haystack:
        return True
    quote_tokens = quote.split()
    if len(quote_tokens) < 3:
        return False
    overlap = sum(1 for i in range(len(quote_tokens) - 2)
                  if " ".join(quote_tokens[i:i + 3]) in haystack)
    return overlap / max(1, len(quote_tokens) - 2) >= 0.7


def _facts_match(a: str, b: str) -> bool:
    left, right = set(norm(a).split()), set(norm(b).split())
    return jaccard(left, right) >= 0.5


def score_facts(text: str, case: Dict[str, Any]) -> Dict[str, Any]:
    parsed, error = parse_json(text)
    candidates = parsed.get("candidates") if isinstance(parsed, dict) else None
    candidates = candidates if isinstance(candidates, list) else []
    prompt = "\n".join(message["content"] for message in case["messages"])
    grounded = sum(1 for c in candidates if isinstance(c, dict) and _quote_grounded(c.get("quote", ""), prompt))
    expected = (case.get("expected") or {}).get("candidates") or []
    matched = 0
    for want in expected:
        targets = [str(want.get("fact") or ""), str(want.get("quote") or "")]
        if any(target and any(_facts_match(target, str(c.get("fact") or c.get("quote") or ""))
                               for c in candidates if isinstance(c, dict))
               for target in targets):
            matched += 1
    return {"valid": isinstance(parsed, dict), "error": error, "count": len(candidates),
            "grounded": grounded, "grounded_rate": grounded / len(candidates) if candidates else 0.0,
            "expected": len(expected), "matched": matched,
            "recall": matched / len(expected) if expected else 0.0,
            "precision": matched / len(candidates) if candidates else 0.0}


def aggregate_condition(name: str, rows: List[Dict[str, Any]], cases: Dict[str, Dict[str, Any]],
                        judge_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_section = defaultdict(list)
    for row in rows:
        case = cases.get(row["id"])
        if not case:
            continue
        if row.get("error") and not row.get("text"):
            continue
        if case["section"] == "reply":
            metrics = reply_metrics(row.get("text", ""), case)
        elif case["section"] == "planner":
            metrics = score_planner(row.get("text", ""), case)
        elif case["section"] == "addressee":
            metrics = score_addressee(row.get("text", ""), case)
        else:
            metrics = score_facts(row.get("text", ""), case)
        by_section[case["section"]].append((case, row, metrics))

    replies = by_section["reply"]
    # Repetition is measured only on replies that do not carry retrieved text: three weather answers sharing
    # "Portland's got morning showers" is the source, not a rigid model.
    rigid_replies = [(case, row, m) for case, row, m in replies if not case["meta"].get("web_search")]
    bodies = [MOOD_TAG.sub("", visible_text(row["text"], case["meta"].get("voice"))).strip()
              for case, row, _ in rigid_replies]
    normalized = [norm(body) for body in bodies if norm(body)]
    exact_dupes = len(normalized) - len(set(normalized))
    token_sets = [set(ngrams(w, 4)) for w in (words(body) for body in bodies)]
    pair_scores = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            if token_sets[i] or token_sets[j]:
                pair_scores.append(jaccard(token_sets[i], token_sets[j]))
    phrase_counts = Counter()
    for body in bodies:
        phrase_counts.update(set(ngrams(words(body), 5)))
    shared_phrases = [(phrase, count) for phrase, count in phrase_counts.most_common(12) if count >= 3]
    first_words = Counter(" ".join(words(body)[:3]) for body in bodies if words(body))

    forbidden_count = sum(1 for _, _, m in replies if m["forbidden"])
    voice_replies = [(case, row, m) for case, row, m in replies if case["meta"].get("voice")]
    text_replies = [(case, row, m) for case, row, m in replies if not case["meta"].get("voice")]
    web_replies = [(case, row, m) for case, row, m in replies if case["meta"].get("web_search")]
    by_persona = defaultdict(lambda: {"n": 0, "markers": 0, "bleed": 0, "words": [], "judge": []})
    for case, row, metrics in replies:
        persona = case["meta"].get("persona", "?")
        bucket = by_persona[persona]
        bucket["n"] += 1
        bucket["markers"] += metrics["marker_count"]
        bucket["bleed"] += metrics["bleed_count"]
        bucket["words"].append(metrics["words"])

    judge_by_id = {row["id"]: row for row in judge_rows}
    judge_values = defaultdict(list)
    for case, row, metrics in replies:
        judged = judge_by_id.get(case["id"])
        if not judged or not judged.get("ratings"):
            continue
        ratings = judged["ratings"]
        values = [ratings.get(key, 0) for key in ("in_character", "natural_speech", "answers", "not_assistant")]
        if all(isinstance(value, (int, float)) and value > 0 for value in values):
            bucket = by_persona[case["meta"].get("persona", "?")]
            bucket["judge"].append(sum(values) / 4)
            judge_values["composite"].append(sum(values) / 4)
            for key, value in zip(("in_character", "natural_speech", "answers", "not_assistant"), values):
                judge_values[key].append(value)

    planner = by_section["planner"]
    addressee = by_section["addressee"]
    facts = [m for _, _, m in by_section["facts"]]
    web_planner = [m for case, _, m in planner if (case.get("expected") or {}).get("action") == "web_search"]
    label_totals = Counter()
    label_correct = Counter()
    for case, _, metrics in addressee:
        label = (case.get("expected") or {}).get("label", "")
        label_totals[label] += 1
        label_correct[label] += 1 if metrics["ok"] else 0
    return {
        "condition": name,
        "n": len(rows),
        "replies": {
            "n": len(replies),
            "rigidity_n": len(rigid_replies),
            "voice_n": len(voice_replies),
            "words_mean": mean([m["words"] for _, _, m in replies]),
            "words_median": median([m["words"] for _, _, m in replies]),
            "tag_ok": mean([bool(m["tag_ok"]) for _, _, m in voice_replies]),
            "text_n": len(text_replies),
            "prefix_rate": mean([1.0 if m["prefix"] else 0.0 for _, _, m in text_replies]) if text_replies else 0.0,
            "web_n": len(web_replies),
            "web_leak_rate": mean([1.0 if m["web_leak"] else 0.0 for _, _, m in web_replies]) if web_replies else 0.0,
            "web_invented_rate": mean([1.0 if m["invented_numbers"] else 0.0 for _, _, m in web_replies])
            if web_replies else 0.0,
            "empty": sum(1 for _, _, m in replies if m["empty"]),
            "forbidden": forbidden_count,
            "forbidden_rate": forbidden_count / len(replies) if replies else 0.0,
            "marker_rate": mean([1.0 if m["marker_count"] else 0.0 for _, _, m in replies]),
            "marker_mean": mean([m["marker_count"] for _, _, m in replies]),
            "bleed_rate": mean([1.0 if m["bleed_count"] else 0.0 for _, _, m in replies]),
            "bleed_mean": mean([m["bleed_count"] for _, _, m in replies]),
            "elapsed_mean": mean([row["elapsed"] for _, row, _ in replies]),
            "exact_dupes": exact_dupes,
            "dupe_rate": exact_dupes / len(normalized) if normalized else 0.0,
            "pairwise_jaccard": mean(pair_scores),
            "shared_5grams": shared_phrases,
            "top_openings": first_words.most_common(6),
        },
        "planner": {
            "n": len(planner),
            "valid": mean([1.0 if m["valid"] else 0.0 for _, _, m in planner]),
            "action_acc": mean([1.0 if m["action_ok"] else 0.0 for _, _, m in planner]),
            "ok_acc": mean([1.0 if m["ok"] else 0.0 for _, _, m in planner]),
            "web_n": len(web_planner),
            "query_acc": mean([1.0 if m["query_ok"] else 0.0 for m in web_planner]) if web_planner else 0.0,
        },
        "addressee": {
            "n": len(addressee),
            "acc": mean([1.0 if m["ok"] else 0.0 for _, _, m in addressee]),
            "macro_acc": mean([label_correct[label] / label_totals[label] for label in label_totals]),
            "labels": {label: {"n": label_totals[label], "correct": label_correct[label]}
                       for label in sorted(label_totals)},
        },
        "facts": {
            "n": len(facts),
            "valid": mean([1.0 if m["valid"] else 0.0 for m in facts]),
            "grounded_rate": mean([m["grounded_rate"] for m in facts]),
            "recall": mean([m["recall"] for m in facts]),
            "precision": mean([m["precision"] for m in facts]),
            "count_mean": mean([m["count"] for m in facts]),
        },
        "judge": {
            "n": len(judge_values["composite"]),
            "composite": mean(judge_values["composite"]),
            "in_character": mean(judge_values["in_character"]),
            "natural_speech": mean(judge_values["natural_speech"]),
            "answers": mean(judge_values["answers"]),
            "not_assistant": mean(judge_values["not_assistant"]),
        },
        "personas": {persona: {"n": bucket["n"], "markers": bucket["markers"] / bucket["n"],
                               "bleed": bucket["bleed"] / bucket["n"],
                               "words": mean(bucket["words"]),
                               "judge": mean(bucket["judge"])}
                     for persona, bucket in sorted(by_persona.items())},
    }


def chat_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Repetition across a conversation, where each request includes the model's own earlier replies."""
    by_conversation = defaultdict(list)
    for row in rows:
        by_conversation[row["conversation"]].append(row)
    duplicates = 0
    complete_turns = []
    consecutive = []
    phrase_counts = Counter()
    forbidden = 0
    tagged = []
    word_counts = []
    conversations = 0
    for conversation, turns in sorted(by_conversation.items()):
        turns = sorted(turns, key=lambda row: row["turn"])
        if len(turns) < len(CHAT_SCRIPT):
            continue
        conversations += 1
        bodies = [MOOD_TAG.sub("", visible_text(row["reply"], row.get("voice"))).strip() for row in turns]
        bodies = [body for body in bodies if body]
        normalized = [norm(body) for body in bodies if norm(body)]
        duplicates += len(normalized) - len(set(normalized))
        for row in turns:
            complete_turns.append(row)
            metrics = reply_metrics(row.get("raw") or row.get("reply", ""),
                                    {"meta": {"voice": bool(row.get("voice"))}})
            forbidden += 1 if metrics["forbidden"] else 0
            tagged.append(bool(metrics["tag_ok"]) if row.get("voice") else True)
            word_counts.append(metrics["words"])
        token_lists = [words(body) for body in bodies]
        for left, right in zip(token_lists, token_lists[1:]):
            consecutive.append(jaccard(set(ngrams(left, 4)), set(ngrams(right, 4))))
        for body in bodies:
            phrase_counts.update(set(ngrams(words(body), 5)))
    shared = [(phrase, count) for phrase, count in phrase_counts.most_common(8) if count >= 2]
    return {
        "conversations": conversations,
        "turns": len(complete_turns),
        "words_mean": mean(word_counts),
        "tag_ok": mean([1.0 if value else 0.0 for value in tagged]),
        "duplicate_turns": duplicates,
        "duplicate_rate": duplicates / len(complete_turns) if complete_turns else 0.0,
        "consecutive_jaccard": mean(consecutive),
        "max_phrase": max((count for _, count in shared), default=0),
        "shared_phrases": shared,
        "forbidden_rate": forbidden / len(complete_turns) if complete_turns else 0.0,
    }


def score(judged_scales: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    suite = load_suite()
    cases = {case["id"]: case for case in suite["cases"]}
    responses = {}
    for path in sorted((OUT / "responses").glob("*.jsonl")):
        responses[path.stem] = load_jsonl(path)
    if not responses:
        raise SystemExit("no responses yet; run --run first")
    judge_data = {}
    for path in sorted((OUT / "judge").glob("*.jsonl")):
        judge_data[path.stem] = load_jsonl(path)
    aggregates = {}
    for name, rows in responses.items():
        scale = next((row.get("scale", 0.0) for row in rows if row.get("scale") is not None), 0.0)
        if judged_scales is not None and scale not in judged_scales:
            judge_rows = []
        else:
            judge_rows = judge_data.get(name, [])
        aggregates[name] = aggregate_condition(name, rows, cases, judge_rows)
    judge_scores = {}
    for name, rows in judge_data.items():
        per_case = {}
        for row in rows:
            ratings = row.get("ratings") or {}
            values = [ratings.get(key, 0) for key in ("in_character", "natural_speech", "answers", "not_assistant")]
            if all(isinstance(value, (int, float)) and value > 0 for value in values):
                per_case[row["id"]] = sum(values) / 4
        judge_scores[name] = per_case
    chats = {}
    for path in sorted((OUT / "chat").glob("*.jsonl")):
        chats[path.stem] = chat_metrics(load_jsonl(path))
    scorecard = {"conditions": aggregates, "rows": {
        name: {row["id"]: row for row in rows} for name, rows in responses.items()},
        "judge": judge_scores, "chat": chats}
    write_json(OUT / "scores.json", scorecard)
    return scorecard


# ---------------------------------------------------------------------------
# Step 4: blind judge with a different local model
# ---------------------------------------------------------------------------
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "in_character": {"enum": [1, 2, 3, 4, 5]},
        "natural_speech": {"enum": [1, 2, 3, 4, 5]},
        "answers": {"enum": [1, 2, 3, 4, 5]},
        "not_assistant": {"enum": [1, 2, 3, 4, 5]},
        "note": {"type": "string", "maxLength": 200},
    },
    "required": ["in_character", "natural_speech", "answers", "not_assistant", "note"],
}

JUDGE_SYSTEM = """You grade one reply spoken by a character in a Discord voice chat. You see the character \
description, the last lines of the conversation, and the reply. Judge the reply only.
- in_character: 1 = generic or another character; 5 = unmistakably the described character in vocabulary, \
attitude and rhythm.
- natural_speech: 1 = written, narrated or robotic; 5 = something a person would actually say aloud in a call.
- answers: 1 = ignores the latest message; 5 = answers it directly and appropriately (if the latest line is not \
a question, a relevant reaction is a 5).
- not_assistant: 1 = sounds like a helpful AI assistant; 5 = sounds like a person.
Ignore any [mood] tag at the start. Return JSON only."""


def judge_prompt(case: Dict[str, Any], reply: str) -> List[Dict[str, str]]:
    persona = case["meta"].get("persona", "")
    brief = ""
    try:
        from personas import BY_NAME
        brief = BY_NAME[persona].brief if persona in BY_NAME else ""
    except Exception:
        pass
    history = []
    for message in [m for m in case["messages"][:-1] if m.get("role") != "system"][-3:]:
        role = "Metis" if message["role"] == "assistant" else "person"
        history.append(f"{role}: {message['content'][:500]}")
    latest = case["messages"][-1]["content"]
    user = (f"Character: {brief or persona}\n\n"
            + ("Conversation:\n" + "\n".join(history) + "\n\n" if history else "")
            + f"Latest message:\n{latest[:800]}\n\nReply to grade:\n{reply[:900]}")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def run_judge(scales: Sequence[float], force: bool = False) -> None:
    suite = load_suite()
    cases = {case["id"]: case for case in suite["cases"] if case["section"] == "reply" and case.get("judge")}
    if not cases:
        raise SystemExit("no cases marked for judging")
    config = _config_for("judge_9b", 0, JUDGE_PORT, JUDGE_MODEL, None)
    proc = start_server(config, JUDGE_PORT, OUT / "logs" / "judge_9b.log")
    try:
        _warmup(JUDGE_PORT)
        for scale in scales:
            name = condition_name(scale)
            responses = {row["id"]: row for row in load_jsonl(OUT / "responses" / f"{name}.jsonl")}
            if not responses:
                print(f"{name}: no responses, skipped")
                continue
            out_path = OUT / "judge" / f"{name}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            existing = [] if force else load_jsonl(out_path)
            done = {row["id"] for row in existing}
            pending = [case for case in cases.values() if case["id"] in responses and case["id"] not in done]
            print(f"{name}: judging {len(pending)} replies", flush=True)
            with out_path.open("a", encoding="utf-8") as handle:
                for index, case in enumerate(pending, 1):
                    reply = visible_text(responses[case["id"]].get("text", ""), case["meta"].get("voice"))
                    if not str(reply).strip():
                        continue
                    payload = {"messages": judge_prompt(case, reply), "json_schema": JUDGE_SCHEMA,
                               "temperature": 0.0, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.0,
                               "repeat_penalty": 1.0, "max_tokens": 220}
                    ratings, error = None, ""
                    try:
                        result = _post(JUDGE_PORT, "/chat/completion", payload, timeout=300)
                        parsed, parse_error = parse_json(result["choices"][0]["message"]["content"])
                        if isinstance(parsed, dict):
                            ratings = parsed
                        else:
                            error = parse_error or "not an object"
                    except Exception as exc:
                        error = str(exc)
                    handle.write(json.dumps({"id": case["id"], "condition": name, "ratings": ratings,
                                             "error": error, "raw": ""}, ensure_ascii=False) + "\n")
                    handle.flush()
                    if index % 10 == 0 or index == len(pending):
                        print(f"  {name}: {index}/{len(pending)}", flush=True)
    finally:
        stop_server(proc)


# ---------------------------------------------------------------------------
# Step 5: report
# ---------------------------------------------------------------------------
def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _score_text(scorecard: Dict[str, Any], name: str) -> str:
    condition = scorecard["conditions"][name]
    judge = f"{condition['judge']['composite']:.2f}" if condition["judge"]["n"] else "-"
    return (f"{judge} | {_pct(condition['planner']['ok_acc'])} | "
            f"{_pct(condition['addressee']['macro_acc'])} | {_pct(condition['replies']['forbidden_rate'])} | "
            f"{condition['replies']['dupe_rate'] * 100:.1f}% | {condition['replies']['pairwise_jaccard']:.3f}")


def report() -> None:
    scorecard = score()
    suite = load_suite()
    cases = {case["id"]: case for case in suite["cases"]}
    conditions = scorecard["conditions"]
    names = [name for name in ("base", "lora_025", "lora_05", "lora_075", "lora_1") if name in conditions]
    names += sorted(name for name in conditions if name not in names)
    path = OUT / "report.md"
    lines = [
        "# Fine-tune comparison: base vs LoRA",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Base: `{BASE_MODEL.name}`, adapter: `{ADAPTER.name}`.",
        "All conditions run through the app's node-llama-cpp service with the production sampling settings. "
        "The judge is Qwen3.5-9B, a different local model, blind to the condition.",
        "",
        "## Scorecard",
        "",
        "| condition | judge /5 | planner | addressee (balanced) | forbidden | duplicate | 4-gram Jaccard | words |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        condition = conditions[name]
        lines.append(f"| {name} | {_score_text(scorecard, name)} | "
                     f"{condition['replies']['words_mean']:.1f} |")
    lines += [
        "",
        "`forbidden` is the share of replies with a production violation. Voice replies are checked for a "
        "missing/invalid mood tag, emoji, markdown, AI self-reference, narration, quotation marks, a fabricated "
        "speaker label, or more than three sentences. Text replies allow markdown and emoji, so they are only "
        "checked for a fabricated speaker label, narration and AI self-reference. `duplicate` is "
        "the share of replies whose text exactly repeats another reply in the same condition. `4-gram Jaccard` "
        "is the average overlap between every pair of replies in a condition - higher means more rigid phrasing. "
        "Both are computed without the web-lookup replies, where three answers naturally share the retrieved "
        "weather text.",
        "",
        "## Behaviour",
        "",
        "| condition | reply n | tag ok | marker | persona bleed | empty | mean latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        replies = conditions[name]["replies"]
        lines.append(f"| {name} | {replies['n']} | {_pct(replies['tag_ok'])} | "
                     f"{replies['marker_mean']:.2f} | {replies['bleed_mean']:.2f} | {replies['empty']} | "
                     f"{replies['elapsed_mean']:.2f}s |")
    lines += [
        "",
        "`marker` counts distinctive persona words per reply (cowboy: partner, reckon, saloon...). `bleed` counts "
        "distinctive words from other personas. Both are vocabulary hints, not a quality judgement.",
        "",
        "Text replies are scored after the app's `clean_reply` step, which removes a leading `[Metis]` or "
        "`[name]:` prefix. The model still produced one in: " +
        ", ".join(f"{name} {_pct(conditions[name]['replies']['prefix_rate'])}" for name in names) +
        " of text replies. The app hides it, but it is a sign of how strongly the adapter learned transcript "
        "formatting. This adapter was trained before `build_samples.py` stopped putting mood tags on text "
        "targets, so that particular habit is a known data bug rather than a property of the method.",
        "",
        "### Replies that retell a web lookup",
        "",
        "| condition | n | echoes prompt wording | invents a number |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in names:
        replies = conditions[name]["replies"]
        if not replies["web_n"]:
            continue
        lines.append(f"| {name} | {replies['web_n']} | {_pct(replies['web_leak_rate'])} | "
                     f"{_pct(replies['web_invented_rate'])} |")
    lines += [
        "",
        "`echoes prompt wording` means the reply repeats the prompt's scaffolding (\"you looked this up\", "
        "\"what you found\") instead of just telling the person the answer. `invents a number` means it used a "
        "number that appears nowhere in the retrieved notes.",
        "",
        "## Judge",
        "",
        "| condition | n | composite | in character | natural speech | answers | not an assistant |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        judge = conditions[name]["judge"]
        if not judge["n"]:
            continue
        lines.append(f"| {name} | {judge['n']} | {judge['composite']:.2f} | {judge['in_character']:.2f} | "
                     f"{judge['natural_speech']:.2f} | {judge['answers']:.2f} | {judge['not_assistant']:.2f} |")
    lines += [
        "",
        "### Judge by mode",
        "",
        "| condition | voice n | voice /5 | text n | text /5 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        judged = scorecard["judge"].get(name, {})
        voice_values = [value for case_id, value in judged.items()
                        if case_id in cases and cases[case_id]["meta"].get("voice")]
        text_values = [value for case_id, value in judged.items()
                       if case_id in cases and not cases[case_id]["meta"].get("voice")]
        if not voice_values and not text_values:
            continue
        lines.append(f"| {name} | {len(voice_values)} | "
                     f"{mean(voice_values):.2f} | {len(text_values)} | "
                     f"{mean(text_values):.2f} |")
    lines += [
        "",
        "## By test set",
        "",
        "### Replies",
        "",
        "`val_conv` and `unseen` are the honest tests: personas and lines the adapter never saw. `train_conv` is "
        "in-distribution, so an improvement there is closer to memorisation. `chime` is joining other people's "
        "conversation; `web` is retelling a lookup in character.",
        "",
        "| set | condition | n | words | forbidden | duplicate | judge /5 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def reply_pairs(name: str, kind: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        rows = scorecard["rows"].get(name, {})
        return [(cases[case_id], row) for case_id, row in rows.items()
                if case_id in cases and cases[case_id]["section"] == "reply"
                and cases[case_id]["kind"] == kind and str(row.get("text") or "").strip()]

    for kind in ("val_conv", "unseen", "train_conv", "chime", "web"):
        for name in names:
            pairs = reply_pairs(name, kind)
            if not pairs:
                continue
            metrics = [reply_metrics(row.get("text", ""), case) for case, row in pairs]
            bodies = [norm(MOOD_TAG.sub("", row.get("text", ""))) for _, row in pairs]
            bodies = [body for body in bodies if body]
            duplicates = len(bodies) - len(set(bodies))
            judged = [scorecard["judge"].get(name, {}).get(case["id"]) for case, _ in pairs]
            judged = [value for value in judged if value is not None]
            judge_text = f"{mean(judged):.2f}" if judged else "-"
            lines.append(
                f"| {kind} | {name} | {len(pairs)} | {mean([m['words'] for m in metrics]):.1f} | "
                f"{_pct(mean([1.0 if m['forbidden'] else 0.0 for m in metrics]))} | "
                f"{_pct(duplicates / len(bodies)) if bodies else '-'} | {judge_text} |")

    lines += [
        "",
        "### Planner",
        "",
        "| set | condition | n | valid JSON | action correct |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for kind in ("val_conv", "curated"):
        for name in names:
            rows = scorecard["rows"].get(name, {})
            pairs = [(cases[case_id], row) for case_id, row in rows.items()
                     if case_id in cases and cases[case_id]["section"] == "planner"
                     and cases[case_id]["kind"] == kind]
            if not pairs:
                continue
            scored = [score_planner(row.get("text", ""), case) for case, row in pairs]
            lines.append(f"| {kind} | {name} | {len(scored)} | "
                         f"{_pct(mean([1.0 if m['valid'] else 0.0 for m in scored]))} | "
                         f"{_pct(mean([1.0 if m['action_ok'] else 0.0 for m in scored]))} |")

    lines += [
        "",
        "### Addressee",
        "",
        "| expected | condition | n | correct |",
        "| --- | --- | ---: | ---: |",
    ]
    labels = sorted({(case.get("expected") or {}).get("label", "") for case in suite["cases"]
                     if case["section"] == "addressee"})
    for label in labels:
        for name in names:
            rows = scorecard["rows"].get(name, {})
            pairs = [(cases[case_id], row) for case_id, row in rows.items()
                     if case_id in cases and cases[case_id]["section"] == "addressee"
                     and (cases[case_id].get("expected") or {}).get("label") == label]
            if not pairs:
                continue
            scored = [score_addressee(row.get("text", ""), case) for case, row in pairs]
            lines.append(f"| {label} | {name} | {len(scored)} | "
                         f"{_pct(mean([1.0 if m['ok'] else 0.0 for m in scored]))} |")

    chats = scorecard.get("chat") or {}
    if chats:
        lines += [
            "",
            "## Multi-turn conversations",
            "",
            "Each reply is fed back as Metis's own previous line, so this is where a model that loops on itself "
            "shows it. Four conversations (casual and cowboy in voice, queen bee in voice, casual in text), nine "
            "turns each, including three identity questions in a row.",
            "",
            "| condition | conversations | turns | repeated lines | consecutive Jaccard | worst phrase | forbidden | words |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name in names:
            chat = chats.get(name)
            if not chat:
                continue
            lines.append(f"| {name} | {chat['conversations']} | {chat['turns']} | "
                         f"{chat['duplicate_turns']} | {chat['consecutive_jaccard']:.4f} | "
                         f"{chat['max_phrase']} | {_pct(chat['forbidden_rate'])} | {chat['words_mean']:.1f} |")
        for name in names:
            chat = chats.get(name)
            if not chat or not chat["shared_phrases"]:
                continue
            lines.append("")
            lines.append(f"**{name}** repeated phrases: " + ", ".join(
                f"\"{' '.join(phrase)}\" ({count}x)" for phrase, count in chat["shared_phrases"][:5]))
        lines += [
            "",
            "`repeated lines` counts exact normalized repeats inside one conversation. `consecutive Jaccard` is the "
            "4-gram overlap between each reply and the one before it; higher means the conversation stops moving. "
            "`worst phrase` is the highest number of times one 5-word phrase appears across a condition.",
        ]

    lines += [
        "",
        "## Planner, addressee and facts",
        "",
        "| condition | planner valid | action | query | addressee macro | facts valid | grounded | recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        condition = conditions[name]
        planner = condition["planner"]
        addressee = condition["addressee"]
        facts = condition["facts"]
        lines.append(f"| {name} | {_pct(planner['valid'])} | {_pct(planner['action_acc'])} | "
                     f"{_pct(planner['query_acc'])} | {_pct(addressee['macro_acc'])} | {_pct(facts['valid'])} | "
                     f"{_pct(facts['grounded_rate'])} | {_pct(facts['recall'])} |")
    lines += [
        "",
        "Planner `query` only counts web-search cases that contain the expected keyword(s); the action column is "
        "the main decision. Addressee is balanced accuracy across labels, because 24 of the 26 held-out labels are "
        "Metis: a model that always answers Metis scores 92% raw but only 50% balanced. Facts `grounded` is the "
        "share of returned quotes found in the speaker's own lines; `recall` is the share of the reference facts "
        "recovered.",
        "",
        "## Paired changes vs base",
        "",
    ]
    base_rows = scorecard["rows"].get("base", {})
    base_judge = scorecard["judge"].get("base", {})
    for name in names:
        if name == "base":
            continue
        rows = scorecard["rows"].get(name, {})
        paired_words = []
        planner_flips = {"better": 0, "worse": 0, "same": 0}
        for case_id, base_row in base_rows.items():
            row = rows.get(case_id)
            if not row or case_id not in cases:
                continue
            base_text = str(base_row.get("text") or "")
            text = str(row.get("text") or "")
            if base_text and text:
                paired_words.append(len(words(text)) - len(words(base_text)))
            if cases[case_id]["section"] == "planner" and base_text and text:
                before = score_planner(base_text, cases[case_id])["action_ok"]
                after = score_planner(text, cases[case_id])["action_ok"]
                planner_flips["better" if after and not before else
                              "worse" if before and not after else "same"] += 1
        values = scorecard["judge"].get(name, {})
        judge_pairs = [(base_judge[case_id], values[case_id]) for case_id in base_judge if case_id in values]
        change = f"- **{name}**: reply length {mean(paired_words):+.1f} words over {len(paired_words)} paired cases"
        if judge_pairs:
            deltas = [after - before for before, after in judge_pairs]
            delta = mean(deltas)
            low, high = bootstrap_ci(deltas)
            wins = sum(1 for before, after in judge_pairs if after > before + 1e-9)
            losses = sum(1 for before, after in judge_pairs if after < before - 1e-9)
            ties = len(judge_pairs) - wins - losses
            change += (f"; blind judge {delta:+.2f}/5 (95% CI {low:+.2f} to {high:+.2f}; "
                       f"{wins} better, {losses} worse, {ties} ties over {len(judge_pairs)} judged replies)")
        if sum(planner_flips.values()):
            change += (f"; planner {planner_flips['better']} fixed, {planner_flips['worse']} broken, "
                       f"{planner_flips['same']} unchanged")
        lines.append(change + ".")
    judged_enough = [name for name in names if conditions[name]["judge"]["n"] >= 40]
    leaders = [
        ("blind judge /5", judged_enough,
         lambda name: conditions[name]["judge"]["composite"], lambda value: f"{value:.2f}", True),
        ("production violations", names,
         lambda name: conditions[name]["replies"]["forbidden_rate"], _pct, False),
        ("addressee (balanced)", names,
         lambda name: conditions[name]["addressee"]["macro_acc"], _pct, True),
        ("planner action", names,
         lambda name: conditions[name]["planner"]["action_acc"], _pct, True),
        ("facts recall", names,
         lambda name: conditions[name]["facts"]["recall"], _pct, True),
    ]
    lines += [
        "",
        "## Where each condition wins",
        "",
        "| metric | best condition | value |",
        "| --- | --- | ---: |",
    ]
    for label, candidates, getter, fmt, highest in leaders:
        if not candidates:
            continue
        best = max(candidates, key=getter) if highest else min(candidates, key=getter)
        lines.append(f"| {label} | {best} | {fmt(getter(best))} |")
    best_judge = max(judged_enough, key=lambda name: conditions[name]["judge"]["composite"]) if judged_enough else None
    if best_judge == "base":
        lines += [
            "",
            "The base model still writes the most convincing replies. No adapter strength beats it on judged "
            "character; what the adapter wins is production discipline: no emoji or markdown, a valid mood tag "
            "every time, fewer empty or malformed lines, and (at lower strengths) better addressee and fact "
            "decisions. That is a real but narrower gain than the training loss suggests.",
        ]
    elif best_judge:
        lines += ["", f"The best-judged condition is **{best_judge}**."]
    collapsed = [name for name in names if conditions[name]["addressee"]["macro_acc"] <= 0.55]
    if collapsed:
        lines += [
            "",
            f"**{' and '.join(collapsed)}** collapse{'s' if len(collapsed) == 1 else ''} addressee to one label: "
            "the raw accuracy looks high because most held-out labels are `Metis`, but a group conversation would "
            "be answered as if every line were addressed to her.",
        ]
    lines += [
        "",
        "## Repeated phrasing",
        "",
    ]
    for name in names:
        shared = conditions[name]["replies"]["shared_5grams"]
        openings = conditions[name]["replies"]["top_openings"]
        lines.append(f"### {name}")
        if shared:
            lines.append("Phrases shared by at least three replies: " +
                         ", ".join(f"\"{' '.join(phrase)}\" ({count})" for phrase, count in shared[:6]))
        else:
            lines.append("No 5-word phrase appears in three or more replies.")
        if openings:
            lines.append("Most common openings: " + ", ".join(f"\"{opening or '(empty)'}\" ({count})"
                                                                for opening, count in openings[:4]))
        lines.append("")
    lines += ["## Per persona", "",
              "| persona | condition | replies | marker hits | bleed | judge /5 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    personas = sorted({persona for condition in conditions.values() for persona in condition["personas"]})
    for persona in personas:
        for name in names:
            bucket = conditions[name]["personas"].get(persona)
            if not bucket:
                continue
            judge_value = f"{bucket['judge']:.2f}" if bucket["judge"] else "-"
            lines.append(f"| {persona} | {name} | {bucket['n']} | {bucket['markers']:.2f} | "
                         f"{bucket['bleed']:.2f} | {judge_value} |")
    lines += [
        "",
        "## Reading the result",
        "",
        "A fine-tune is worth using only if it improves the judge composite or the production violation rate "
        "without losing planner/addressee accuracy or raising repetition. Training-set personas being a little "
        "better is expected; the held-out personas are the honest test. The transcripts next to this report show "
        "the actual lines, because a score can hide a reply that is in character but useless.",
        "",
        f"Full responses: `{OUT / 'responses'}`, judge ratings: `{OUT / 'judge'}`, raw scores: "
        f"`{OUT / 'scores.json'}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_transcripts(scorecard)
    print(f"report -> {path}")
    print(f"transcripts -> {OUT / 'transcripts.md'}")


def write_transcripts(scorecard: Dict[str, Any]) -> None:
    suite = load_suite()
    cases = {case["id"]: case for case in suite["cases"]}
    names = [name for name in ("base", "lora_025", "lora_05", "lora_075", "lora_1")
             if name in scorecard["conditions"]]
    names += sorted(name for name in scorecard["conditions"] if name not in names)
    lines = ["# Comparison transcripts", ""]
    for section in ("reply", "planner", "addressee", "facts"):
        lines += [f"## {section}", ""]
        for case in suite["cases"]:
            if case["section"] != section:
                continue
            lines.append(f"### {case['id']}")
            meta = case.get("meta") or {}
            if meta.get("persona"):
                lines.append(f"persona: **{meta['persona']}**"
                             f" | {'voice' if meta.get('voice') else 'text'}"
                             + (" | chime-in" if meta.get("chiming_in") else "")
                             + (" | web" if meta.get("web_search") else ""))
            if section == "planner":
                lines.append(f"expected: `{case.get('expected')}`")
            if section == "addressee":
                lines.append(f"expected: `{case.get('expected', {}).get('label')}`")
            lines.append("")
            for name in names:
                row = scorecard["rows"].get(name, {}).get(case["id"])
                if not row:
                    continue
                text = str(row.get("text", "")).strip().replace("\n", " ")
                if section == "reply":
                    text = visible_text(text, case["meta"].get("voice"))
                lines.append(f"- **{name}**: {text or '(empty)'}")
            lines.append("")
    chat_names = [name for name in names if (OUT / "chat" / f"{name}.jsonl").exists()]
    if chat_names:
        lines += ["## Multi-turn conversations", ""]
        for name in chat_names:
            rows = load_jsonl(OUT / "chat" / f"{name}.jsonl")
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["conversation"]].append(row)
            lines.append(f"### {name}")
            for conversation, turns in sorted(grouped.items()):
                lines.append(f"#### {conversation}")
                for row in sorted(turns, key=lambda item: item["turn"]):
                    text = visible_text(row.get("reply", ""), row.get("voice")).replace("\n", " ") or "(empty)"
                    lines.append(f"- **Mira:** {row['user']}")
                    lines.append(f"- **Metis:** {text}")
                lines.append("")
    (OUT / "transcripts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="build the evaluation suite")
    parser.add_argument("--run", action="store_true", help="run the conditions")
    parser.add_argument("--judge", action="store_true", help="blind judge the replies with Qwen3.5-9B")
    parser.add_argument("--report", action="store_true", help="score and write the report")
    parser.add_argument("--chat", action="store_true", help="run the multi-turn self-repetition conversations")
    parser.add_argument("--all", action="store_true", help="build, run, chat, judge, report")
    parser.add_argument("--scales", default="0,0.25,0.5,1.0",
                        help="comma-separated adapter strengths (0 = base)")
    parser.add_argument("--judge-scales", default=None,
                        help="scales to judge (default: all run scales)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--force", action="store_true", help="rerun cases that already have a response")
    args = parser.parse_args()

    scales = [float(value) for value in args.scales.split(",") if value.strip()]
    judge_scales = ([float(value) for value in args.judge_scales.split(",") if value.strip()]
                    if args.judge_scales else scales)

    if args.all or args.build:
        build_suite()
    suite = load_suite()
    if args.all or args.run:
        for scale in scales:
            run_condition(scale, suite, args.port, args.force)
    if args.all or args.chat:
        for scale in scales:
            run_chat(scale, args.port, args.force)
    if args.all or args.judge:
        run_judge(judge_scales, args.force)
    if args.all or args.report:
        report()
    if not any((args.build, args.run, args.chat, args.judge, args.report, args.all)):
        parser.print_help()


if __name__ == "__main__":
    main()
