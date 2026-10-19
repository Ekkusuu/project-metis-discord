"""
Metis launcher: runs everything from this one window, shows a tidy status, and shuts it all
down cleanly on Ctrl+C (or when the window is closed).

    python metis.py             start Metis (builds the dashboard first if it changed)
    python metis.py --open      ... and open the dashboard in your browser
    python metis.py --dev       also run the dashboard's live-reload dev server (for UI work)
    python metis.py --verbose   show every log line, not just the important ones
    python metis.py --stop      stop a Metis that was left running (e.g. after a crash)

Full logs are written to logs/llm.log and logs/backend.log (logs/dashboard.log with --dev).
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import json
import webbrowser
from pathlib import Path
from typing import Callable, Deque, List, Optional

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
WINDOWS = os.name == "nt"

# --------------------------------------------------------------------------- console output
if WINDOWS:
    os.system("")  # turn on ANSI colours in the Windows console
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

_COLOUR = sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text


MINT, DIM, BOLD, RED, YELLOW, LAVENDER = "38;2;168;230;207", "2", "1", "38;2;255;123;134", "38;2;243;199;107", "38;2;184;169;255"
_print_lock = threading.Lock()


def say(text: str = "") -> None:
    with _print_lock:
        print(text, flush=True)


def step(text: str) -> None:
    say(f"  {_c(DIM, '•')} {text}")


def ok(text: str) -> None:
    say(f"  {_c(MINT, '✓')} {text}")


def warn(text: str) -> None:
    say(f"  {_c(YELLOW, '!')} {text}")


def fail(text: str) -> None:
    say(f"  {_c(RED, '✗')} {text}")


# Lines from the services that are only noise in the main window (they still go to logs/).
_NOISE = [
    re.compile(p) for p in (
        r"^INFO:\s+127\.0\.0\.1:\d+ - ",          # uvicorn access log
        r"^INFO:\s+(Started server process|Waiting for application|Application startup complete|"
        r"Uvicorn running|Shutting down|Waiting for connections|Finished server process|"
        r"Application shutdown complete)",
        r"^=+$", r"^-+$", r"^\s*$",
        r"^\[NeMo [IW]", r"^\s{4}", r"^Model load options",
        r"^(Initializing node-llama-cpp|Loading model from|Checking LLM service|Indexing knowledge base)",
        r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d INFO\s+discord\.(client|gateway|voice_state|ext\.voice_recv)",
        r"^> ", r"^\s*VITE v", r"^\s*➜", r"press h \+ enter",
        r"^⏱️|^\s+(Transcription|Judgment|Tool|RAG|LLM|TTS|⚡|Total)\b",
    )
]


def is_noise(line: str) -> bool:
    return any(p.search(line) for p in _NOISE)


# --------------------------------------------------------------------------- config
def load_config() -> dict:
    sys.path.insert(0, str(ROOT))
    from backend.config import get_config

    return get_config()


def http_json(url: str, timeout: float = 2.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())
    except Exception:
        return None


# --------------------------------------------------------------------------- services
class Service:
    """A child process whose output goes to logs/<name>.log and (filtered) to this window."""

    def __init__(self, name: str, label: str, colour: str, cmd: List[str], cwd: Path, env: dict, verbose: bool):
        self.name, self.label, self.colour = name, label, colour
        self.cmd, self.cwd, self.env, self.verbose = cmd, cwd, env, verbose
        self.proc: Optional[subprocess.Popen] = None
        self.tail: Deque[str] = collections.deque(maxlen=40)
        self.show_output = False  # quiet while starting; errors are shown from the tail if it fails
        self.log_path = LOGS / f"{name}.log"

    def start(self) -> None:
        LOGS.mkdir(exist_ok=True)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if WINDOWS else 0
        self.proc = subprocess.Popen(
            self.cmd, cwd=self.cwd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=flags, start_new_session=not WINDOWS,
        )
        threading.Thread(target=self._pump, daemon=True, name=f"{self.name}-output").start()

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        with open(self.log_path, "a", encoding="utf-8") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            for raw in self.proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                log.write(line + "\n")
                log.flush()
                self.tail.append(line)
                if self.show_output and line and (self.verbose or not is_noise(line)):
                    say(f"  {_c(self.colour, self.label)} {line}")

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout: float) -> None:
        """Ask nicely (Ctrl+Break = graceful shutdown), then force the whole process tree."""
        if not self.running:
            return
        try:
            self.proc.send_signal(signal.CTRL_BREAK_EVENT if WINDOWS else signal.SIGINT)
            self.proc.wait(timeout=timeout)
        except Exception:
            kill_tree(self.proc.pid)

    def show_tail(self, lines: int = 15) -> None:
        say(_c(DIM, f"    last lines of {self.log_path.relative_to(ROOT)}:"))
        for line in list(self.tail)[-lines:]:
            say(_c(DIM, f"    {line}"))


def kill_tree(pid: int) -> None:
    try:
        import psutil

        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except Exception:
        if WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


# --------------------------------------------------------------------------- setup steps
def find(exe: str) -> Optional[str]:
    return shutil.which(exe) or shutil.which(exe + ".cmd")


def run_quiet(cmd: List[str], cwd: Path, log_name: str, env: Optional[dict] = None) -> bool:
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / log_name, "w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return result.returncode == 0


def ensure_node_modules(npm: str) -> bool:
    for folder in (ROOT / "backend" / "llm_service", ROOT / "frontend"):
        if not (folder / "node_modules").exists():
            step(f"Installing dependencies in {folder.relative_to(ROOT)} (first run)...")
            if not run_quiet([npm, "install"], folder, f"npm-install-{folder.name}.log"):
                fail(f"npm install failed - see logs/npm-install-{folder.name}.log")
                return False
    return True


def dashboard_is_stale() -> bool:
    dist = ROOT / "frontend" / "dist" / "index.html"
    if not dist.exists():
        return True
    built = dist.stat().st_mtime
    sources = [ROOT / "frontend" / p for p in ("index.html", "package.json", "vite.config.ts")]
    for folder in ("src", "public"):
        sources += [p for p in (ROOT / "frontend" / folder).rglob("*") if p.is_file()]
    return any(p.exists() and p.stat().st_mtime > built for p in sources)


def port_free(port: int) -> bool:
    from backend.config import port_in_use

    return not port_in_use("127.0.0.1", port)


def wait_until(check: Callable[[], bool], service: Service, timeout: float, what: str) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if check():
            return True
        if not service.running:
            fail(f"{what} stopped while starting.")
            service.show_tail()
            return False
        time.sleep(0.5)
    fail(f"{what} did not start within {int(timeout)} s.")
    service.show_tail()
    return False


# --------------------------------------------------------------------------- stop a stray Metis
def stop_running(cfg: dict) -> int:
    import psutil

    ports = {int(cfg["llm_service"]["port"]), int(cfg["backend"]["port"]), 5173}
    found = False
    for conn in psutil.net_connections(kind="tcp"):
        if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port in ports and conn.pid:
            try:
                name = " ".join(psutil.Process(conn.pid).cmdline())
            except Exception:
                name = "?"
            if conn.laddr.port == 5173 and "vite" not in name:
                continue
            say(f"  stopping {name[:90]} (port {conn.laddr.port})")
            kill_tree(conn.pid)
            found = True
    ok("Metis stopped.") if found else step("Nothing to stop - Metis isn't running.")
    return 0


# --------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description="Start Metis (LLM service + backend/Discord bot + dashboard).")
    parser.add_argument("--open", action="store_true", help="open the dashboard in the browser once ready")
    parser.add_argument("--dev", action="store_true", help="run the dashboard's live-reload dev server too")
    parser.add_argument("--verbose", action="store_true", help="show every log line")
    parser.add_argument("--stop", action="store_true", help="stop a Metis that was left running")
    args = parser.parse_args()

    if WINDOWS:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW("Metis")

    cfg = load_config()
    if args.stop:
        return stop_running(cfg)

    llm_port, backend_port = int(cfg["llm_service"]["port"]), int(cfg["backend"]["port"])
    model_name = Path(cfg["model"]["path"]).stem
    say()
    say(f"  {_c(MINT + ';' + BOLD, 'Metis')}  {_c(DIM, 'local AI companion for Discord')}")
    say()

    # 1. Setup checks
    node, npm = find("node"), find("npm")
    if not node or not npm:
        fail("Node.js is not installed (needed for the LLM service). Get it from https://nodejs.org")
        return 1
    if not ensure_node_modules(npm):
        return 1
    for port, setting in ((llm_port, "llm_service.port"), (backend_port, "backend.port")):
        if not port_free(port):
            fail(f"Port {port} is already in use - another program, or Metis is already running.")
            say(_c(DIM, f"    Run 'stop.bat' if it's a leftover Metis, or change {setting} in config.yaml."))
            return 1
    if not (ROOT / ".env").exists() and cfg.get("discord", {}).get("enabled", True):
        warn("No .env file: the Discord bot won't connect. Copy .env.example to .env and add your token.")

    # 2. Dashboard
    if not args.dev:
        if dashboard_is_stale():
            step("Building the dashboard...")
            if not run_quiet([npm, "run", "build"], ROOT / "frontend", "dashboard-build.log"):
                warn("Dashboard build failed (see logs/dashboard-build.log); Metis will run without it.")
            else:
                ok("Dashboard built")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "FORCE_COLOR": "0"}
    llm = Service("llm", "llm    ", LAVENDER, [node, "server.js"], ROOT / "backend" / "llm_service", env, args.verbose)
    backend = Service("backend", "metis  ", MINT, [sys.executable, "-m", "backend"], ROOT, env, args.verbose)
    services: List[Service] = [llm, backend]
    dashboard: Optional[Service] = None
    if args.dev:
        dev_env = {**env, "VITE_API_URL": f"http://127.0.0.1:{backend_port}"}
        dashboard = Service("dashboard", "ui     ", YELLOW, [npm, "run", "dev", "--", "--port", "5173", "--strictPort"],
                            ROOT / "frontend", dev_env, args.verbose)
        services.append(dashboard)

    stopping = threading.Event()

    def shutdown(reason: str = "") -> None:
        if stopping.is_set():
            return
        stopping.set()
        say()
        step(reason or "Stopping Metis...")
        for s in services:
            s.show_output = False
        if dashboard:
            dashboard.stop(timeout=5)
        if backend.running:
            step("Disconnecting the bot and saving what Metis learned...")
            backend.stop(timeout=90)
        llm.stop(timeout=10)
        ok("Stopped. Bye!")

    if WINDOWS:
        # Closing the window: Windows gives us a few seconds - stop the children properly.
        import ctypes

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
        def on_console_event(event: int) -> bool:
            if event in (2, 5, 6):  # CTRL_CLOSE, CTRL_LOGOFF, CTRL_SHUTDOWN
                shutdown("Window closed - stopping Metis...")
                return True
            return False

        ctypes.windll.kernel32.SetConsoleCtrlHandler(on_console_event, True)
        main._handler = on_console_event  # keep a reference so it isn't garbage-collected

        def on_break(*_: object) -> None:
            raise KeyboardInterrupt  # Ctrl+Break stops Metis the same way Ctrl+C does

        signal.signal(signal.SIGBREAK, on_break)

    try:
        # 3. LLM service
        t0 = time.monotonic()
        step(f"Loading the language model ({model_name})...")
        llm.start()
        health = f"http://127.0.0.1:{llm_port}/health"
        if not wait_until(lambda: bool((http_json(health) or {}).get("model_loaded")), llm, 300, "The LLM service"):
            shutdown("Startup failed - cleaning up...")
            return 1
        ok(f"Language model ready {_c(DIM, f'({time.monotonic() - t0:.0f}s)')}")

        # 4. Backend + Discord bot
        t0 = time.monotonic()
        step("Starting Metis (memory, knowledge base, Discord bot)...")
        backend.start()
        if not wait_until(lambda: http_json(f"http://127.0.0.1:{backend_port}/health") is not None, backend, 300, "The backend"):
            shutdown("Startup failed - cleaning up...")
            return 1
        ok(f"Backend ready {_c(DIM, f'({time.monotonic() - t0:.0f}s)')}")

        if dashboard:
            dashboard.start()

        bot_line = _c(DIM, "disabled in config.yaml")
        if cfg.get("discord", {}).get("enabled", True):
            bot_line = _c(YELLOW, "not connected yet - check logs/backend.log")
            for _ in range(60):
                status = http_json(f"http://127.0.0.1:{backend_port}/status") or {}
                if status.get("discord_connected"):
                    bot_line = _c(MINT, f"connected as {status.get('bot_user')}")
                    break
                if not backend.running:
                    break
                time.sleep(0.5)

        url = "http://127.0.0.1:5173" if dashboard else f"http://127.0.0.1:{backend_port}"
        say()
        say(f"  {_c(MINT, 'Metis is running')}")
        say(f"    Dashboard   {_c(BOLD, url)}")
        say(f"    Discord     {bot_line}")
        say(f"    Model       {model_name}")
        say(f"    Logs        {_c(DIM, str(LOGS.relative_to(ROOT)) + os.sep)}")
        say()
        say(_c(DIM, "  Press Ctrl+C to stop. Activity:"))
        say()
        if args.open:
            webbrowser.open(url)
        for s in services:
            s.show_output = True

        # 5. Keep an eye on the services
        while not stopping.is_set():
            for s in services:
                if not s.running and not stopping.is_set():
                    say()
                    fail(f"{s.name} stopped unexpectedly (exit code {s.proc.returncode}).")
                    s.show_tail()
                    shutdown("Stopping the rest of Metis...")
                    return 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()
    return 0


if __name__ == "__main__":
    code = main()
    if code != 0 and WINDOWS and sys.stdin and sys.stdin.isatty():
        try:
            input("\n  Press Enter to close this window...")
        except (EOFError, KeyboardInterrupt):
            pass
    sys.exit(code)
