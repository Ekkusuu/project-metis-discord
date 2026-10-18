"""
Launcher: `python -m backend`

Checks that the configured port is free *before* importing the app (which loads torch,
transformers and the embedding model), so a port clash is reported in a second instead
of after the whole startup.
"""
import os
import sys

from backend.config import CONFIG_PATH, get_config, port_in_use


def main() -> None:
    cfg = get_config()["backend"]
    host = cfg.get("host", "127.0.0.1")
    port = int(os.getenv("BACKEND_PORT") or cfg.get("port", 8730))
    if port_in_use(host, port):
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        sys.exit(
            f"✗ Port {port} is already in use by another program (or another copy of Metis).\n"
            f"  Close it, or pick a free port under backend.port in {CONFIG_PATH}."
        )

    import uvicorn

    uvicorn.run("backend.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
