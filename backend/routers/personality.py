"""
Personality endpoints for the dashboard: list the personalities in system_prompts.yaml and switch.

A switch applies to the next reply (prompts are resolved per request) and is written to
config.yaml, so it survives a restart.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import CONFIG_PATH, get_config
from backend.prompt_loader import load_prompts, reload_prompts

router = APIRouter(prefix="/personality", tags=["personality"])


def available() -> List[Dict[str, Any]]:
    """Personalities that have a text prompt, with a one-line description taken from it."""
    prompts = load_prompts()
    voice = prompts.get("voice_chat") or {}
    result = []
    for name, prompt in (prompts.get("text_chat") or {}).items():
        first = str(prompt).strip().split(".")[0]
        description = re.sub(r"^You are Metis,?\s*", "", first).strip()
        result.append({
            "name": name,
            "description": description[:1].upper() + description[1:] if description else "",
            "has_voice": name in voice,
        })
    return result


def save_to_config(name: str) -> bool:
    """Set `personality:` in config.yaml, keeping its comments and layout. False if it couldn't be written."""
    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
        line = f'personality: "{name}"'
        if re.search(r"(?m)^personality:.*$", text):
            text = re.sub(r"(?m)^personality:.*$", line, text, count=1)
        else:
            text = line + "\n" + text
        CONFIG_PATH.write_text(text, encoding="utf-8")
        return True
    except OSError as e:
        print(f"Could not save the personality to {CONFIG_PATH}: {e}")
        return False


@router.get("")
async def get_personalities():
    reload_prompts()  # pick up personalities added to system_prompts.yaml since startup
    return {"current": get_config().get("personality"), "available": available()}


class PersonalityChange(BaseModel):
    name: str


@router.get("/{name}")
async def get_personality(name: str):
    """Both prompts as written in system_prompts.yaml, for the editor."""
    from backend.personality_store import SECTIONS, raw_prompt

    if name not in [p["name"] for p in available()]:
        raise HTTPException(status_code=404, detail=f"No personality called {name}")
    return {"name": name, **{section: raw_prompt(name, section) for section in SECTIONS}}


class PersonalityPrompts(BaseModel):
    text_chat: str
    voice_chat: str


@router.put("/{name}")
async def save_personality(name: str, prompts: PersonalityPrompts):
    """Create or update a personality in system_prompts.yaml."""
    from backend.personality_store import PersonalityError, save

    existed = name in [p["name"] for p in available()]
    try:
        save(name, prompts.model_dump())
    except PersonalityError as e:
        raise HTTPException(status_code=400, detail=str(e))
    print(f"🎭 Personality {name} {'updated' if existed else 'added'} from the dashboard")
    return {"name": name, "created": not existed, "available": available()}


@router.delete("/{name}")
async def delete_personality(name: str):
    from backend.personality_store import PersonalityError, delete

    if name == get_config().get("personality"):
        raise HTTPException(status_code=400, detail="That's the personality in use; switch to another one first.")
    try:
        delete(name)
    except PersonalityError as e:
        raise HTTPException(status_code=400, detail=str(e))
    print(f"🎭 Personality {name} deleted from the dashboard")
    return {"available": available()}


@router.put("")
async def set_personality(change: PersonalityChange):
    names = [p["name"] for p in available()]
    if change.name not in names:
        raise HTTPException(status_code=404, detail=f"Unknown personality. Available: {', '.join(names)}")
    get_config()["personality"] = change.name
    saved = save_to_config(change.name)
    print(f"🎭 Personality switched to {change.name} from the dashboard")
    return {"current": change.name, "saved": saved}
