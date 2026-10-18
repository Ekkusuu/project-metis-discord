"""
Memory endpoints: who Metis knows and what she has learned.
"""
import asyncio

from fastapi import APIRouter, HTTPException

from backend.people import get_people

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/people")
async def list_people():
    return {"people": get_people().all()}


@router.get("/people/{person_id}")
async def get_person(person_id: str):
    person = get_people().get(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Unknown person")
    return person


@router.delete("/people/{person_id}/facts/{fact_id}")
async def delete_fact(person_id: str, fact_id: str):
    people = get_people()
    if not people.remove_fact(person_id, fact_id):
        raise HTTPException(status_code=404, detail="Fact not found")
    await asyncio.to_thread(people.export_markdown, [person_id])
    return {"status": "ok"}


@router.delete("/people/{person_id}")
async def forget_person(person_id: str):
    if not get_people().forget(person_id):
        raise HTTPException(status_code=404, detail="Unknown person")
    return {"status": "ok"}


@router.get("/notes")
async def list_notes():
    return {"notes": get_people().notes}


@router.delete("/notes/{note_id}")
async def delete_note(note_id: str):
    people = get_people()
    if not people.remove_note(note_id):
        raise HTTPException(status_code=404, detail="Note not found")
    await asyncio.to_thread(people.export_markdown, [])
    return {"status": "ok"}


@router.post("/learn")
async def learn_now():
    """Learn from everything said since the last learning pass, and wait for it to finish."""
    from backend import discord_bot

    futures = await asyncio.to_thread(discord_bot.flush_learning)
    learned = 0
    for future in futures:
        result = await asyncio.wrap_future(future)
        if result:
            learned += result["facts"] + result["notes"]
    return {"status": "ok", "learned": learned}


@router.get("/stats")
async def memory_stats():
    from backend.memory_manager import stats

    people = get_people()
    return {
        **stats,
        "people_known": len(people.people),
        "facts_total": sum(len(p["facts"]) for p in people.people.values()),
        "notes_total": len(people.notes),
    }
