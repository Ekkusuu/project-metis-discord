"""
Live feed of Discord messages for the dashboard (Server-Sent Events).
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/discord", tags=["discord"])

MAX_MESSAGES = 200


class MessageFeed:
    """
    Ring buffer of recent messages with monotonically increasing sequence numbers.
    SSE clients track the last seq they saw, so trimming the buffer or clearing it
    never makes them miss (or re-send) messages.
    """

    def __init__(self, maxlen: int = MAX_MESSAGES) -> None:
        self.messages: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self.seq = 0
        self.clear_count = 0
        self.version = 0
        self._changed = asyncio.Condition()

    async def add(self, message: Dict[str, Any]) -> Dict[str, Any]:
        self.seq += 1
        message = {**message, "seq": self.seq}
        message.setdefault("id", f"msg-{self.seq}")
        self.messages.append(message)
        await self._notify()
        return message

    async def clear(self) -> None:
        self.messages.clear()
        self.clear_count += 1
        await self._notify()

    def since(self, seq: int) -> List[Dict[str, Any]]:
        return [m for m in self.messages if m["seq"] > seq]

    async def _notify(self) -> None:
        async with self._changed:
            self.version += 1
            self._changed.notify_all()

    async def wait_for_change(self, seen_version: int, timeout: float) -> None:
        """Wait until something changed after `seen_version` (returns immediately if it already has)."""
        async with self._changed:
            try:
                await asyncio.wait_for(self._changed.wait_for(lambda: self.version != seen_version), timeout)
            except asyncio.TimeoutError:
                pass


feed = MessageFeed()


@router.get("/messages")
async def get_messages():
    return {"messages": list(feed.messages)}


@router.post("/add-message")
async def add_message(message: Dict[str, Any]):
    """Called by the Discord bot for every user message and reply."""
    stored = await feed.add(message)
    return {"status": "success", "seq": stored["seq"]}


@router.post("/clear")
async def clear_messages():
    await feed.clear()
    return {"status": "success", "message": "Discord messages cleared"}


@router.get("/stream")
async def stream_messages(request: Request):
    """
    SSE stream. Each message is sent with `id: <seq>`, so EventSource's automatic
    reconnect (Last-Event-ID) resumes exactly where it left off. A `clear` event
    tells clients to empty their list.
    """
    last_id = request.headers.get("last-event-id")
    start_seq: Optional[int] = int(last_id) if last_id and last_id.isdigit() else None

    async def events():
        # New connections only get live messages; history comes from /discord/messages.
        last_seq = feed.seq if start_seq is None else start_seq
        clears = feed.clear_count
        yield "retry: 3000\n\n"
        while not await request.is_disconnected():
            seen_version = feed.version
            if feed.clear_count != clears:
                clears = feed.clear_count
                yield "event: clear\ndata: {}\n\n"
            for message in feed.since(last_seq):
                last_seq = message["seq"]
                yield f"id: {last_seq}\ndata: {json.dumps(message, ensure_ascii=False)}\n\n"
            await feed.wait_for_change(seen_version, timeout=15)
            yield ": heartbeat\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
