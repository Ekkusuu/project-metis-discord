"""Async unit tests for MessageFeed in backend/routers/discord.py.

The feed is a pure in-memory ring buffer; no Discord connection is involved.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from backend.routers.discord import MessageFeed

pytestmark = pytest.mark.asyncio  # asyncio_mode = auto in pytest.ini as well


async def test_add_assigns_increasing_seq_and_default_id():
    feed = MessageFeed()
    first = await feed.add({"author": "user", "content": "hi"})
    second = await feed.add({"author": "metis", "content": "hello"})
    assert first["seq"] == 1
    assert second["seq"] == 2
    assert first["id"] == "msg-1"
    assert second["id"] == "msg-2"
    assert [m["content"] for m in feed.messages] == ["hi", "hello"]


async def test_add_keeps_a_provided_id():
    feed = MessageFeed()
    stored = await feed.add({"id": "custom-id", "content": "x"})
    assert stored["id"] == "custom-id"


async def test_add_increments_version():
    feed = MessageFeed()
    assert feed.version == 0
    await feed.add({"content": "x"})
    assert feed.version == 1


async def test_ring_buffer_keeps_only_the_newest_messages():
    feed = MessageFeed(maxlen=3)
    for i in range(5):
        await feed.add({"content": str(i)})
    assert [m["seq"] for m in feed.messages] == [3, 4, 5]
    assert len(feed.messages) == 3


async def test_since_returns_newer_messages_even_after_eviction():
    feed = MessageFeed(maxlen=3)
    for i in range(5):
        await feed.add({"content": str(i)})
    assert [m["seq"] for m in feed.since(0)] == [3, 4, 5]
    assert [m["seq"] for m in feed.since(2)] == [3, 4, 5]
    assert [m["seq"] for m in feed.since(4)] == [5]
    assert feed.since(5) == []


async def test_clear_empties_and_increments_clear_count():
    feed = MessageFeed()
    await feed.add({"content": "x"})
    version_before = feed.version

    await feed.clear()

    assert len(feed.messages) == 0
    assert feed.clear_count == 1
    assert feed.version == version_before + 1
    assert feed.seq == 1  # seq keeps counting, so clients do not re-see old ids


async def test_wait_for_change_returns_immediately_if_version_already_changed():
    feed = MessageFeed()
    await feed.add({"content": "x"})  # version 0 -> 1

    start = time.monotonic()
    await feed.wait_for_change(seen_version=0, timeout=5.0)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0  # not the 5s timeout


async def test_wait_for_change_times_out_silently():
    feed = MessageFeed()
    start = time.monotonic()
    await feed.wait_for_change(seen_version=feed.version, timeout=0.05)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.03


async def test_wait_for_change_wakes_up_on_add():
    feed = MessageFeed()

    async def add_later():
        await asyncio.sleep(0.02)
        await feed.add({"content": "wake up"})

    task = asyncio.create_task(add_later())
    await feed.wait_for_change(seen_version=feed.version, timeout=2.0)
    await task

    assert feed.seq == 1
