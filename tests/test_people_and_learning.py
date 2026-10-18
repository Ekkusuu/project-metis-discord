"""People registry, fact parsing and URL safety (offline)."""
from __future__ import annotations

import pytest

from backend.learning import _clean_fact, _speakers, parse_facts, parse_legacy_transcript
from backend.people import PeopleStore, is_duplicate
from backend.tools.web_page_tool import UnsafeURLError, _check_public, find_urls


@pytest.fixture
def store(tmp_path, monkeypatch):
    import backend.people as people_mod

    monkeypatch.setattr(people_mod, "PROFILE_DIR", tmp_path / "people")
    monkeypatch.setattr(people_mod, "NOTES_FILE", tmp_path / "notes.md")
    return PeopleStore(tmp_path / "people.json")


def test_seen_tracks_names_and_aliases(store):
    store.seen(1, "Sava", ["sava_x"])
    store.seen(1, "Sava the Great", ["sava_x"])
    person = store.get("1")
    assert person["name"] == "Sava the Great"
    assert set(person["aliases"]) == {"sava_x", "Sava"}
    assert person["message_count"] == 2


def test_facts_are_deduplicated(store):
    store.seen(1, "Sava")
    assert store.add_facts("1", ["has a cat named Pixel", "works as a nurse"]) == 2
    assert store.add_facts("1", ["Has a cat named Pixel.", "has a cat named pixel"]) == 0
    assert store.add_facts("1", ["is moving to Berlin"]) == 1
    assert [f["text"] for f in store.get("1")["facts"]] == ["has a cat named Pixel", "works as a nurse", "is moving to Berlin"]


def test_name_only_profile_merges_into_real_user(store):
    store.add_facts("name:sava", ["likes Elden Ring"], name="Sava")
    store.seen(42, "Sava")
    assert store.get("name:sava") is None
    assert [f["text"] for f in store.get("42")["facts"]] == ["likes Elden Ring"]


def test_resolve_prefers_conversation_candidates(store):
    store.seen(1, "Sava", ["sava_x"])
    assert store.resolve("sava_x") == "1"
    assert store.resolve("@Sava") == "1"
    assert store.resolve("Sava", {"Sava": "99"}) == "99"
    assert store.resolve("nobody") is None


def test_remove_and_forget(store):
    store.seen(1, "Sava")
    store.add_facts("1", ["works as a nurse"])
    fact_id = store.get("1")["facts"][0]["id"]
    assert store.remove_fact("1", fact_id)
    assert store.get("1")["facts"] == []
    assert store.forget("1")
    assert store.get("1") is None


def test_persistence_roundtrip(store, tmp_path):
    store.seen(1, "Sava")
    store.add_facts("1", ["works as a nurse"])
    store.add_notes(["movie night is on Fridays"])
    again = PeopleStore(tmp_path / "people.json")
    assert again.get("1")["facts"][0]["text"] == "works as a nurse"
    assert again.notes[0]["text"] == "movie night is on Fridays"


def test_profile_block_and_export(store, tmp_path):
    store.seen(1, "Sava", ["sava_x"])
    store.add_facts("1", ["works as a nurse"])
    block = store.profile_block(["1", "unknown"])
    assert block == "- Sava (also known as sava_x): works as a nurse"
    written = store.export_markdown(["1"])
    assert "Sava works as a nurse" in written[0].read_text(encoding="utf-8")


def test_is_duplicate():
    assert is_duplicate("has a cat named Pixel", ["Has a cat named Pixel!"])
    assert not is_duplicate("has a dog", ["has a cat named Pixel"])


def test_parse_facts_assigns_lines_to_speakers():
    speakers = {"Sava": "1", "Ekkusu (ekk)": "2"}
    response = "\n".join([
        "Sava | works as a nurse",
        "- Ekkusu | is learning Japanese",  # prefix match for a disambiguated label
        "[Sava] | Sava has a cat named Pixel",
        "Stranger | should be dropped",
        "NOTE | movie night on Friday",
        "random line without a pipe",
        "NONE",
    ])
    facts, notes = parse_facts(response, speakers)
    assert facts == {"Sava": ["works as a nurse", "has a cat named Pixel"], "Ekkusu (ekk)": ["is learning Japanese"]}
    assert notes == ["movie night on Friday"]


def test_clean_fact_strips_name():
    assert _clean_fact("Sava's cat is called Pixel.", "Sava") == "cat is called Pixel"
    assert _clean_fact('"Sava likes tea"', "Sava") == "likes tea"


def test_speakers_from_ids_and_brackets():
    msgs = [
        {"role": "user", "content": "[Sava]: hi", "user_id": 1, "name": "Sava"},
        {"role": "assistant", "content": "hey"},
        {"role": "user", "content": "[Ghost]: boo"},
    ]
    speakers = _speakers(msgs)
    assert speakers["Sava"] == "1"
    assert speakers["Ghost"].startswith("name:") or speakers["Ghost"].isdigit()


def test_parse_legacy_transcript():
    msgs = parse_legacy_transcript("User: [sava]: hi\nAI: hello\n\nnoise")
    assert msgs == [{"role": "user", "content": "[sava]: hi"}, {"role": "assistant", "content": "hello"}]


def test_find_urls():
    assert find_urls("see https://a.com/x?y=1, and (http://b.org).") == ["https://a.com/x?y=1", "http://b.org"]


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/", "http://localhost/", "http://10.0.0.5/", "file:///etc/passwd"])
def test_private_urls_are_refused(url):
    with pytest.raises(UnsafeURLError):
        _check_public(url)


def test_selection_schema_only_allows_enabled_tools():
    from backend.tool_manager import ToolRegistry, _selection_schema
    from backend.tools.web_page_tool import WebPageTool
    from backend.tools.web_search_tool import WebSearchTool

    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(WebPageTool(), enabled=False)
    schema = _selection_schema(registry, 2)
    variants = schema["items"]["oneOf"]
    assert schema["maxItems"] == 2
    assert [v["properties"]["tool"]["const"] for v in variants] == ["web_search"]
    assert variants[0]["properties"]["arguments"]["properties"] == {"query": {"type": "string"}}
