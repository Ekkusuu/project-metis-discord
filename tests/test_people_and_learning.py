"""People registry, fact parsing and URL safety (offline)."""
from __future__ import annotations

import pytest

from backend.learning import _clean_fact, _speakers, extract_facts, is_grounded, parse_legacy_transcript, person_view
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


CONVO = [
    {"role": "user", "content": "[Sava]: night shifts as a nurse are brutal, at least Pixel my cat was waiting", "user_id": 1, "name": "Sava"},
    {"role": "user", "content": "[Ekkusu]: pixel is the best cat lol. i hate horror games btw", "user_id": 2, "name": "Ekkusu"},
    {"role": "assistant", "content": "Haha noted!"},
    {"role": "user", "content": "[Sava]: also i'm moving to Berlin next month", "user_id": 1, "name": "Sava"},
]


def test_person_view_shows_only_their_messages_with_context():
    view, own = person_view(CONVO, "Sava")
    assert view.splitlines() == [
        "Sava: night shifts as a nurse are brutal, at least Pixel my cat was waiting",
        "(context) Metis: Haha noted!",
        "Sava: also i'm moving to Berlin next month",
    ]
    assert "horror" not in own


def test_is_grounded_rejects_invented_and_misattributed_facts():
    _, own = person_view(CONVO, "Sava")
    assert is_grounded("works as a nurse", own)
    assert is_grounded("is moving to Berlin next month", own)
    assert not is_grounded("works as a teacher", own)        # invented
    assert not is_grounded("dislikes horror games", own)     # someone else's fact


def test_extract_facts_filters_ungrounded_and_transient(monkeypatch):
    import backend.learning as learning

    monkeypatch.setattr(learning, "chat_completion", lambda *a, **k: (
        "works as a nurse\n- has a cat named Pixel\nhates horror games\nis tired\nworks as a teacher\nNONE"))
    assert extract_facts(CONVO, "Sava") == ["works as a nurse", "has a cat named Pixel"]


def test_requests_and_facts_about_metis_are_not_remembered():
    from backend.learning import _NOT_ABOUT_THEM

    for fact in ["is asking for the exact release date of a Tesla Roadster", "wants to know what happened in France",
                 "does not want Metis to leave", "asked the bot to search for news"]:
        assert _NOT_ABOUT_THEM.search(fact), fact
    for fact in ["is vegetarian", "wants to become a pilot", "is curiously good at chess", "metisse"]:
        assert not _NOT_ABOUT_THEM.search(fact), fact


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


def test_personality_switch_keeps_config_comments(tmp_path, monkeypatch):
    import backend.routers.personality as personality

    cfg = tmp_path / "config.yaml"
    cfg.write_text('# Global Personality Setting\npersonality: "flirty"\n\nchat:\n  temperature: 0.7\n', encoding="utf-8")
    monkeypatch.setattr(personality, "CONFIG_PATH", cfg)
    assert personality.save_to_config("casual")
    assert cfg.read_text(encoding="utf-8") == '# Global Personality Setting\npersonality: "casual"\n\nchat:\n  temperature: 0.7\n'


def test_personalities_are_listed_with_descriptions():
    from backend.routers.personality import available

    found = {p["name"]: p for p in available()}
    assert "casual" in found and found["casual"]["description"].startswith("A casual")
