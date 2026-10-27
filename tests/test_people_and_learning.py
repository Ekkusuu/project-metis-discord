"""People registry, fact parsing and URL safety (offline)."""
from __future__ import annotations

import pytest

from backend.learning import (
    _clean_fact, _speakers, extract_facts, is_grounded, parse_legacy_transcript, person_view, statements,
)
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


def test_who_is_who_names_people_without_their_facts(store):
    store.seen(1, "Sava", ["sava_x"])
    store.add_facts("1", ["works as a nurse"])
    assert store.who_is_who(["1", "unknown"]) == "Sava (also known as sava_x)"


def test_notes_are_exported_for_the_knowledge_base_but_facts_are_not(store, tmp_path):
    store.seen(1, "Sava")
    store.add_facts("1", ["works as a nurse"])
    assert store.export_notes() == []
    store.add_notes(["movie night is on Fridays"])
    written = store.export_notes()
    assert "movie night is on Fridays" in written[0].read_text(encoding="utf-8")
    assert not (tmp_path / "people").exists()  # no per-person Markdown profile: facts live in the store only


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


def test_statements_leave_out_the_questions():
    """Only what someone states is evidence of a fact; asking about something isn't."""
    assert statements("Who's Mochi?") == ""
    assert statements("my cat is Pixel. who's Mochi? i'm moving to Berlin!") == \
        "my cat is Pixel. i'm moving to Berlin!"


def test_a_question_cannot_ground_a_fact_about_the_person_asking():
    """Asked "Who's Mochi?" about Metis's cat, "has a cat named Mochi" was filed under the person asking."""
    convo = [{"role": "user", "content": "[Sava]: who's Mochi? has she got a cat named Mochi?",
              "user_id": 1, "name": "Sava"}]
    view, own = person_view(convo, "Sava")
    assert "who's Mochi?" in view  # still shown: it is what they said, for the extractor to read
    assert "Mochi" not in own
    assert not is_grounded("has a cat named Mochi", own)


def test_statements_next_to_questions_still_ground_facts():
    convo = [{"role": "user", "content": "[Sava]: who said I work nights? i'm a nurse on night shifts",
              "user_id": 1, "name": "Sava"}]
    _, own = person_view(convo, "Sava")
    assert own == "i'm a nurse on night shifts"
    assert is_grounded("works night shifts as a nurse", own)


def test_is_grounded_rejects_invented_and_misattributed_facts():
    _, own = person_view(CONVO, "Sava")
    assert is_grounded("works as a nurse", own)
    assert is_grounded("is moving to Berlin next month", own)
    assert not is_grounded("works as a teacher", own)        # invented
    assert not is_grounded("dislikes horror games", own)     # someone else's fact


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


@pytest.fixture
def prompts_copy(tmp_path, monkeypatch):
    import shutil
    import backend.prompt_loader as loader
    from backend.config import PROJECT_ROOT

    path = tmp_path / "system_prompts.yaml"
    shutil.copy(PROJECT_ROOT / "system_prompts.yaml", path)
    monkeypatch.setattr(loader, "_prompts_file", path)
    monkeypatch.setattr(loader, "_prompts_cache", None)
    yield path
    loader._prompts_cache = None


def test_personalities_can_be_added_edited_and_deleted(prompts_copy):
    import yaml
    from backend.personality_store import delete, raw_prompt, save

    original = yaml.safe_load(prompts_copy.read_text(encoding="utf-8"))
    assert raw_prompt("casual", "voice_chat").startswith("You are Metis, a girl")

    save("pirate", {"text_chat": "You are Metis, a pirate.\nSay arr.", "voice_chat": "You are Metis, a pirate.\n\n  No emoji."})
    data = yaml.safe_load(prompts_copy.read_text(encoding="utf-8"))
    assert data["text_chat"]["pirate"].strip() == "You are Metis, a pirate.\nSay arr."
    assert data["voice_chat"]["pirate"].strip() == "You are Metis, a pirate.\n\n  No emoji."
    assert data["text_chat"]["flirty"] == original["text_chat"]["flirty"]
    assert raw_prompt("pirate", "voice_chat") == "You are Metis, a pirate.\n\n  No emoji."

    save("casual", {"text_chat": "edited text", "voice_chat": "edited voice"})
    data = yaml.safe_load(prompts_copy.read_text(encoding="utf-8"))
    assert data["voice_chat"]["casual"].strip() == "edited voice"
    assert data["voice_chat"]["mean"] == original["voice_chat"]["mean"]

    delete("pirate")
    data = yaml.safe_load(prompts_copy.read_text(encoding="utf-8"))
    assert "pirate" not in data["text_chat"] and "pirate" not in data["voice_chat"]
    assert len(data["text_chat"]) == len(original["text_chat"])
    assert "# Voice Chat Prompts" in prompts_copy.read_text(encoding="utf-8")
    assert prompts_copy.with_suffix(".yaml.bak").exists()


def test_bad_personality_names_and_empty_prompts_are_refused(prompts_copy):
    from backend.personality_store import PersonalityError, save

    with pytest.raises(PersonalityError):
        save("Bad Name!", {"text_chat": "x", "voice_chat": "y"})
    with pytest.raises(PersonalityError):
        save("ok", {"text_chat": "x", "voice_chat": "  "})


def test_extract_facts_keeps_only_grounded_ones(monkeypatch):
    import backend.learning as learning

    def c(quote, fact, about="themselves", kind=learning.WHO, importance=7):
        return {"quote": quote, "about": about, "kind": kind, "importance": importance, "fact": fact}

    monkeypatch.setattr(learning, "json_completion", lambda *a, **k: {"candidates": [
        c("night shifts as a nurse", "works as a nurse"),
        c("Pixel my cat was waiting", "- has a cat named Pixel"),
        c("i hate horror games", "hates horror games"),                         # someone else said it
        c("night shifts as a nurse", "works as a teacher"),                     # not what they said
        c("night shifts are brutal", "finds night shifts brutal", kind=learning.OPINION),
        c("i'm moving to Berlin next month", "is moving to Berlin", importance=3),  # judged unimportant
        c("i'm moving to Berlin next month", "Lena is moving to Berlin", about="someone else")]})
    assert extract_facts(CONVO, "Sava") == ["works as a nurse", "has a cat named Pixel"]


def test_extraction_is_taught_by_examples_including_empty_ones():
    """Like mem0: chatter, teasing Metis and reading out someone else's info are shown returning no facts."""
    import json
    from backend.learning import _FACT_EXAMPLES

    answers = [json.loads(m["content"])["candidates"] for m in _FACT_EXAMPLES if m["role"] == "assistant"]
    assert [] in answers and any(answers)
    labels = {(c["about"], c["kind"]) for answer in answers for c in answer}
    assert len(labels) >= 4  # kept and rejected kinds are both shown


def fact(text, importance=7):
    return {"fact": text, "quote": text, "importance": importance}


def test_reconcile_adds_updates_and_confirms(store, monkeypatch):
    """NONE now says which stored fact was repeated, so that fact is confirmed instead of the repeat being dropped."""
    import backend.learning as learning

    monkeypatch.setattr(learning, "get_people", lambda: store)
    store.add_facts("1", ["lives in Paris", "plays jungle"], name="Sava")
    monkeypatch.setattr(learning, "json_completion", lambda *a, **k: {"decisions": [
        {"event": "UPDATE", "id": "1"}, {"event": "NONE", "id": "2"}, {"event": "ADD", "id": ""},
        {"event": "UPDATE", "id": "9"}]})  # an id that doesn't exist is treated as new
    new = [fact("moved to Berlin"), fact("mains jungle"), fact("has a dog"), fact("likes tea")]
    to_add, updates, same = learning.reconcile("1", new)
    paris, jungle = (f["id"] for f in store.get("1")["facts"])
    assert updates == {paris: new[0]}
    assert same == {jungle: new[1]}
    assert to_add == new[2:]


def test_reconcile_compares_with_the_most_similar_facts_not_the_latest(store, monkeypatch):
    """With only the last 15 facts shown, a paraphrase of an older one was added a second time."""
    import backend.learning as learning

    monkeypatch.setattr(learning, "get_people", lambda: store)
    store.add_facts("1", ["has a dog named Biscuit"] + [f"hobby number {i} is {w}" for i, w in enumerate(
        "chess painting hiking baking singing sailing pottery skiing cycling surfing".split())], name="Sava")
    shown = []

    def fake(messages, *a, **k):
        shown.append(messages[-1]["content"])
        return {"decisions": [{"event": "NONE", "id": "1"}]}

    monkeypatch.setattr(learning, "json_completion", fake)
    to_add, updates, same = learning.reconcile("1", [fact("owns a dog called Biscuit")])
    assert len(shown[0].split("New facts")[0].splitlines()) <= 10  # "Stored facts:", a blank line and at most 8 facts
    assert "has a dog named Biscuit" in shown[0]  # the oldest fact, but the closest one
    assert to_add == [] and list(same) == [store.get("1")["facts"][0]["id"]]


def test_reconcile_skips_the_model_for_near_identical_wording(store, monkeypatch):
    import backend.learning as learning

    monkeypatch.setattr(learning, "get_people", lambda: store)
    store.add_facts("1", ["works as a nurse"], name="Sava")
    monkeypatch.setattr(learning, "json_completion", lambda *a, **k: pytest.fail("no model call needed"))
    to_add, updates, same = learning.reconcile("1", [fact("Works as a nurse.")])
    assert to_add == [] and updates == {} and len(same) == 1


def test_learn_supersedes_a_changed_fact_and_keeps_the_old_one(store, monkeypatch):
    import backend.learning as learning

    monkeypatch.setattr(learning, "get_people", lambda: store)
    store.add_facts("1", ["lives in Paris"], name="Sava")
    moved = {"quote": "i finally moved to berlin", "about": "themselves", "kind": learning.WHO, "importance": 8,
             "fact": "moved to Berlin"}
    replies = iter([{"candidates": [moved]}, {"decisions": [{"event": "UPDATE", "id": "1"}]}])
    monkeypatch.setattr(learning, "json_completion", lambda *a, **k: next(replies))
    convo = [{"role": "user", "content": "[Sava]: i finally moved to berlin last month, loving it", "user_id": 1,
              "name": "Sava"}]
    learning.learn(convo)
    assert [f["text"] for f in store.get("1")["facts"]] == ["moved to Berlin"]
    paris = next(m for m in store.memories._inactive() if m["text"] == "lives in Paris")
    berlin = store.get("1")["facts"][0]
    assert paris["status"] == "superseded" and paris["superseded_by"] == berlin["id"]
    assert berlin["importance"] == 8
    assert store.memories.get(berlin["id"])["quote"] == "i finally moved to berlin"


def test_grounding_compares_stems_and_allows_short_words():
    """5-character prefixes and 4+ letter words dropped "has a dog" and let "plays" match "playground"."""
    own = "my dog needs a walk. i've been playing a lot of guitar lately"
    assert is_grounded("has a dog", own)
    assert is_grounded("plays guitar", own)
    assert not is_grounded("has a cat", own)
    assert not is_grounded("works at a playground", own)  # "playg" is no longer a prefix of anything said
    assert not is_grounded("has a dog", "my dogma professor is up")  # whole stems, not substrings


def test_a_quote_must_be_one_of_the_speakers_own_sentences():
    from backend.learning import own_sentences, quote_supported

    convo = [{"role": "user", "content": "[Sava]: i work nights. who's Mochi? my cat is called Pixel", "user_id": 1,
              "name": "Sava"},
             {"role": "assistant", "content": "I have a dog named Rex, obviously"}]
    sentences = own_sentences(convo, "Sava")
    assert sentences == ["i work nights.", "my cat is called Pixel"]
    assert quote_supported("my cat is called Pixel", sentences)
    assert not quote_supported("i have a dog named Rex", sentences)       # Metis said it
    assert not quote_supported("who's Mochi", sentences)                   # a question
    assert not quote_supported("i work nights and my cat is Pixel", sentences)  # stitched from two sentences


def test_nothing_about_metis_or_that_only_metis_said_is_stored(monkeypatch):
    import backend.learning as learning

    convo = [{"role": "assistant", "content": "I have a cat named Mochi and I adore pineapple pizza"},
             {"role": "user", "content": "[Sava]: ha, sure. i adore pineapple pizza too lol",
              "user_id": 1, "name": "Sava"}]

    def c(quote, fact, about="themselves"):
        return {"quote": quote, "about": about, "kind": learning.LIKES, "importance": 7, "fact": fact}

    monkeypatch.setattr(learning, "json_completion", lambda *a, **k: {"candidates": [
        c("i adore pineapple pizza too", "loves pineapple pizza"),             # theirs
        c("a cat named Mochi", "has a cat named Mochi"),                       # a cat only Metis mentioned
        c("i adore pineapple pizza too", "Metis loves pineapple pizza"),       # about her
        c("I have a cat named Mochi", "has a cat named Mochi")]})              # her words as the quote
    assert [x["fact"] for x in learning.extract_candidates(convo, "Sava")] == ["loves pineapple pizza"]

