"""The memory store: migration, decay, reinforcement, consolidation into episodes, cued recall, the API (offline)."""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import backend.memory_store as memory_store
from backend.memory_store import retention, stability
from backend.people import PeopleStore
from backend.recall import recall, recall_block

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def iso(days: float) -> str:
    return (T0 + timedelta(days=days)).isoformat(timespec="seconds")


def ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")


@pytest.fixture
def store(tmp_path, monkeypatch):
    import backend.people as people_mod

    monkeypatch.setattr(people_mod, "PROFILE_DIR", tmp_path / "people")
    monkeypatch.setattr(people_mod, "NOTES_FILE", tmp_path / "notes.md")
    return PeopleStore(tmp_path / "people.json")


def said(*lines):
    return [{"role": "user", "content": f"[Ekkusu]: {line}"} for line in lines]


# --- migration ------------------------------------------------------------------------------------------------
def test_facts_in_people_json_move_into_the_store_once(tmp_path, monkeypatch):
    import backend.people as people_mod

    profiles = tmp_path / "people"
    profiles.mkdir()
    (profiles / "ekkusu-1.md").write_text("old profile", encoding="utf-8")
    monkeypatch.setattr(people_mod, "PROFILE_DIR", profiles)
    path = tmp_path / "people.json"
    path.write_text(json.dumps({"people": {"1": {
        "id": "1", "name": "ekkusu", "aliases": [], "first_seen": "x", "last_seen": "y", "message_count": 3,
        "facts": [{"id": "ea618e6448", "text": "plays League of Legends", "added": "2026-10-24T06:36:01+00:00"},
                  {"id": "4e1ec271b5", "text": "parents don't allow them to have a cat",
                   "added": "2026-10-24T07:54:29+00:00"}]}}, "notes": []}), encoding="utf-8")

    people = PeopleStore(path)
    facts = people.get("1")["facts"]
    assert [(f["id"], f["text"], f["added"], f["importance"]) for f in facts] == [
        ("ea618e6448", "plays League of Legends", "2026-10-24T06:36:01+00:00", 6),
        ("4e1ec271b5", "parents don't allow them to have a cat", "2026-10-24T07:54:29+00:00", 6)]
    assert "facts" not in json.loads(path.read_text(encoding="utf-8"))["people"]["1"]
    assert not list(profiles.glob("*.md"))  # facts were also indexed as Markdown profiles; not any more

    assert len(PeopleStore(path).get("1")["facts"]) == 2  # loading again does not duplicate them


# --- strength -------------------------------------------------------------------------------------------------
def test_stability_grows_with_importance_and_rehearsal_and_has_a_floor():
    assert stability("fact", 5, 0, 0) == pytest.approx(60)
    assert stability("episode", 5, 0, 0) == pytest.approx(14)
    assert stability("fact", 7, 0, 0) == pytest.approx(60 * 1.7)
    assert stability("fact", 5, 2, 1) == pytest.approx(60 * (1 + 0.5 * math.log(4)))
    assert stability("fact", 1, 0, 0) == 3 and stability("episode", 2, 0, 0) == 3  # importance can't take it below 3 days


def memory(kind="fact", importance=5, confirmations=0, recalls=0, confirmed=0.0, recalled=None):
    return {"kind": kind, "importance": importance, "confirmations": confirmations, "recalls": recalls,
            "last_confirmed": iso(confirmed), "last_recalled": iso(recalled) if recalled is not None else ""}


def test_retention_is_ebbinghaus_from_the_last_confirmation_or_recall():
    assert retention(memory(), T0) == pytest.approx(1.0)
    assert retention(memory(), T0 + timedelta(days=60)) == pytest.approx(math.exp(-1))
    # Recalled on day 50: the clock restarts from there, and the recall made it last longer.
    recalled = memory(recalls=1, recalled=50)
    assert retention(recalled, T0 + timedelta(days=60)) == pytest.approx(
        math.exp(-10 / stability("fact", 5, 0, 1)))
    assert retention(memory("episode"), T0 + timedelta(days=14)) < retention(memory("fact"), T0 + timedelta(days=14))


def test_decay_never_deletes_anything(store):
    fact = store.add_fact("1", "plays League of Legends", name="Ekkusu")
    store.memories._update(fact["id"], last_confirmed=ago(200), created=ago(200))
    assert retention(store.memories.get(fact["id"])) < 0.25
    assert [f["text"] for f in store.get("1")["facts"]] == ["plays League of Legends"]
    assert recall(store.memories, ["play"], ["1"]) == []  # too faint to come to mind on a passing cue...
    assert [m["text"] for m in recall(store.memories, ["play"], ["1"], explicit=True)] == ["plays League of Legends"]


# --- reinforcement vs supersede ---------------------------------------------------------------------------------
def test_hearing_it_again_confirms_a_fact_instead_of_adding_it(store):
    fact = store.add_fact("1", "has a dog named Biscuit", importance=6, name="Sava")
    store.memories._update(fact["id"], last_confirmed=ago(30))
    assert store.add_fact("1", "has a dog named Biscuit.", importance=8) is None
    again = store.memories.get(fact["id"])
    assert again["confirmations"] == 1 and again["importance"] == 8
    assert parse(again["last_confirmed"]) > datetime.now(timezone.utc) - timedelta(minutes=1) and len(store.get("1")["facts"]) == 1


def parse(iso_time):
    return memory_store.parse_time(iso_time)


def test_a_changed_fact_is_superseded_not_overwritten(store):
    paris = store.add_fact("1", "lives in Paris", importance=6, name="Sava")
    assert store.supersede_fact("1", paris["id"], "moved to Berlin", importance=7, quote="i moved to berlin")
    assert [f["text"] for f in store.get("1")["facts"]] == ["moved to Berlin"]
    old = store.memories.get(paris["id"])
    new = store.get("1")["facts"][0]
    assert old["status"] == "superseded" and old["superseded_by"] == new["id"]
    assert recall(store.memories, ["where do you live, paris?"], ["1"], explicit=True) == []  # no longer true


def test_over_the_cap_the_faintest_facts_are_let_go(store, monkeypatch):
    monkeypatch.setattr(memory_store, "get_config", lambda: {"memory": {"max_facts_per_person": 2}})
    import backend.people as people_mod

    monkeypatch.setattr(people_mod, "get_config", memory_store.get_config)
    old = store.add_fact("1", "collects stamps", name="Sava")
    store.memories._update(old["id"], last_confirmed=ago(400))
    store.add_fact("1", "works as a nurse")
    store.add_fact("1", "has a cat named Pixel")
    assert sorted(f["text"] for f in store.get("1")["facts"]) == ["has a cat named Pixel", "works as a nurse"]
    assert store.memories.get(old["id"])["status"] == "forgotten"


# --- consolidation into episodes ----------------------------------------------------------------------------------
CALL = [
    {"role": "user", "content": "[Robin]: my chemistry exam is tomorrow, i haven't slept at all", "user_id": 1, "name": "Robin"},
    {"role": "user", "content": "[Lena]: you always do well, you've got this", "user_id": 2, "name": "Lena"},
    {"role": "assistant", "content": "Good luck! I once failed a surfing exam on Mars."},
    {"role": "user", "content": "[Robin]: thanks. after it i'm adopting a puppy named Biscuit", "user_id": 1, "name": "Robin"},
]


def test_a_conversation_is_consolidated_into_episodes_for_everyone_in_it(store, monkeypatch):
    import backend.learning as learning

    monkeypatch.setattr(learning, "get_people", lambda: store)
    asked = []

    def fake(messages, schema, *args, **kwargs):
        asked.append((messages, kwargs))
        return {"episodes": [
            {"summary": "Robin was nervous about a chemistry exam the next day and said they would adopt a puppy "
                        "named Biscuit afterwards. Lena encouraged them.", "importance": 7},
            {"summary": "Robin failed a surfing exam on Mars.", "importance": 9},               # Metis made that up
            {"summary": "Everyone talked about exams and puppies for a while.", "importance": 5},  # names nobody
            {"summary": "Robin was nervous about a chemistry exam.", "importance": 4}]}          # repeats the first

    monkeypatch.setattr(learning, "json_completion", fake)
    assert learning.consolidate(CALL, session="voice-1@x") == 1
    assert len(asked) == 1 and asked[0][1]["priority"] == "background"
    assert "Robin: my chemistry exam" in asked[0][0][-1]["content"] and "Metis: Good luck" in asked[0][0][-1]["content"]
    [episode] = store.memories.active("episode")
    assert episode["person_ids"] == ["1", "2"] and episode["importance"] == 7 and episode["session"] == "voice-1@x"
    assert store.get("2")["name"] == "Lena" and [e["id"] for e in store.get("2")["episodes"]] == [episode["id"]]
    assert store.get("1")["episodes"][0]["people"] == ["Robin", "Lena"]


def test_small_talk_yields_no_episode_and_tiny_conversations_cost_no_call(store, monkeypatch):
    import backend.learning as learning

    monkeypatch.setattr(learning, "get_people", lambda: store)
    monkeypatch.setattr(learning, "json_completion", lambda *a, **k: {"episodes": []})
    chat = [{"role": "user", "content": f"[Sam]: {t}", "user_id": 3, "name": "Sam"}
            for t in ("hi", "what's up", "lol ok, brb then, talk later")]
    assert learning.consolidate(chat) == 0 and store.memories.active("episode") == []

    monkeypatch.setattr(learning, "json_completion", lambda *a, **k: pytest.fail("not worth a model call"))
    assert learning.consolidate(CALL[:1]) == 0
    assert learning.consolidate([{"role": "user", "content": "[Sam]: hi", "user_id": 3, "name": "Sam"}] * 3) == 0


def test_episode_extraction_is_taught_by_examples_including_empty_ones():
    from backend.learning import _EPISODE_EXAMPLES

    answers = [json.loads(m["content"])["episodes"] for m in _EPISODE_EXAMPLES if m["role"] == "assistant"]
    assert [] in answers and any(answers)


def test_conversations_end_when_the_call_ends_on_reset_when_dropped_and_after_a_quiet_spell(monkeypatch):
    import backend.discord_bot as bot_mod
    import backend.memory_manager as mm

    queued = []
    monkeypatch.setattr(mm, "learn_from", lambda messages, session="": queued.append(("learn", len(messages))))
    monkeypatch.setattr(mm, "consolidate_from", lambda messages, session="": queued.append(("episode", len(messages))))
    user = SimpleNamespace(id=1, name="A", display_name="A", bot=False)

    conv = bot_mod.Conversation(mode="text", key="text-1")
    conv.add_user(user, "one")
    conv.add_user(user, "two")
    conv.messages[-1]["ts"] -= 11 * 60  # quiet for 11 minutes before the next message
    conv.add_user(user, "three")
    assert queued == [("learn", 2), ("episode", 2)]  # what came before the gap, not the new message
    conv.reset()
    assert queued[2:] == [("learn", 3), ("episode", 1)]  # (the two before it come along as context for learning)
    conv.reset()
    assert queued[4:] == []  # nothing left: not consolidated twice

    voice = bot_mod.Conversation(mode="voice", key="voice-1")
    voice.add_user(user, "one")
    voice.messages[-1]["ts"] -= 3600
    voice.add_user(user, "two")
    assert queued[4:] == []  # a voice call ends when it ends, not when it is quiet


async def test_leaving_the_call_consolidates_it(monkeypatch):
    import backend.discord_bot as bot_mod
    import backend.memory_manager as mm
    from tests.test_voice_etiquette import FakeVoiceClient

    queued = []
    monkeypatch.setattr(mm, "learn_from", lambda messages, session="": queued.append("learn"))
    monkeypatch.setattr(mm, "consolidate_from", lambda messages, session="": queued.append("episode"))
    receiver = bot_mod.VoiceReceiver(FakeVoiceClient(), text_channel=None)
    receiver.conversation.add_user(SimpleNamespace(id=1, name="A", display_name="A", bot=False), "see you")
    receiver.stop()
    assert queued == ["learn", "episode"]


# --- recall ------------------------------------------------------------------------------------------------------
@pytest.fixture
def ekkusu(store):
    store.seen(1, "Ekkusu", ["ekkusu_x"])
    store.add_facts("1", ["plays League of Legends", "parents don't allow them to have a cat"])
    return store


def block(store, *lines, ids=("1",), present=()):
    return recall_block(store.memories, store, said(*lines), list(ids), present)


def test_a_greeting_recalls_nothing(ekkusu):
    """"Hello Metis, how are you?" got "How's your game since the parents won't let you have a cat?"."""
    ekkusu.add_episode(["1", "2"], "Ekkusu and Okeida planned a ranked League of Legends night with Metis.", 7)
    for greeting in ("Hello Metis, how are you?", "hey Metis", "Hello Ekkusu, how are you today?", "lol", "what's up?"):
        assert block(ekkusu, greeting) == "", greeting
    assert all(m["recalls"] == 0 for m in ekkusu.memories.active())  # nothing was brought to mind


def test_a_topical_cue_recalls_the_right_fact_framed_as_background(ekkusu):
    text = block(ekkusu, "what should I play tonight")
    assert "plays League of Legends" in text and "cat" not in text
    assert "only if it fits" in text and "never bring it up on your own" in text
    assert "Ekkusu" in text  # whose it is

    assert "cat" in block(ekkusu, "my friend has a cat and i'm jealous") and "League" not in block(ekkusu, "I want a cat")


def test_the_lines_before_the_latest_are_part_of_the_cue(ekkusu):
    assert "plays League" in block(ekkusu, "been playing a lot lately", "how are you?")
    assert block(ekkusu, "been playing a lot lately", "ok", "fine", "how are you?") == ""  # ...but only the last three


def test_remembering_strengthens_the_memory(ekkusu):
    before = {m["text"]: m for m in ekkusu.memories.active()}
    block(ekkusu, "what should I play tonight")
    after = {m["text"]: m for m in ekkusu.memories.active()}
    league, cat = after["plays League of Legends"], after["parents don't allow them to have a cat"]
    assert league["recalls"] == 1 and league["last_recalled"]
    assert cat["recalls"] == 0 and not cat["last_recalled"]
    assert retention(league) >= retention(before["plays League of Legends"])
    assert stability("fact", 6, 0, 1) > stability("fact", 6, 0, 0)


def test_a_faded_memory_needs_a_stronger_cue(ekkusu):
    league = next(m for m in ekkusu.memories.active() if "League" in m["text"])
    ekkusu.memories._update(league["id"], last_confirmed=ago(150), created=ago(150))
    assert retention(ekkusu.memories.get(league["id"])) < 0.25
    assert block(ekkusu, "what should I play tonight") == ""
    ekkusu.memories.reinforce(league["id"])  # she heard it again
    assert "plays League" in block(ekkusu, "what should I play tonight")


def test_at_most_three_memories_and_one_episode(ekkusu):
    ekkusu.add_facts("1", ["plays on the weekend", "plays guitar", "plays chess online"])
    ekkusu.add_episode(["1"], "Ekkusu played ranked all night and climbed to gold.", 8)
    ekkusu.add_episode(["1"], "Ekkusu played a long chess match against Okeida.", 8)
    chosen = recall(ekkusu.memories, ["let's play"], ["1"])
    assert len(chosen) == 3 and sum(m["kind"] == "episode" for m in chosen) <= 1


def test_only_the_people_in_the_conversation_are_candidates(ekkusu):
    ekkusu.seen(2, "Okeida")
    ekkusu.add_facts("2", ["plays the violin"])
    assert "violin" not in block(ekkusu, "what should I play tonight")
    assert "violin" in block(ekkusu, "what should I play tonight", ids=("1", "2"))
    assert "violin" in block(ekkusu, "what should I play tonight", present=("Okeida",))  # in the voice channel, quiet


def test_asked_directly_she_digs_for_anyone_even_someone_absent(ekkusu):
    ekkusu.seen(2, "Okeida")
    ekkusu.add_facts("2", ["trains voice models", "lives in Bucharest"])
    ekkusu.add_episode(["2"], "Okeida showed Metis a voice model she trained.", 6)
    text = block(ekkusu, "what do you know about Okeida?")           # Okeida isn't in the conversation
    assert "trains voice models" in text and "Bucharest" in text and "voice model she trained" in text
    assert "League" not in text
    assert "League" in block(ekkusu, "what do you know about me?")
    assert "use only this" in text
    # Not asked about anyone: a plain greeting that names her still recalls nothing.
    assert block(ekkusu, "hello Okeida!", ids=("1",)) == ""


def test_the_remember_tool_searches_facts_and_episodes(ekkusu):
    from backend.recall import think_back

    ekkusu.add_episode(["1"], "Ekkusu asked about chemistry exam tips before a big test.", 6)
    found = think_back(ekkusu.memories, ekkusu, "what subject was the exam?")
    assert "chemistry exam" in found and "(earlier today)" in found
    assert think_back(ekkusu.memories, ekkusu, "what is the capital of France?") == ""


# --- embeddings ---------------------------------------------------------------------------------------------------
def fake_embedder(monkeypatch, up=True):
    """Topics as vectors: [games, pets, greeting]. A real model puts these meanings close the same way."""
    groups = (("play", "league", "game", "ranked"), ("cat", "dog", "pet"), ("hello", "hey", "how are you"))

    def embed(texts, query=False):
        if not up:
            return None
        return [[float(any(w in t.lower() for w in g)) + 0.01 for g in groups] for t in texts]

    monkeypatch.setattr(memory_store, "_embed", embed)


def test_embeddings_are_used_when_the_embedder_is_up_and_missing_ones_are_filled_in(ekkusu, monkeypatch):
    fake_embedder(monkeypatch)
    assert all(m["embedding"] is None for m in ekkusu.memories.active())  # as after the migration
    chosen = recall(ekkusu.memories, ["any good ranked games tonight?"], ["1"])
    assert [(m["text"], m["mode"]) for m in chosen] == [("plays League of Legends", "vec")]  # no shared word needed
    assert all(m["embedding"] for m in ekkusu.memories.active())  # embedded lazily
    assert recall(ekkusu.memories, ["Hello, how are you?"], ["1"]) == []


def test_memories_without_a_vector_fall_back_to_words_next_to_ones_with(ekkusu, monkeypatch):
    fake_embedder(monkeypatch, up=False)
    ekkusu.add_facts("1", ["owns a dog named Biscuit"])
    fake_embedder(monkeypatch)
    plain = ekkusu.memories.add("fact", ["1"], "keeps a pet turtle", embed=False)
    ekkusu.memories._update(plain["id"], embedding=None)
    modes = {m["text"]: m["mode"] for m in recall(ekkusu.memories, ["i want a dog"], ["1"], now=None)}
    assert modes.get("owns a dog named Biscuit") == "vec"


def test_the_stemmer_and_word_similarity():
    from backend.memory_store import content_stems, lexical_similarity

    assert content_stems("He plays, played and was playing; she nurses") >= {"play", "nurs"}
    assert content_stems("moving, moved") == {"mov"} and content_stems("running stopped") == {"run", "stop"}
    assert content_stems("Hello, how are you?") == set()
    assert content_stems("hello Ekkusu", ignore=["Ekkusu"]) == set()
    assert lexical_similarity(content_stems("play tonight"), content_stems("plays League of Legends")) == pytest.approx(0.5)
    assert lexical_similarity(set(), {"a"}) == 0.0


# --- the API ------------------------------------------------------------------------------------------------------
def test_the_api_lists_facts_and_episodes_and_deletes_them(ekkusu, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import backend.routers.memory as memory_router

    monkeypatch.setattr(memory_router, "get_people", lambda: ekkusu)
    episode = ekkusu.add_episode(["1"], "Ekkusu told Metis about a ranked climb.", 6)
    client = TestClient(FastAPI())
    client.app.include_router(memory_router.router)

    [person] = client.get("/memory/people").json()["people"]
    assert person["name"] == "Ekkusu" and person["aliases"] == ["ekkusu_x"]
    assert {f["text"] for f in person["facts"]} == {"plays League of Legends", "parents don't allow them to have a cat"}
    assert all({"id", "text", "added", "importance", "retention"} <= set(f) for f in person["facts"])
    assert person["episodes"][0]["id"] == episode["id"] and person["episodes"][0]["people"] == ["Ekkusu"]
    assert client.get("/memory/people/1").json()["facts"] == person["facts"]
    assert client.get("/memory/stats").json()["facts_total"] == 2

    fact_id = person["facts"][0]["id"]
    assert client.delete(f"/memory/people/1/facts/{fact_id}").status_code == 200
    assert client.delete(f"/memory/people/1/facts/{fact_id}").status_code == 404
    assert client.delete(f"/memory/people/1/episodes/{episode['id']}").status_code == 200
    assert len(client.get("/memory/people/1").json()["facts"]) == 1
    assert client.get("/memory/people/1").json()["episodes"] == []
    assert client.delete("/memory/people/1").status_code == 200
    assert ekkusu.memories.active() == []  # forgetting a person forgets what was remembered of them
