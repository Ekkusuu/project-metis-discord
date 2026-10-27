"""Phonetic correction of misheard vocabulary names (backend/stt_vocabulary.py), preferred sites, empty answers."""
import pytest

from backend.stt_vocabulary import PhoneticCorrector, Vocabulary, _extract

NAMES = ["Kha'Zix", "Kog'Maw", "Bel'Veth", "Kai'Sa", "Naafiri", "Tryndamere", "Yunara", "Aurelion Sol", "K'Sante",
         "Kindred", "Ezreal", "Sett", "Garen", "Illaoi", "Nunu & Willump"]


@pytest.fixture(scope="module")
def corrector():
    return PhoneticCorrector([Vocabulary("League", NAMES, ["league", "counter", "counters", "patch", "build"])],
                             protected=["Metis", "Sava"])


@pytest.mark.parametrize("heard, fixed", [
    ("What is the best counter for Kaziks?", "What is the best counter for Kha'Zix?"),
    ("What's Kindred's win rate against Kazakhs?", "What's Kindred's win rate against Kha'Zix?"),
    ("I keep losing to Bel Veth in the jungle.", "I keep losing to Bel'Veth in the jungle."),
    ("Is Cogma any good right now?", "Is Kog'Maw any good right now?"),
    ("What runes for Kai Sa?", "What runes for Kai'Sa?"),
    ("Nafiri is broken in mid.", "Naafiri is broken in mid."),
    ("Is Trendomere good this patch?", "Is Tryndamere good this patch?"),
    ("Best Unara build", "Best Yunara build"),
    ("Aurelian Sol is my main.", "Aurelion Sol is my main."),
])
def test_misheard_names_are_fixed(corrector, heard, fixed):
    corrector._last_on_topic.clear()
    assert corrector.correct(heard) == fixed


@pytest.mark.parametrize("text", [
    "I went to Kazakhstan last summer with my family.",
    "my brother Kazakhs live in Almaty",  # a real word, and nobody is talking about the game
    "Is this real or not?",
    "Garden party tomorrow",
    "Hello, Metis.",
    "Sava, are you there?",
])
def test_ordinary_speech_is_left_alone(corrector, text):
    corrector._last_on_topic.clear()
    assert corrector.correct(text) == text


def test_on_topic_ordinary_words_are_left_alone(corrector):
    corrector._last_on_topic.clear()
    corrector.correct("best counter for Garen")  # the conversation is about the game now
    for text in ["let me set it up", "this is real", "I am going to the garden later",
                 "the counter strike tournament was sick", "honestly I'd rather play something else"]:
        assert corrector.correct(text) == text


def test_real_word_changes_only_on_topic(corrector):
    corrector._last_on_topic.clear()
    assert corrector.correct("they hate Kazakhs") == "they hate Kazakhs"
    corrector.correct("what's the best build for Kindred")
    assert corrector.correct("they hate Kazakhs") == "they hate Kha'Zix"


def test_names_from_json():
    data = [{"id": -1, "name": "None"}, {"id": 1, "name": "Annie"}, {"id": 2, "name": "Olaf"}]
    assert _extract(data, "name") == ["None", "Annie", "Olaf"]
    assert _extract({"a": {"name": "Ahri"}}, "name") == ["Ahri"]
    assert _extract(["Zed", ""], "name") == ["Zed"]


def test_preferred_site_kept_only_on_topic(monkeypatch):
    from backend import tool_manager

    monkeypatch.setattr(tool_manager, "preferred_sites", lambda: [
        {"site": "u.gg", "for": "League of Legends statistics: win rates, tier lists, builds, counters"}])
    assert tool_manager.keep_named_site("Kha'Zix counters site:u.gg", "top five counters for Kha'Zix") \
        == "Kha'Zix counters site:u.gg"
    assert tool_manager.keep_named_site("weather Chisinau site:u.gg", "what's the weather in Chisinau") \
        == "weather Chisinau"


def test_answer_that_repeats_the_question_is_not_found():
    from backend.research import adds_information

    assert not adds_information("Kazik wound rate vs Kindred on Yudor.gg",
                                "What is the Kindred wound rate against Kazik on Yudor.gg?")
    assert adds_information("u.gg: Thresh", "Who is the best support?")
    assert adds_information("u.gg, patch 26.21: Kindred 49.2% vs Kha'Zix", "Kindred win rate vs Kha'Zix")


def test_correct_name_is_not_swallowed_by_a_longer_span(corrector):
    corrector._last_on_topic.clear()
    assert corrector.correct("Aurelion Sol is my main.") == "Aurelion Sol is my main."
    assert corrector.correct("Kha'Zix is so annoying") == "Kha'Zix is so annoying"


@pytest.mark.parametrize("query, said, expected", [
    ("Yone best build site:u.gg", "What's the best build for Yunara?", "Yunara best build site:u.gg"),
    ("best League of Legends champion 2024 site:u.gg", "what is the best League of Legends champion right now?",
     "best League of Legends champion site:u.gg"),
    ("World Cup 2026 winner", "who won the World Cup in 2026", "World Cup 2026 winner"),
    ("Tesla Roadster release date", "when was the tesla roadster released?", "Tesla Roadster release date"),
    ("best Ahri runes site:mobalytics.com", "Check mobalytics for the best Ahri runes",
     "best Ahri runes site:mobalytics.com"),
])
def test_query_keeps_the_names_people_said(query, said, expected):
    from backend.tool_manager import keep_said_names

    assert keep_said_names(query, said) == expected


@pytest.mark.parametrize("heard, fixed", [
    ("Matisse, what do you think?", "Metis, what do you think?"),  # heard in a real call
    ("What do you think, Matisse?", "What do you think, Metis?"),
    ("Hey Matthis, are you there?", "Hey Metis, are you there?"),
    ("Hello Madis.", "Hello Metis."),
    ("Medis is so slow today", "Metis is so slow today"),  # not a word: sounds like her name anywhere
])
def test_her_own_name_is_fixed_by_sound(corrector, heard, fixed):
    assert corrector.correct_own_name(heard, ["Metis"]) == fixed


@pytest.mark.parametrize("text", [
    "I love Matisse paintings.",  # a real name, nobody is talking to her
    "Matthews scored twice last night.",
    "Hey mates, what's up?",
    "Sava, are you there?",  # someone in the call
    "Metis, you there?",
])
def test_other_words_are_not_turned_into_her_name(corrector, text):
    assert corrector.correct_own_name(text, ["Metis"]) == text
