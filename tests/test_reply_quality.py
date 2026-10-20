"""Reply guidance, web-search triggers and speech-recognition name aliases."""
import pytest

from backend.routers.chat import build_prompt, reply_guidance
from backend.tool_manager import _FRESH_INFO, _SEARCH_REQUEST, is_question_like


def wants_web(text):
    return bool(_SEARCH_REQUEST.search(text) or (is_question_like(text) and _FRESH_INFO.search(text)))


def test_guidance_names_the_speaker_and_sits_right_before_their_message():
    messages = [{"role": "system", "content": "persona"}, {"role": "user", "content": "[ekkusu]: when was it released?"}]
    guidance = reply_guidance(messages, voice=True, has_context=True)
    assert "ekkusu" in guidance and "first sentence" in guidance and "spoken aloud" in guidance
    prompt = build_prompt(messages, "", "", guidance)
    assert prompt[-1] == messages[-1] and "How to reply" in prompt[-2]["content"]
    assert prompt[0] == messages[0]  # persona untouched


def test_text_guidance_has_no_voice_rule():
    assert "spoken aloud" not in reply_guidance([{"role": "user", "content": "[a]: hi"}], voice=False, has_context=False)


@pytest.mark.parametrize("text", [
    "Could you please tell me um when was the Tesla Roadster released?",
    "No, just tell me the date when it was released, please fi search for it.",
    "Please give me the date and time when the Tesla Roadster was released.",
    "can you look it up",
    "how much does a ps5 cost",
])
def test_fact_questions_and_search_requests_use_the_web(text):
    assert wants_web(text)


@pytest.mark.parametrize("text", [
    "Not much really, I was just sitting in this waste shot.",
    "You know, I'm actually looking for a girlfriend.",
    "Okay, then you can leave.",
    "i wonder what she meant",
])
def test_small_talk_does_not_trigger_search(text):
    assert not wants_web(text)


def test_voice_name_aliases():
    import backend.voice_service as vs

    svc = vs.VoiceService.__new__(vs.VoiceService)
    svc.config, svc._names, svc._aliases = {}, [], {"matthis": "Metis", "mathis": "Metis"}
    assert svc._correct_names("Hello Matthis, how are you") == "Hello Metis, how are you"
    assert svc._correct_names("I bought a new mattress") == "I bought a new mattress"


def test_search_query_ignores_metis_replies_and_made_up_words(monkeypatch):
    """A query built from Metis's own made-up reply searched for 'roadster' when asked about France."""
    import asyncio
    import backend.tool_manager as tm

    seen = {}

    def fake_query(messages, *args):
        seen["prompt"] = messages[-1]["content"]
        return "roadster date France news"

    monkeypatch.setattr(tm, "json_completion", lambda *a, **k: "web_search")
    monkeypatch.setattr(tm, "chat_completion", fake_query)
    messages = [
        {"role": "user", "content": "[ekkusu]: hello"},
        {"role": "assistant", "content": "Are you here for the roadster date today?"},
        {"role": "user", "content": "[ekkusu]: What happened in France recently that everyone is talking about?"},
    ]
    calls = asyncio.run(tm.decide_tools(messages, messages[-1]["content"]))
    assert "roadster" not in seen["prompt"]
    assert calls[0].arguments["query"] == "What happened in France recently that everyone is talking about?"
    assert calls[0].arguments["search_type"] == "news"


def test_grounded_query_keeps_good_queries():
    from backend.tool_manager import grounded_query

    metis = "Are you here for the roadster date today?"
    assert grounded_query("France news", "what happened in France? it is on the news", metis, "x") == "France news"
    assert grounded_query("roadster France news", "what happened in France?", metis, "x") == "x"
    assert grounded_query("Tesla Roadster release date", "when was the tesla roadster released", metis, "x") == \
        "Tesla Roadster release date"
    assert grounded_query("f1 race winner", "who won the f1 race yesterday?", "", "x") == "f1 race winner"
