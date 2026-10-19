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
