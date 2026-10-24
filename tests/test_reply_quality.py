"""Reply guidance, web-search triggers and speech-recognition name aliases."""
import pytest

from backend.routers.chat import build_prompt, reply_guidance


def test_guidance_names_the_speaker_and_sits_right_before_their_message():
    messages = [{"role": "system", "content": "persona"}, {"role": "user", "content": "[ekkusu]: when was it released?"}]
    guidance = reply_guidance(messages, voice=True, has_context=True)
    assert "ekkusu" in guidance and "first sentence" in guidance and "spoken aloud" in guidance
    prompt = build_prompt(messages, "", "", guidance)
    assert prompt[-1] == messages[-1] and "How to reply" in prompt[-2]["content"]
    assert prompt[0] == messages[0]  # persona untouched


def test_text_guidance_has_no_voice_rule():
    assert "spoken aloud" not in reply_guidance([{"role": "user", "content": "[a]: hi"}], voice=False, has_context=False)


def test_voice_name_aliases():
    import backend.voice_service as vs

    svc = vs.VoiceService.__new__(vs.VoiceService)
    svc.config, svc._names, svc._aliases = {}, [], {"matthis": "Metis", "mathis": "Metis"}
    assert svc._correct_names("Hello Matthis, how are you") == "Hello Metis, how are you"
    assert svc._correct_names("I bought a new mattress") == "I bought a new mattress"


def test_grounded_query_keeps_good_queries():
    from backend.tool_manager import grounded_query

    metis = "Are you here for the roadster date today?"
    assert grounded_query("France news", "what happened in France? it is on the news", metis, "x") == "France news"
    assert grounded_query("roadster France news", "what happened in France?", metis, "x") == "x"
    assert grounded_query("Tesla Roadster release date", "when was the tesla roadster released", metis, "x") == \
        "Tesla Roadster release date"
    assert grounded_query("f1 race winner", "who won the f1 race yesterday?", "", "x") == "f1 race winner"


def test_weather_place_comes_from_the_question():
    from backend.tools.weather import WEATHER_QUESTION, place_words

    assert place_words("Uh could you please search for me what is the weather in Moldova Right Now?") == ["Moldova"]
    assert place_words("What is the weather in Chisinau right now") == ["Chisinau"]
    assert WEATHER_QUESTION.search("what's the temperature in London") and not WEATHER_QUESTION.search("hi there")


def test_metis_does_not_ping_herself():
    from backend.text_utils import strip_name_prefix

    assert strip_name_prefix("@Metis it's 13 degrees") == "it's 13 degrees"
    assert strip_name_prefix("Metis: hi") == "hi"


def test_weather_place_ignores_ordinary_words():
    from backend.tools.weather import place_words

    # "give" fuzzy-matched Givet, France, and the trailing "." spoiled "Moldova."
    assert place_words("No, just give me the weather for Moldova.") == ["Moldova"]
    assert place_words("Could you please search for Moldova's weather?") == ["Moldova"]


def test_spoken_text_has_no_quotes_or_actions():
    from backend.text_utils import clean_for_speech

    assert clean_for_speech('"Yo. You want the weather? Fine."') == "Yo. You want the weather? Fine."
    assert clean_for_speech("*checks phone, rolls her eyes*") == ""


def test_backchannel_needs_no_answer():
    from backend.turn_taking import is_acknowledgement

    for text in ["Mm-hmm.", "Uh-huh.", "Mhm", "Hmm, I see."]:
        assert is_acknowledgement(text), text
    assert not is_acknowledgement("Um, what is the weather?")


def test_voice_guidance_lists_the_voice_emotions(monkeypatch):
    import backend.routers.chat as chat

    messages = [{"role": "user", "content": "[a]: hi"}]
    monkeypatch.setattr(chat, "get_config", lambda: {"voice": {
        "tts_engine": "qwen3", "qwen_tts": {"emotions": {"happy": {}, "teasing": {}}}}})
    guidance = reply_guidance(messages, voice=True, has_context=False)
    assert "[neutral] [happy] [teasing]" in guidance
    assert "[happy]" not in reply_guidance(messages, voice=False, has_context=False)
    monkeypatch.setattr(chat, "get_config", lambda: {"voice": {"tts_engine": "kokoro"}})
    assert "[happy]" not in reply_guidance(messages, voice=True, has_context=False)


async def test_stream_regenerates_a_reply_that_repeats_the_previous_one(monkeypatch):
    import json as _json

    import backend.routers.chat as chat

    previous = "You sound like a dog who just found out the Great Discovery exists, so what are you waiting for?"
    final = [{"role": "assistant", "content": previous}, {"role": "user", "content": "[a]: I'm not waiting"}]
    calls = []

    async def fake_prepare(req, notify=None):
        return final, {}

    async def fake_stream(messages, stats=None, temperature=0.7, **kw):
        calls.append(temperature)
        text = ("[annoyed] " + previous) if len(calls) == 1 else "[teasing] Fine, then sit there."
        for word in text.split(" "):
            yield word + " "

    monkeypatch.setattr(chat, "_prepare", fake_prepare)
    monkeypatch.setattr(chat, "astream_chat", fake_stream)
    req = chat.ChatRequest(messages=[{"role": "user", "content": "[a]: hi"}], voice_mode=True)
    response = await chat.chat_stream(req, None)
    lines = [_json.loads(c) async for c in response.body_iterator]
    said = "".join(line.get("delta", "") for line in lines)
    assert said.strip() == "[teasing] Fine, then sit there."  # nothing of the repeat reached the client
    assert len(calls) == 2 and calls[1] > calls[0]


def test_made_up_voice_tags_map_to_the_nearest_emotion():
    from backend.text_utils import take_emotion

    known = ["excited", "annoyed", "scared", "teasing"]
    assert take_emotion("[crazy] Yes!", known) == ("excited", "Yes!")
    assert take_emotion("[hmph] Whatever.", known) == ("annoyed", "Whatever.")
    assert take_emotion("[danger] Run!", known) == ("scared", "Run!")
    assert take_emotion("[lucky] Nice.", known) == (None, "Nice.")  # "happy" isn't one of this voice's moods


async def test_stream_catches_a_repeat_of_an_older_reply_and_says_what_to_avoid(monkeypatch):
    """"I'm the girl who just asked you about Kindred builds" came three turns in a row."""
    import json as _json

    import backend.routers.chat as chat

    old = "I'm the girl who just asked you about Kindred builds, so basically your new best friend."
    final = [{"role": "assistant", "content": old}, {"role": "user", "content": "[a]: who are you?"},
             {"role": "assistant", "content": "Just me, hanging out."},
             {"role": "system", "content": "How to reply:\n- Say something new."},
             {"role": "user", "content": "[a]: are you a girl?"}]
    prompts = []

    async def fake_prepare(req, notify=None):
        return final, {}

    async def fake_stream(messages, stats=None, temperature=0.7, **kw):
        prompts.append(messages)
        text = old if len(prompts) == 1 else "Yep, last time I checked."
        for word in text.split(" "):
            yield word + " "

    monkeypatch.setattr(chat, "_prepare", fake_prepare)
    monkeypatch.setattr(chat, "astream_chat", fake_stream)
    req = chat.ChatRequest(messages=[{"role": "user", "content": "[a]: hi"}], voice_mode=True)
    response = await chat.chat_stream(req, None)
    said = "".join([_json.loads(c).get("delta", "") async for c in response.body_iterator])
    assert said.strip() == "Yep, last time I checked."
    assert "Don't start like you did before (\"I'm the girl who just asked" in prompts[1][-2]["content"]
    assert "Don't start like" not in prompts[0][-2]["content"]  # only on the retry


def test_voice_tag_written_like_a_name_label_leaves_no_colon():
    from backend.text_utils import take_emotion

    assert take_emotion("[happy]: Hey you.", ["happy"]) == ("happy", "Hey you.")
    assert take_emotion("[neutral] : Morning.", ["happy"]) == ("neutral", "Morning.")


def test_long_bracketed_text_is_spoken_without_brackets():
    from backend.text_utils import clean_for_speech

    assert clean_for_speech("[op.gg, Patch 16.21: Yasuo counters include Yone] Zed is easiest.") == \
        "op.gg, Patch 16.21: Yasuo counters include Yone Zed is easiest."
