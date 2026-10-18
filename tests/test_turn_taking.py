"""Scenario tests for conversational etiquette (backend/turn_taking.py)."""
from __future__ import annotations

import pytest

from backend.turn_taking import (
    ASK, RESPOND, SKIP, WAIT_OPEN, Settings, Turn, addressed_by_name, decide, is_acknowledgement,
    is_room_question, name_variants,
)

BOT = 999
SAVA, EKK, MIRA = 1, 2, 3
NAMES = {SAVA: "Sava", EKK: "Ekkusu", MIRA: "Mira | 21"}
NOW = 10_000.0


def t(speaker, text, ago=0.0, mentions=(), reply_to=None, to=None):
    name = "Metis" if speaker is None else NAMES[speaker]
    return Turn(speaker, name, text, NOW - ago, tuple(mentions), reply_to, to)


def verdict(turns, **kw):
    return decide(turns, BOT, ["Metis"], NAMES, now=NOW, **kw).verdict


# --- explicit signals ---------------------------------------------------------------------
def test_dm_always_answered():
    assert verdict([t(SAVA, "lol")], is_dm=True) == RESPOND


def test_mention_reply_and_name_address_metis():
    assert verdict([t(SAVA, "what do you think?", mentions=[BOT])]) == RESPOND
    assert verdict([t(SAVA, "haha exactly", reply_to=BOT)]) == RESPOND
    for text in ["metis what's up", "hey Metis!", "thanks metis", "right, metis?", "Metis, can you help", "@Metis look"]:
        assert verdict([t(EKK, "random stuff", 30), t(SAVA, text)]) == RESPOND, text


def test_talking_about_metis_is_not_addressing_her():
    assert not addressed_by_name("i told metis about it yesterday", ["metis"])
    assert not addressed_by_name("metis's voice is so good", ["metis"])


def test_messages_for_someone_else_are_left_alone():
    assert verdict([t(SAVA, "check this out", mentions=[EKK])]) == SKIP
    assert verdict([t(SAVA, "omg same", reply_to=EKK)]) == SKIP
    assert verdict([t(SAVA, "ekkusu did you finish the boss?")]) == SKIP
    assert verdict([t(SAVA, "gn mira")]) == SKIP                 # 'Mira | 21' answers to 'mira'
    assert verdict([t(SAVA, "what do you think, ekkusu?")]) == SKIP


def test_both_metis_and_someone_else_mentioned_counts_as_metis():
    assert verdict([t(SAVA, "metis tell ekkusu he's wrong", mentions=[EKK])]) == RESPOND


# --- conversation flow --------------------------------------------------------------------
def test_follow_up_in_an_exchange_with_metis():
    turns = [t(SAVA, "metis recommend a game", 40), t(None, "Try Hades!", 35, to=SAVA), t(SAVA, "is it hard though")]
    assert verdict(turns) == RESPOND


def test_short_acknowledgement_is_not_answered():
    turns = [t(SAVA, "metis recommend a game", 40), t(None, "Try Hades!", 35, to=SAVA), t(SAVA, "ok thanks")]
    assert verdict(turns) == SKIP


def test_someone_else_cutting_into_metis_exchange_is_not_a_follow_up():
    turns = [t(SAVA, "metis recommend a game", 40), t(None, "Try Hades!", 35, to=SAVA),
             t(EKK, "sava wanna play tonight?", 20), t(SAVA, "sure, at 9")]
    assert verdict(turns) == SKIP


def test_stale_exchange_is_not_continued():
    turns = [t(SAVA, "metis recommend a game", 900), t(None, "Try Hades!", 890, to=SAVA), t(SAVA, "ugh work again")]
    assert verdict(turns) == ASK


def test_two_people_chatting_are_left_alone():
    turns = [t(SAVA, "dinner?", 50), t(EKK, "pizza", 40), t(SAVA, "again lol", 30), t(EKK, "you love pizza", 20),
             t(SAVA, "fair point honestly")]
    assert verdict(turns) == SKIP


def test_people_chatting_and_mentioning_metis_is_ambiguous():
    turns = [t(SAVA, "did you see that", 30), t(EKK, "yeah", 20), t(SAVA, "metis would know the answer probably")]
    assert verdict(turns) == ASK


def test_question_to_the_room_waits_for_humans():
    turns = [t(SAVA, "anyone know a good pizza place?")]
    assert verdict(turns) == WAIT_OPEN
    assert verdict(turns, settings=Settings(engagement="reserved")) == SKIP


def test_lone_remark_goes_to_the_model_unless_chatty():
    turns = [t(SAVA, "ugh so tired after work today")]
    assert verdict(turns) == ASK
    assert verdict(turns, settings=Settings(engagement="chatty")) == RESPOND


def test_lone_acknowledgement_is_ignored():
    assert verdict([t(SAVA, "lol")]) == SKIP


# --- cues ---------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["ok", "lol", "LMAO", "thanks", "thank you", "nice!", "gg", "haha", "k", "yeah"])
def test_acknowledgements(text):
    assert is_acknowledgement(text)


@pytest.mark.parametrize("text", ["ok but why", "thanks, what about tomorrow?", "lol that's wild, tell me more"])
def test_not_acknowledgements(text):
    assert not is_acknowledgement(text)


def test_room_questions():
    assert is_room_question("does anyone know how to fix this?")
    assert is_room_question("what do you guys think?")
    assert not is_room_question("anyone home")            # not a question
    assert not is_room_question("what time is it?")       # no room cue


def test_name_variants():
    assert set(name_variants("Mira | 21")) >= {"mira | 21", "mira"}
    assert set(name_variants("Ekkusu (ekk)")) >= {"ekkusu", "ekk"}


def test_name_closing_a_question_addresses_that_person():
    turns = [t(SAVA, "this movie was weird", 60), t(EKK, "agreed", 50), t(SAVA, "what did you think metis?")]
    assert verdict(turns) == RESPOND
    assert verdict([t(SAVA, "you there ekkusu?")]) == SKIP
    assert verdict([t(SAVA, "you're the best metis")]) == RESPOND
    assert not addressed_by_name("i was talking about metis?", ["metis"])


def test_lone_question_waits_for_the_room():
    assert verdict([t(SAVA, "what's the capital of australia?")]) == WAIT_OPEN
    assert verdict([t(SAVA, "how do magnets work")]) == WAIT_OPEN
    assert verdict([t(SAVA, "what's the capital of australia?")], settings=Settings(engagement="reserved")) == ASK


@pytest.mark.parametrize("text", ["brb dinner", "going to sleep, night all", "gtg", "back", "omw", "gn everyone"])
def test_status_updates_are_not_answered(text):
    assert verdict([t(SAVA, text)]) == SKIP


def test_status_update_to_metis_by_name_is_answered():
    assert verdict([t(SAVA, "gn metis")]) == RESPOND


def test_alone_with_metis_everything_is_for_her():
    turns = [t(SAVA, "hey, what's up?")]
    assert decide(turns, BOT, ["Metis"], NAMES, now=NOW, alone_with_metis=True).verdict == RESPOND
    assert decide([t(SAVA, "okay")], BOT, ["Metis"], NAMES, now=NOW, alone_with_metis=True).verdict == SKIP


def test_wait_kind_is_reported():
    assert decide([t(SAVA, "anyone know a pizza place?")], BOT, ["Metis"], NAMES, now=NOW).signals["wait"] == "room"
    assert decide([t(SAVA, "how do magnets work")], BOT, ["Metis"], NAMES, now=NOW).signals["wait"] == "lone"
