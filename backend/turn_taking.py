"""
Conversational etiquette: is a message meant for Metis, and is it her turn?

Deciding this with one "should I reply?" prompt is unreliable for a small local
model, and it ignores the strongest evidence Discord gives us for free. So the
decision is layered:

1. Structural signals (deterministic, cover most messages):
   - DMs, @mentions of Metis, replies to her messages, her name used to address
     her ("Metis, ...", "... right metis?")                      -> respond
   - @mentions of / replies to / name-address of someone else    -> stay quiet
   - follow-ups in an exchange Metis is already having with this person
                                                                 -> respond
     (short acknowledgements like "ok thanks" are let go, like a person would)
   - two or more people talking among themselves without her     -> stay quiet
   - questions to the room ("anyone know ...?")                  -> wait, and
     only answer if no human answers first
2. Only what's left (genuinely ambiguous) goes to the model, as one constrained
   choice: who is the last message meant for?

Everything here is pure (no Discord or network objects) so it's easy to test.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

RESPOND = "respond"   # it's Metis's turn
SKIP = "skip"         # not for Metis / not her turn
ASK = "ask"           # ambiguous: ask the model who it's for
WAIT_OPEN = "wait"    # a question to the room: give humans the first chance


@dataclass
class Turn:
    speaker_id: Optional[int]          # None = Metis
    name: str
    text: str
    ts: float
    mentions: Tuple[int, ...] = ()     # user ids @mentioned in the message
    reply_to: Optional[int] = None     # author id of the message this replies to
    to: Optional[int] = None           # for Metis's turns: who she was answering

    @property
    def is_metis(self) -> bool:
        return self.speaker_id is None


@dataclass
class Decision:
    verdict: str
    reason: str
    signals: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text cues
# ---------------------------------------------------------------------------
_GREETING = r"(?:hey|hi|hello|hiya|yo|oi|ok|okay|so|and|but|well|thanks|thank you|thx|ty|bye|cya|gn|gm|good (?:morning|night|evening)|welcome back)"
_ACK_WORD = (
    r"(?:ok(?:ay)?|k+|kk|alright|aight|sure|yeah|yea|yep|yup|ya|true|fair|right|nice|cool|great|perfect|"
    r"lol+|lmao+|lmfao|haha+|hehe+|xd+|ty|thx|thanks?(?: (?:you|u|a lot|so much))?|tysm|np|got it|gotcha|"
    r"bet|word|gg|ggs|same|mood|rip|oof|damn|wow|nah|no|yes|ye|hm+|ah+|oh+|bye|cya|gn|night|will do|sounds good|"
    # Spoken backchannel, as the transcriber writes it ("Mm-hmm.", "Uh-huh.", "Mhm.")
    r"m+-?hm+|mhm+|uh-?huh|uh+|um+|er+m*|i see)"
)
_ACK = re.compile(rf"^\W*{_ACK_WORD}(?:[\W_]+{_ACK_WORD})*\W*$", re.IGNORECASE)
# After a name at the start of a message, these mean the person is being spoken TO
# ("Metis, ...", "metis tell him", "metis can you ..."), not ABOUT ("metis would know").
_AFTER_VOCATIVE = (
    r"(?:\s*[,:!?.]|\s*$|\s+(?:hi\b|hello\b|hey\b|(?:can|could|would|will|do|did|are|have|should)\s+(?:you|u)\b|you\b|u\b|ur\b|"
    r"pls\b|please\b|what\b|how\b|why\b|when\b|where\b|who\b|which\b|tell\b|say\b|play\b|sing\b|give\b|show\b|"
    r"help\b|look\b|check\b|remind\b|search\b|find\b|explain\b|stop\b|wait\b|listen\b|guess\b|think\b|"
    r"come\b|join\b|leave\b|answer\b|pick\b|choose\b|rate\b|describe\b|write\b|read\b|recommend\b))"
)
_ROOM = re.compile(
    r"\b(any(one|body| of you)|every(one|body)|you (guys|all|two)|y'?all|ya'?ll|chat|guys|people|folks|someone|somebody)\b",
    re.IGNORECASE,
)


def is_acknowledgement(text: str) -> bool:
    """'ok', 'lol', 'thanks', 'nice' ... -- things that don't need an answer."""
    text = re.sub(r"<a?:\w+:\d+>|[\U0001F000-\U0001FAFF☀-➿]", "", text).strip()
    return bool(_ACK.match(text)) or len(text) == 0


_STATUS = re.compile(
    r"^\W*(brb|afk|gtg|g2g|bbl|omw|back|im back|i'?m back|be right back|be back|"
    r"(going|gonna|off|heading|went) (to )?(sleep|bed|eat|shower|work|school|out|afk|offline)|"
    r"(dinner|lunch|food|shower|bathroom|bio) ?(time|break)?|night( all| everyone| guys)?|good ?night( all| everyone)?|"
    r"gn( all| everyone| guys)?|cya( all| later| guys)?|later( all| guys)?|see (ya|you)( later| all)?)\b",
    re.IGNORECASE,
)


def is_status_update(text: str) -> bool:
    """'brb dinner', 'going to sleep, night all', 'back' -- announcements that don't ask for a reply."""
    return bool(_STATUS.match(text)) and "?" not in text


def is_room_question(text: str) -> bool:
    """A question thrown at the whole room ('anyone know ...?', 'what do you guys think?')."""
    return "?" in text and bool(_ROOM.search(text))


def name_variants(name: str) -> List[str]:
    """'Sava | 21' -> ['sava | 21', 'sava'] ; 'Ekkusu (ekk)' -> ['ekkusu (ekk)', 'ekkusu', 'ekk']."""
    name = name.strip().lower()
    out = {name}
    paren = re.match(r"^(.*?)\s*\((.+)\)$", name)
    if paren:
        out.update({paren.group(1).strip(), paren.group(2).strip()})
    first = re.split(r"[\s|_.\-]+", name)[0]
    if len(first) >= 3:
        out.add(first)
    return [v for v in out if len(v) >= 2]


def _sound(word: str) -> str:
    """A rough sound key: consonants only, d/t, z/s, c/k merged, repeats collapsed ("Medy's" ~ "Metis")."""
    word = re.sub(r"[^a-z]", "", word.lower())
    word = word.replace("ph", "f").translate(str.maketrans("dzcq", "tskk"))
    key = word[:1] + re.sub(r"[aeiouyhw]", "", word[1:])
    return re.sub(r"(.)\1+", r"\1", key)


# Real words that sound like "Metis" (m-t-s) and are said in a call.
_SOUND_ALIKE_WORDS = {"mats", "meets", "mates", "meats", "mitts", "motes", "mutts", "matts", "moots", "mists"}


def spoken_names(text: str, names: Iterable[str]) -> str:
    """
    `text` with words that sound like one of `names` replaced by the name. Speech-to-text writes "Metis" as
    "Medy's", "Matisse" or "Mattis"; "Medy's Hello" went unanswered.
    """
    keys = {_sound(n): n for n in names if len(n) >= 4 and " " not in n}

    def swap(m: re.Match) -> str:
        word = m.group(0)
        if word.lower() in _SOUND_ALIKE_WORDS or not 4 <= len(word) <= 9:
            return word
        return keys.get(_sound(word), word)

    return re.sub(r"[A-Za-z][A-Za-z']*", swap, text) if keys else text


def addressed_by_name(text: str, names: Iterable[str]) -> bool:
    """
    True if one of `names` is used to *address* someone (vocative), not just talked about:
    'metis, what do you think', 'hey sava!', 'thanks ekkusu', 'right, metis?', '@sava look'.
    'i told sava about it' or "metis's voice is weird" don't count.
    """
    # Spoken messages run several sentences together: "She said that again. Metis, I'm talking to you."
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    if len(sentences) > 1:
        return any(addressed_by_name(s, names) for s in sentences)
    low = text.lower()
    for name in names:
        n = re.escape(name)
        if re.search(rf"@{n}(?![\w'’])", low):
            return True
        # Set off by commas mid-sentence: "Also, Metis, please do talk with Whimsy."
        if re.search(rf"(?:^|[,;]\s*|\b(?:also|and|so|but|ok|okay|well|yo)\s+){n}\s*,", low):
            return True
        if re.search(rf"^\W*{_GREETING}[\s,!]+{n}(?![\w'’])", low):
            return True
        if re.search(rf"^\W*{n}(?![\w'’]){_AFTER_VOCATIVE}", low):
            return True
        if re.search(rf"(?:,|\b(?:{_GREETING}|right|huh|yeah|no|bro|dude))\s*{n}\s*[?!.~]*\s*$", low):
            return True
        if re.search(rf"\b(?:{_GREETING}|right|huh|yeah)\s+{n}\s*[?!.,~]*\s*$", low):
            return True
        # A name closing a question or exclamation: "what did you think metis?", "you there metis?"
        if re.search(rf"(?<![\w']){n}\s*[?!]+\W*$", low) and not re.search(rf"\b(?:is|was|about|like|with|of|to|for|and|or)\s+{n}\s*[?!]+\W*$", low):
            return True
        # A sentence speaking to "you" that ends with the name: "you're the best metis"
        if re.search(rf"\b(?:you|u|ur|your|you're|youre)\b.*\s{n}\s*[.~]*\s*$", low):
            return True
    return False


_QUESTION_START = re.compile(
    r"^\W*(?:what|whats|what's|who|whos|who's|when|where|why|how|hows|how's|which|is|are|was|were|do|does|did|"
    r"can|could|would|will|should|shall|have|has|any)\b",
    re.IGNORECASE,
)


def is_question(text: str) -> bool:
    return "?" in text or bool(_QUESTION_START.match(text))


def mentions_name(text: str, names: Iterable[str]) -> bool:
    low = text.lower()
    return any(re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", low) for n in names)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    engagement: str = "balanced"          # reserved | balanced | chatty
    continuation_window: float = 180.0    # s: how long an exchange with Metis stays "open"
    recent_window: float = 300.0          # s: what counts as the current conversation
    voice: bool = False
    chime_in_after: float = 0.0           # s of people talking among themselves before Metis joins in (0: never)


def decide(
    turns: Sequence[Turn],
    bot_id: int,
    bot_names: Iterable[str],
    names_by_id: Dict[int, str],
    is_dm: bool = False,
    alone_with_metis: bool = False,
    now: Optional[float] = None,
    settings: Optional[Settings] = None,
) -> Decision:
    """Decide whether the last turn is for Metis (see the module docstring)."""
    s = settings or Settings()
    now = now if now is not None else time.time()
    if not turns or turns[-1].is_metis:
        return Decision(SKIP, "nothing new from a person")
    last = turns[-1]
    metis_names = [v for n in bot_names for v in name_variants(n)]
    text = spoken_names(last.text.strip(), metis_names) if s.voice else last.text.strip()
    others = {uid: name_variants(n) for uid, n in names_by_id.items() if uid not in (bot_id, last.speaker_id)}

    if is_dm:
        return Decision(RESPOND, "direct message")
    if alone_with_metis:
        # Only one person here (e.g. alone in voice with her): they're talking to Metis.
        if is_acknowledgement(text) or is_status_update(text):
            return Decision(SKIP, "acknowledgement / status update that needs no answer")
        return Decision(RESPOND, "the only person here is talking to Metis")

    # 1. Explicitly for Metis
    if bot_id in last.mentions:
        return Decision(RESPOND, "@mentioned Metis")
    if last.reply_to == bot_id:
        return Decision(RESPOND, "replied to Metis")
    if addressed_by_name(text, metis_names):
        return Decision(RESPOND, "addressed Metis by name")

    # 2. Explicitly for someone else
    other_mentions = [m for m in last.mentions if m not in (bot_id, last.speaker_id)]
    if other_mentions:
        return Decision(SKIP, "@mentioned someone else", {"to": other_mentions})
    if last.reply_to is not None and last.reply_to not in (bot_id, last.speaker_id):
        return Decision(SKIP, "replied to someone else", {"to": last.reply_to})
    named = [uid for uid, variants in others.items() if addressed_by_name(text, variants)]
    if named:
        return Decision(SKIP, "addressed someone else by name", {"to": named})

    # Current conversation: recent turns only
    recent = [t for t in turns if now - t.ts <= s.recent_window]
    before = recent[:-1]
    metis_idx = max((i for i, t in enumerate(before) if t.is_metis), default=None)

    # 3. An exchange Metis is already part of
    if metis_idx is not None:
        metis_turn = before[metis_idx]
        # "Yeah." or "Hmm." from someone else doesn't make it their conversation: with okeida's "Yeah." in
        # between, ekkusu's "Huh? What are you talking about?" to Metis was judged to be for someone else.
        since = [t for t in before[metis_idx + 1:] if not is_acknowledgement(t.text)]
        other_humans_since = {t.speaker_id for t in since if t.speaker_id not in (None, last.speaker_id)}
        talking_to_speaker = (metis_turn.to == last.speaker_id
                              or (metis_turn.to is None and _spoke_just_before(before, metis_idx, last.speaker_id)))
        if (talking_to_speaker and not other_humans_since
                and now - metis_turn.ts <= s.continuation_window):
            if is_acknowledgement(text):
                return Decision(SKIP, "acknowledgement that needs no answer")
            return Decision(RESPOND, "continuing an exchange with Metis")
        # The first answer to a question she just asked, whoever gives it: she asked ekkusu "how's your day?",
        # okeida said "Pretty good." and the model thought that was for someone else.
        if (not since and metis_turn.text.rstrip().endswith("?") and now - metis_turn.ts <= s.continuation_window
                and not is_acknowledgement(text)):
            return Decision(RESPOND, "answering Metis's question")

    # 4. Questions to the whole room: humans get the first chance
    if is_room_question(text):
        if s.engagement == "reserved":
            return Decision(SKIP, "question to the room (reserved mode)")
        return Decision(WAIT_OPEN, "question to the room", {"wait": "room"})

    # 5. People talking among themselves: two or more people have spoken since Metis last did
    #    (or recently, if she hasn't spoken at all).
    since_metis = before[metis_idx + 1:] if metis_idx is not None else before[-6:]
    humans = ({t.speaker_id for t in since_metis[-6:] if not t.is_metis and not is_acknowledgement(t.text)}
              | {last.speaker_id})
    if len(humans) >= 2:
        if mentions_name(text, metis_names):
            return Decision(ASK, "people are talking and mention Metis")
        # Joining in now and then, like a friend in the call. Skipping all of it kept her silent for
        # minutes while two people talked, until they asked why she never says anything.
        quiet_since = before[metis_idx].ts if metis_idx is not None else (since_metis[0].ts if since_metis else now)
        humans_since = sum(1 for t in since_metis if not t.is_metis) + 1
        if (s.chime_in_after > 0 and s.engagement != "reserved" and humans_since >= 4
                and now - quiet_since >= s.chime_in_after and not is_acknowledgement(text)
                and len(text.split()) >= 4):
            return Decision(RESPOND, "chiming in after people talked for a while", {"chime": True})
        if s.voice and s.engagement != "reserved" and not is_acknowledgement(text) and not is_status_update(text):
            # In a call the model reads who each line is for, as group-chat bots do: questions to her, "she's
            # actually right" about her, or something for the other person. Word rules ("she/her", "?") missed
            # most of these and misread others.
            return Decision(ASK, "people are talking; who is this for?")
        return Decision(SKIP, "people are talking among themselves")

    if is_acknowledgement(text):
        return Decision(SKIP, "acknowledgement that needs no answer")
    if is_status_update(text):
        return Decision(SKIP, "a status update that doesn't ask for a reply")

    # 6. A lone person with no clear addressee. "Nobody else around" means nobody else spoke lately, not just
    #    since Metis: "what does that mean?" right after the other person spoke was for them.
    if len(humans | {t.speaker_id for t in before if not t.is_metis}) == 1:
        if s.engagement == "chatty":
            return Decision(RESPOND, "someone is talking and nobody else is around (chatty)")
        if is_question(text) and s.engagement != "reserved":
            # Nobody else is around to answer: treat it like a question to the room.
            return Decision(WAIT_OPEN, "a question with nobody else around", {"wait": "lone"})
        return Decision(ASK, "a lone remark with no clear addressee")

    return Decision(ASK, "unclear who this is for")


def _spoke_just_before(turns: Sequence[Turn], metis_idx: int, speaker_id: Optional[int]) -> bool:
    """Was `speaker_id` the last person Metis heard from before her latest turn?"""
    for t in reversed(turns[:metis_idx]):
        if not t.is_metis:
            return t.speaker_id == speaker_id
    return False


# ---------------------------------------------------------------------------
# Model fallback for the ambiguous cases
# ---------------------------------------------------------------------------
ADDRESSEE_CHOICES = ["Metis", "talking about Metis", "another person in the chat", "everyone in the chat",
                     "nobody in particular"]

ADDRESSEE_PROMPT = """You read a Discord chat that includes an AI companion called Metis.
Decide who the LAST message is meant for.
- Metis: it talks to Metis, asks her something, or continues a conversation with her.
- talking about Metis: people talk about Metis to each other, often as "she" or "her" ("she's right", \
"she really likes that"). In a voice chat "she" or "her" without a name usually means Metis.
- another person in the chat: it continues a conversation between people, or answers someone.
- everyone in the chat: it is said to the whole group.
- nobody in particular: thinking out loud, a status update, a reaction.
Answer with one option."""

# Worked examples, given as earlier turns: a small model follows examples far better than the definitions.
ADDRESSEE_EXAMPLES = [
    ("Chat:\nSam: did you finish the essay?\nAna: almost, one page left\nSam: nice, want to grab food after?\n\n"
     "Who is the LAST message meant for?", "another person in the chat"),
    ("Chat:\nSam: is shaco any good?\nMetis (to Sam): He's top tier in the jungle right now.\nAna: no way\n"
     "Sam: wait, she's actually right\n\nWho is the LAST message meant for?", "talking about Metis"),
    ("Chat:\nSam: aatrox is useless now\nAna: yeah he doesn't even heal\n"
     "Ana: admit it, aatrox is a terrible champion, right?\n\nWho is the LAST message meant for?", "Metis"),
    ("Chat:\nSam: brb getting water\n\nWho is the LAST message meant for?", "nobody in particular"),
    ("Chat:\nAna: anyone know a good pizza place around here?\n\nWho is the LAST message meant for?",
     "everyone in the chat"),
]


def addressee_messages(turns: Sequence[Turn], names_by_id: Dict[int, str]) -> List[Dict[str, str]]:
    """The prompt for the addressee check (answers are JSON strings: the call is constrained to the choices)."""
    examples = [m for question, answer in ADDRESSEE_EXAMPLES
                for m in ({"role": "user", "content": question}, {"role": "assistant", "content": f'"{answer}"'})]
    return [{"role": "system", "content": ADDRESSEE_PROMPT}, *examples,
            {"role": "user", "content": "Chat:\n" + transcript_for_model(turns, names_by_id)
                                        + "\n\nWho is the LAST message meant for?"}]


# ---------------------------------------------------------------------------
# Leaving the voice call when asked
# ---------------------------------------------------------------------------
# Cheap word check first, so the model is only asked when leaving could actually be meant.
LEAVE_HINT = re.compile(
    r"\b(leave|leaving|go away|get out|you can go|go now|disconnect|hang up|log ?off|sign off|bye|goodbye|"
    r"good ?night|see (you|ya)|head out|dip|exit|quit|scram|kick(ed)? you)\b",
    re.IGNORECASE,
)

LEAVE_CHOICES = ["Metis should leave the call", "Metis should stay"]

LEAVE_PROMPT = """You read a Discord voice chat that includes an AI companion called Metis.
Decide whether the LAST speaker is asking Metis to leave the voice call now.
- Metis should leave the call: they tell or ask Metis to leave, go, disconnect or hang up \
("Metis, leave the call", "you can go now Metis", "bye Metis, get out of here", or right after \
Metis spoke: "thanks, you can leave now").
- Metis should stay: anything else, e.g. the speaker says THEY are leaving ("I have to go, bye"), \
says bye to someone else, talks about leaving a game or place, or asks whether Metis is leaving.
Answer with one option."""


def might_ask_to_leave(text: str) -> bool:
    return bool(LEAVE_HINT.search(text))


# A plain command needs no model. The model was asked anyway and, after a few in-character replies
# like "I'd never leave", it answered "stay" to "Okay, just leave." and "I told you, Metis, please leave."
_LEAVE_VERB = r"(leave|go away|get out|disconnect|hang up|log ?off)"
_FILLER = (r"(?:ok(?:ay)?|no|yes|yeah|well|so|alright|right|now|then|and|but|uh|um|hey|please|just|"
           r"i said|i told you|i mean|like i said|metis|you can|can you|could you|would you|will you)")
# A clause that starts with the command, after any fillers: "okay just leave", "metis, leave the call".
_LEAVE_COMMAND = re.compile(rf"^(?:{_FILLER}\s+)*{_LEAVE_VERB}(?:\s+(?:now|please|already|the (?:voice|vc|call|channel|chat)\b.*))?$")
# "leave the voice channel" anywhere, unless the speaker is the one leaving.
_LEAVE_THE_CALL = re.compile(r"\b(leave|disconnect from|get out of) the (voice|vc|call|channel|voice chat|voice channel)\b")
_SPEAKER_LEAVING = re.compile(r"\b(i|i'm|im|i'll|we|we're|we'll|gonna|going to|have to|need to|about to|want to)\s+"
                              r"(\w+\s+)?(leave|disconnect|get out)\b")
_NEGATED = re.compile(r"\b(don't|dont|do not|never|not|no need to|shouldn't|why|when|if|after|before|once|until)\b")


def clearly_asks_to_leave(text: str, alone_with_metis: bool) -> bool:
    """
    True for unmistakable commands to leave the call. Bare commands ("just leave") count only if
    they name Metis or nobody else is in the call; otherwise they could be meant for someone else.
    """
    lowered = text.lower().replace("’", "'")
    if "?" in lowered and not re.search(r"\b(can|could|would|will) you\b", lowered):
        return False  # "are you leaving?"
    if not (alone_with_metis or re.search(r"\bmetis\b", lowered)):
        return False
    for clause in re.split(r"[.,!;:]+", lowered):
        clause = re.sub(r"[^a-z' ]", " ", clause)
        clause = re.sub(r"\s+", " ", clause).strip()
        if not clause or _NEGATED.search(clause):
            continue
        if _LEAVE_COMMAND.match(clause):
            return True
        if _LEAVE_THE_CALL.search(clause) and not _SPEAKER_LEAVING.search(clause):
            return True
    return False


def transcript_for_model(turns: Sequence[Turn], names_by_id: Dict[int, str], limit: int = 10) -> str:
    lines = []
    for t in turns[-limit:]:
        who = "Metis" if t.is_metis else t.name
        arrow = ""
        target = t.reply_to if t.reply_to is not None else (t.mentions[0] if t.mentions else None)
        if target is not None:
            arrow = f" (replying to {names_by_id.get(target, 'someone')})"
        elif t.is_metis and t.to is not None:
            arrow = f" (to {names_by_id.get(t.to, 'someone')})"
        lines.append(f"{who}{arrow}: {t.text[:300]}")
    return "\n".join(lines)


def verdict_from_addressee(addressee: str, settings: Optional[Settings] = None, text: str = "") -> Decision:
    s = settings or Settings()
    if addressee == "Metis":
        return Decision(RESPOND, "the model thinks it's for Metis")
    if addressee == "talking about Metis" and s.engagement != "reserved":
        # Like a person hearing friends talk about her, she reacts ("she's actually right" got no reply).
        return Decision(RESPOND, "people are talking about Metis")
    if addressee == "everyone in the chat":
        if s.engagement == "reserved":
            return Decision(SKIP, "said to everyone (reserved mode)")
        if text and not is_question(text) and s.engagement != "chatty":
            # A misheard "Alomed this hoarding" counted as said to everyone and got a reply about hoarding.
            # Remarks to the group are joined in on through chime_in_after instead.
            return Decision(SKIP, "a remark to everyone, not a question")
        return Decision(WAIT_OPEN, "said to everyone")
    if addressee == "nobody in particular" and s.engagement == "chatty":
        return Decision(RESPOND, "a remark Metis can react to (chatty)")
    return Decision(SKIP, f"the model thinks it's for {addressee}")
