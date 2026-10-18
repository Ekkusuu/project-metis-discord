"""
Compatibility patch for discord-ext-voice-recv with Discord's end-to-end encrypted voice (DAVE).

With DAVE, a few incoming frames can't be decrypted yet: they arrive before Discord has told us
which user an audio stream belongs to, or before the encryption session is ready. voice-recv
hands those to the Opus decoder anyway, the decoder raises "corrupted stream", and that single
exception ends voice-recv's receive thread for good (it calls stop_listening()) - Metis then
hears nothing for the rest of the session.

This patch drops undecodable frames instead (a lost 20 ms frame is inaudible) and counts them,
so a real decryption problem shows up in the log as "everything dropped" rather than silence.
Applied from our own code so it survives reinstalling the library.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("metis.voice")

stats = {"decoded": 0, "dropped": 0}
_lock = threading.Lock()
_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return
    from discord.ext.voice_recv import opus as vr_opus
    from discord.opus import OpusError

    original = vr_opus.PacketDecoder._decode_packet

    def _decode_packet(self, packet):
        try:
            result = original(self, packet)
        except OpusError:
            with _lock:
                stats["dropped"] += 1
                dropped, decoded = stats["dropped"], stats["decoded"]
            if dropped == 1 or dropped % 500 == 0:
                print(f"🔐 Voice: dropped {dropped} undecodable frame(s) so far ({decoded} decoded). "
                      "A few at the start of someone speaking is normal with Discord's voice encryption; "
                      "if nothing is ever decoded, incoming audio isn't being decrypted.")
            return packet, b""
        with _lock:
            stats["decoded"] += 1
        return result

    vr_opus.PacketDecoder._decode_packet = _decode_packet
    _applied = True
