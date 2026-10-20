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
    _patch_reader()
    _applied = True


def _patch_reader() -> None:
    """
    voice-recv logs "CryptoError decoding packet data" at ERROR for every packet it can't decrypt.
    While Metis talks, Discord sends RTCP reports about her outgoing audio several times a second
    that it can't decrypt: harmless (there is nothing to hear in them), but it filled the window
    with errors. Those are dropped quietly; a failure on actual audio (RTP) is still reported,
    once per second per stream, with what's needed to diagnose it.
    """
    import time

    from discord.ext.voice_recv import reader as vr_reader, rtp
    from nacl.exceptions import CryptoError

    original_callback = vr_reader.AudioReader.callback
    last_report: dict = {}
    failures: dict = {}

    def callback(self, packet_data: bytes) -> None:
        if len(packet_data) < 12:
            return original_callback(self, packet_data)
        rtcp = rtp.is_rtcp(packet_data)
        try:
            if rtcp:
                self.decryptor.decrypt_rtcp(packet_data)
            else:
                self.decryptor.decrypt_rtp(rtp.decode_rtp(packet_data))
        except CryptoError:
            if rtcp:
                return
            ssrc = int.from_bytes(packet_data[8:12], "big")
            failures[ssrc] = failures.get(ssrc, 0) + 1
            now = time.monotonic()
            if now - last_report.get(ssrc, 0) >= 1.0:
                last_report[ssrc] = now
                user_id = self.voice_client._get_id_from_ssrc(ssrc)
                print(f"🔐 Voice: can't decrypt audio from ssrc {ssrc} (user {user_id}): "
                      f"{failures[ssrc]} packet(s) so far; header {packet_data[:2].hex()}, {len(packet_data)} bytes")
            return
        except Exception:
            pass  # let voice-recv handle and log anything else as before
        return original_callback(self, packet_data)

    vr_reader.AudioReader.callback = callback
