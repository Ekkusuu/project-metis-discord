"""The DAVE compatibility patch keeps voice receive alive when a frame can't be decoded."""
from types import SimpleNamespace

import backend.discord_bot  # noqa: F401  (applies the patch)
from backend import voice_compat
from discord.ext.voice_recv.opus import PacketDecoder


class _BrokenDecoder:
    def decode(self, data, fec=False):
        from discord.opus import OpusError
        error = OpusError.__new__(OpusError)  # its __init__ needs libopus loaded
        error.code = -4  # OPUS_INVALID_PACKET, reported as "corrupted stream"
        raise error


class _GoodDecoder:
    def decode(self, data, fec=False):
        return b"\x01\x02"


def _decoder(inner):
    d = PacketDecoder.__new__(PacketDecoder)
    d._decoder = inner
    return d


def test_undecodable_frame_is_dropped_not_raised():
    before = voice_compat.stats["dropped"]
    packet = SimpleNamespace(decrypted_data=b"ciphertext")
    out_packet, pcm = PacketDecoder._decode_packet(_decoder(_BrokenDecoder()), packet)
    assert out_packet is packet and pcm == b""
    assert voice_compat.stats["dropped"] == before + 1


def test_good_frames_still_decode():
    packet = SimpleNamespace(decrypted_data=b"opus")
    assert PacketDecoder._decode_packet(_decoder(_GoodDecoder()), packet) == (packet, b"\x01\x02")
