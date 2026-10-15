import wave

import pytest

from app.cancellation import CancellationToken, CancelledError
from app.services.audio_processor import AudioProcessor


def test_ffmpeg_produces_16khz_mono_wav(tmp_path):
    source = tmp_path / "stereo.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(44100)
        output.writeframes(b"\x00\x00\x00\x00" * 4410)
    destination = AudioProcessor().normalize(source, tmp_path / "normalized.wav", CancellationToken())
    with wave.open(str(destination), "rb") as audio:
        assert audio.getframerate() == 16000
        assert audio.getnchannels() == 1
        assert audio.getnframes() == 1600


def test_pre_cancelled_audio_terminates_child(tmp_path):
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CancelledError):
        AudioProcessor().normalize(tmp_path / "missing.wav", tmp_path / "out.wav", token)
