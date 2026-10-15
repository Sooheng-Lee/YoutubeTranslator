import sys
from types import SimpleNamespace

import pytest

from app.cancellation import CancellationToken
from app.models import AppSettings
from app.services.whisper_transcriber import WhisperTranscriber


@pytest.mark.parametrize("cuda,fail_cuda,expected", [(True, False, "cuda"), (False, False, "cpu"), (True, True, "cpu")])
def test_select_gpu_and_fallback_without_duplicate_segments(monkeypatch, tmp_path, cuda, fail_cuda, expected):
    calls = []

    class Model:
        def __init__(self, path, device, compute_type):
            calls.append((device, compute_type))
            self.device = device

        def transcribe(self, *args, **kwargs):
            def segments():
                if self.device == "cuda" and fail_cuda:
                    raise RuntimeError("fixture CUDA initialization failure")
                yield SimpleNamespace(start=0, end=1, text="hello")
            return segments(), SimpleNamespace(language="en", duration=1)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", SimpleNamespace(download_model=lambda *args, **kwargs: str(tmp_path)))
    monkeypatch.setattr(WhisperTranscriber, "_cuda_configuration", staticmethod(lambda: ("cuda", "float16") if cuda else None))
    transcriber = WhisperTranscriber()
    segments, messages = [], []
    language = transcriber.transcribe("audio.wav", AppSettings(), CancellationToken(), lambda *args: messages.append(args), segments.append)
    assert language == "en" and len(segments) == 1
    assert transcriber.device == expected
    assert calls[-1] == (expected, "float16" if expected == "cuda" else "int8")
    if fail_cuda:
        assert calls == [("cuda", "float16"), ("cpu", "int8")]
        assert any("전환" in label for label, _ in messages)
