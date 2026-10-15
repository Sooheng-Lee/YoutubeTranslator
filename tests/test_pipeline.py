from dataclasses import asdict

import pytest

from app.cancellation import CancellationToken, CancelledError
from app.models import AppSettings, SubtitleSegment, VideoMetadata
from app.pipeline import TranslationPipeline
from app.repositories.job_repository import JobRepository


class Downloader:
    def inspect(self, *args):
        return VideoMetadata("fixture", "https://youtu.be/fixture", "Fixture", 2)

    def download_audio(self, *args):
        return "fixture.audio"


class Audio:
    def normalize(self, *args):
        return "fixture.wav"


class STT:
    calls = 0

    def transcribe(self, path, settings, token, report, callback):
        self.calls += 1
        callback(SubtitleSegment(1, 0, 1, "hello"))
        callback(SubtitleSegment(2, 1, 2, "world"))
        return "en"

    def unload(self):
        pass


class Translator:
    def __init__(self):
        self.calls = []
        self.fail = True

    def load(self, *args):
        pass

    def translate(self, text, source, target, token):
        self.calls.append(text)
        if self.fail and text == "world":
            raise ValueError("fixture failure")
        return "번역:" + text

    def unload(self):
        pass


def test_retry_retains_success_and_reuses_stt(tmp_path):
    stt, translator = STT(), Translator()
    pipeline = TranslationPipeline(JobRepository(tmp_path), Downloader(), Audio(), stt, translator)
    settings = AppSettings(url="https://youtu.be/fixture")
    first = pipeline.run(settings, CancellationToken(), lambda *a: None, lambda *a: None)
    assert first.status == "일부 실패"
    assert [s.id for s in first.segments] == [1, 2]
    translator.fail = False
    second = pipeline.run(settings, CancellationToken(), lambda *a: None, lambda *a: None)
    assert second.status == "완료"
    assert stt.calls == 1
    assert translator.calls == ["hello", "world", "world"]
    assert [(s.start, s.end) for s in second.segments] == [(0, 1), (1, 2)]
    # Changing target invalidates translation, but not the completed STT cache.
    settings.target_language = "ja"
    pipeline.run(settings, CancellationToken(), lambda *a: None, lambda *a: None)
    assert stt.calls == 1
    assert translator.calls[-2:] == ["hello", "world"]


def test_cancel_during_translation_preserves_stt(tmp_path):
    token = CancellationToken()

    class CancellingTranslator(Translator):
        def translate(self, text, source, target, token):
            token.cancel()
            token.check()

    repository = JobRepository(tmp_path)
    pipeline = TranslationPipeline(repository, Downloader(), Audio(), STT(), CancellingTranslator())
    snapshots = []
    with pytest.raises(CancelledError):
        pipeline.run(AppSettings(url="https://youtu.be/fixture"), token, lambda *a: None,
                     lambda job: snapshots.append(job.to_dict()))
    job = repository.load(snapshots[-1]["id"])
    assert job.status == "취소" and job.stt_complete
    assert len(job.segments) == 2
