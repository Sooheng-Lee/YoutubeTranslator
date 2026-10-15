import gc

from app.models import SubtitleSegment
from app.repositories.settings_repository import data_directory


class WhisperTranscriber:
    def __init__(self):
        self.model = None

    def transcribe(self, path, settings, token, report, on_segment):
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
        token.check()
        # CPU STT avoids requiring a separately installed cuDNN runtime on Windows.
        report("Whisper 모델 준비 (CPU/int8)", -1)
        model_path = download_model(settings.whisper_model, output_dir=str(
            data_directory() / "models" / "whisper" / settings.whisper_model))
        token.check()
        self.model = WhisperModel(model_path, device="cpu", compute_type="int8")
        token.check()
        segments, info = self.model.transcribe(str(path), language=None if settings.source_language == "auto" else settings.source_language,
                                               task="transcribe", vad_filter=True)
        for index, segment in enumerate(segments, 1):
            token.check()
            text = segment.text.strip()
            if text and segment.end > segment.start:
                on_segment(SubtitleSegment(index, segment.start, segment.end, text))
            report("음성 인식", min(100, int(segment.end / max(info.duration, 1) * 100)))
        return info.language

    def unload(self):
        self.model = None
        gc.collect()
