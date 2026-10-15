from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from app.cancellation import CancelledError
from app.models import TranslationJob
from app.repositories.job_repository import JobRepository
from app.repositories.settings_repository import write_json
from app.services.audio_processor import AudioProcessor
from app.services.translategemma_translator import TranslateGemmaTranslator
from app.services.whisper_transcriber import WhisperTranscriber
from app.services.youtube_downloader import YouTubeDownloader


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:24]


class TranslationPipeline:
    def __init__(self, repository=None, downloader=None, audio=None, transcriber=None, translator=None, access_token=None):
        self.repository = repository or JobRepository()
        self.downloader = downloader or YouTubeDownloader()
        self.audio = audio or AudioProcessor()
        self.transcriber = transcriber or WhisperTranscriber()
        self.translator = translator or TranslateGemmaTranslator(access_token=access_token)

    def run(self, settings, token, report, on_job):
        from app.models import SubtitleSegment
        metadata = self.downloader.inspect(settings.url, token, report)
        stt_key = fingerprint({"video": metadata.id, "model": settings.whisper_model,
                               "source": settings.source_language, "vad": True, "sample_rate": 16000,
                               "device_policy": "cuda-auto-fallback-cpu", "version": 2})
        job_id = fingerprint({"stt": stt_key, "target": settings.target_language,
                              "llm": settings.model_path, "precision": settings.precision, "version": 1})
        job = self.repository.load(job_id)
        if job is None:
            job = TranslationJob(job_id, asdict(settings), asdict(metadata))
        else:
            job.settings = asdict(settings)
        work = self.repository.workspace(stt_key)
        stt_path = work / "transcript.json"

        def publish():
            self.repository.save(job)
            on_job(job)

        try:
            job.status = "준비"
            publish()
            if settings.save_video:
                job.video_path = str(self.downloader.download_video(settings.url, metadata.id, settings.output_dir,
                                                                    settings.quality, token, report))
            if not job.stt_complete and stt_path.exists():
                try:
                    saved = json.loads(stt_path.read_text(encoding="utf-8"))
                    job.segments = [SubtitleSegment(**s) for s in saved["segments"]]
                    job.language = saved["language"]
                    job.stt_complete = True
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if not job.stt_complete:
                source = Path(job.video_path) if settings.save_video and job.video_path else self.downloader.download_audio(settings.url, work, token, report)
                report("오디오 전처리", -1)
                audio_path = self.audio.normalize(source, work / "normalized.wav", token)
                job.status = "STT"
                job.segments = []

                def add_segment(segment):
                    job.segments.append(segment)
                    on_job(job)

                try:
                    job.language = self.transcriber.transcribe(audio_path, settings, token, report, add_segment)
                finally:
                    self.transcriber.unload()
                job.stt_complete = True
                write_json(stt_path, {"language": job.language, "segments": [asdict(s) for s in job.segments]})
                publish()
            if not job.segments:
                job.status = "완료 (음성 없음)"
                publish()
                return job
            token.check()
            pending = [s for s in job.segments if s.status != "완료"]
            if job.language == settings.target_language:
                for segment in pending:
                    segment.translated, segment.status, segment.error = segment.original, "완료", ""
                report("원본과 목표 언어가 같아 번역 생략", 100)
            elif pending:
                job.status = "번역"
                publish()
                try:
                    self.translator.load(settings, token, report)
                    for index, segment in enumerate(job.segments):
                        token.check()
                        if segment.status == "완료":
                            continue
                        try:
                            segment.translated = self.translator.translate(segment.original, job.language,
                                                                          settings.target_language, token)
                            segment.status, segment.error = "완료", ""
                        except CancelledError:
                            raise
                        except Exception as exc:
                            segment.status, segment.error = "실패", str(exc)
                            # OOM affects all remaining segments; preserve progress and return control.
                            if "out of memory" in str(exc).lower():
                                publish()
                                raise RuntimeError("GPU 메모리가 부족합니다. 다른 GPU 프로그램을 닫고 4bit 모드로 재시도하세요.") from exc
                        report(f"번역 {index + 1}/{len(job.segments)}", int((index + 1) / len(job.segments) * 100))
                        publish()
                finally:
                    self.translator.unload()
            job.status = "일부 실패" if any(s.status != "완료" for s in job.segments) else "완료"
            publish()
            return job
        except CancelledError:
            job.status = "취소"
            publish()
            raise
        except Exception:
            job.status = "실패"
            publish()
            raise
