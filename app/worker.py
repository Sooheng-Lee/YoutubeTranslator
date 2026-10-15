from copy import deepcopy
from pathlib import Path
import tempfile

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from app.cancellation import CancelledError, CancellationToken
from app.pipeline import TranslationPipeline
from app.services.subtitle_exporter import SubtitleExporter
from app.services.youtube_downloader import YouTubeDownloader


class PipelineWorker(QObject):
    progress = pyqtSignal(str, int)
    job_updated = pyqtSignal(object)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, action, settings, extra=None):
        super().__init__()
        self.action, self.settings, self.extra = action, settings, extra or {}
        self.token = CancellationToken()

    @pyqtSlot()
    def run(self):
        try:
            downloader = YouTubeDownloader()
            report = self.progress.emit
            if self.action == "translate":
                result = TranslationPipeline(downloader=downloader).run(
                    self.settings, self.token, report,
                    lambda job: self.job_updated.emit(deepcopy(job)))
            elif self.action == "inspect":
                result = downloader.inspect(self.settings.url, self.token, report)
            elif self.action == "video":
                info = downloader.inspect(self.settings.url, self.token, report)
                result = downloader.download_video(self.settings.url, info.id, self.settings.output_dir,
                                                   self.settings.quality, self.token, report)
            elif self.action == "subtitle":
                with tempfile.TemporaryDirectory(prefix="yt-subtitle-") as work:
                    source = downloader.download_subtitle(self.settings.url, self.extra["language"],
                                                          self.extra["source"], work, self.token, report)
                    self.token.check()
                    result = SubtitleExporter().convert(source, Path(self.extra["destination"]))
            else:
                raise ValueError("알 수 없는 작업입니다.")
            self.token.check()
            self.result.emit(result)
        except CancelledError as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            message = str(exc)
            if "gated" in message.lower() or "401" in message or "403" in message:
                message = "모델 또는 영상 접근이 거부되었습니다. TranslateGemma 모델 페이지에서 이용 조건에 동의한 뒤 login_huggingface.bat으로 로그인하세요.\n\n" + message
            self.error.emit(message[-2500:])
        finally:
            self.finished.emit()
