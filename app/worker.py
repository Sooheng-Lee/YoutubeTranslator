from copy import deepcopy
from pathlib import Path
import tempfile
import re

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from app.cancellation import CancelledError, CancellationToken
from app.pipeline import TranslationPipeline
from app.services.subtitle_exporter import SubtitleExporter
from app.services.youtube_downloader import YouTubeDownloader
from app.services.preflight import TranslationPreflight, PreparationError
from app.services.llm_model_downloader import LLMModelDownloader
from app.models import DownloadResult


class PipelineWorker(QObject):
    progress = pyqtSignal(str, int)
    job_updated = pyqtSignal(object)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, action, settings, extra=None, access_token=None):
        super().__init__()
        self.action, self.settings, self.extra = action, settings, extra or {}
        self.token = CancellationToken()
        self.access_token = access_token or None

    @pyqtSlot()
    def run(self):
        try:
            report = self.progress.emit
            if self.action == "translate":
                try:
                    TranslationPreflight().check(self.settings, self.access_token, self.token, report)
                except PreparationError as exc:
                    raise PreparationError("번역 준비가 완료되지 않았습니다.\n\n" + str(exc)) from exc
                except CancelledError:
                    raise
                except Exception as exc:
                    raise PreparationError("번역 준비가 완료되지 않았습니다.\n실행 모듈 또는 장치 초기화에 실패했습니다. setup.bat과 check_environment.py로 환경을 확인하세요.\n\n" + str(exc)) from exc
            downloader = None if self.action == "model" else YouTubeDownloader()
            if self.action == "model":
                path = LLMModelDownloader().download(self.settings.model_path, self.access_token, self.token, report)
                result = DownloadResult(kind="llm", path=str(path), source=self.settings.model_path)
            elif self.action == "translate":
                result = TranslationPipeline(downloader=downloader, access_token=self.access_token).run(
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
            if self.access_token:
                message = message.replace(self.access_token, "[토큰 숨김]")
            message = re.sub(r"hf_[A-Za-z0-9]+", "[토큰 숨김]", message)
            if "gated" in message.lower() or "401" in message or "403" in message:
                message = "모델 또는 영상 접근이 거부되었습니다. TranslateGemma 모델 이용 조건에 동의하고 Access 토큰의 읽기 권한을 확인하세요. 토큰 입력 또는 login_huggingface.bat 로그인을 사용할 수 있습니다.\n\n" + message
            self.error.emit(message[-2500:])
        finally:
            self.access_token = None
            self.finished.emit()
