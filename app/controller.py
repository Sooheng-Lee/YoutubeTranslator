from pathlib import Path

from PyQt6.QtCore import QObject, QThread, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from app.models import DownloadResult, TranslationJob, VideoMetadata
from app.repositories.job_repository import JobRepository
from app.repositories.settings_repository import SettingsRepository
from app.services.subtitle_exporter import SubtitleExporter
from app.services.youtube_downloader import YouTubeDownloader
from app.worker import PipelineWorker


class AppController(QObject):
    def __init__(self, window, settings_repository=None, job_repository=None):
        super().__init__(window)
        self.window = window
        self.settings_repository = settings_repository or SettingsRepository()
        self.job_repository = job_repository or JobRepository()
        self.thread = None
        self.worker = None
        self.job = None
        self.metadata = None
        self.closing = False
        self.last_message = "대기"
        window.start_button.clicked.connect(lambda: self.start("translate"))
        window.retry_button.clicked.connect(lambda: self.start("translate"))
        window.inspect_button.clicked.connect(lambda: self.start("inspect"))
        window.video_button.clicked.connect(lambda: self.start("video"))
        window.model_download_button.clicked.connect(lambda: self.start("model"))
        window.cancel_button.clicked.connect(self.cancel)
        window.original_button.clicked.connect(lambda: self.export("original"))
        window.translated_button.clicked.connect(lambda: self.export("translated"))
        window.both_button.clicked.connect(lambda: self.export("both"))
        window.caption_button.clicked.connect(self.download_caption)
        window.folder_button.clicked.connect(self.open_folder)
        window.table.itemChanged.connect(self.edit_translation)
        window.close_handler = self.request_close

    def alert(self, message):
        QMessageBox.warning(self.window, "확인", message)

    def start(self, action, extra=None):
        if self.thread is not None:
            return
        settings = self.window.settings()
        try:
            if action != "model":
                YouTubeDownloader.validate_url(settings.url)
                if not settings.output_dir:
                    raise ValueError("저장 폴더를 지정하세요.")
                output = Path(settings.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                # Check write access using a unique, disposable file.
                import tempfile
                with tempfile.TemporaryFile(dir=output):
                    pass
            if action in {"translate", "model"}:
                if not settings.model_path:
                    raise ValueError("LLM ID Path에 모델 ID 또는 로컬 경로를 입력하세요.")
            self.settings_repository.save(settings)
        except Exception as exc:
            self.alert(str(exc))
            return
        self.window.set_busy(True)
        self.window.status.setText("준비 중")
        self.window.progress.setRange(0, 0)
        self.last_message = "작업 종료"
        self.active_action = action
        self.task_succeeded = False
        self.active_url = settings.url
        self.worker = PipelineWorker(action, settings, extra,
                                     access_token=self.window.access_token.text().strip())
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress)
        self.worker.job_updated.connect(self.update_job)
        self.worker.result.connect(self.on_result)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.finish)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def progress(self, label, percent):
        self.window.status.setText(label)
        self.window.progress.setRange(0, 0 if percent < 0 else 100)
        if percent >= 0:
            self.window.progress.setValue(percent)

    def update_job(self, job):
        self.job = job
        self.window.show_job(job)

    def on_result(self, result):
        self.task_succeeded = True
        if isinstance(result, DownloadResult) and result.kind == "llm":
            self.last_message = f"LLM 모델 다운로드 완료: {result.path}"
            self.window.info.setText(self.last_message)
        elif isinstance(result, VideoMetadata):
            self.metadata = result
            self.inspected_url = self.active_url
            heights = ", ".join(f"{h}p" for h in result.heights) or "확인 불가"
            self.window.info.setText(f"{result.title} · {result.duration:.0f}초 · 제공 화질: {heights}")
            self.window.quality.clear()
            self.window.quality.addItem("최고 화질", "best")
            for height in result.heights:
                self.window.quality.addItem(f"최대 {height}p", str(height))
            self.window.captions.clear()
            for subtitle in result.subtitles:
                name = "수동" if subtitle["source"] == "manual" else "자동 생성"
                self.window.captions.addItem(f"{subtitle['language']} · {name}", subtitle)
            if not result.subtitles:
                self.window.captions.addItem("제공되는 기존 자막이 없습니다. STT 원본 자막을 생성할 수 있습니다.", None)
            self.last_message = "영상 정보 조회 완료"
        elif isinstance(result, TranslationJob):
            self.update_job(result)
            failed = sum(s.status != "완료" for s in result.segments)
            self.last_message = f"{result.status} · {len(result.segments)}개 구간 · 미완료 {failed}개"
            if result.video_path:
                self.last_message += f" · 영상: {result.video_path}"
        else:
            self.last_message = f"저장 완료: {result}"

    def on_error(self, message):
        self.last_message = message
        if not (self.worker and self.worker.token.cancelled):
            self.alert(message)

    def finish(self):
        self.thread = None
        self.worker = None
        self.window.set_busy(False)
        self.window.progress.setRange(0, 100)
        self.window.progress.setValue(100 if self.active_action == "model" and self.task_succeeded else 0)
        self.window.status.setText(self.last_message)
        if self.closing:
            QTimer.singleShot(0, self.window.close)

    def cancel(self):
        if self.worker:
            self.worker.token.cancel()
            self.window.cancel_button.setEnabled(False)
            self.window.status.setText("취소 요청됨 · 현재 파일 다운로드/모델 연산이 끝나면 중단합니다. 완료 파일은 재사용됩니다.")

    def edit_translation(self, item):
        if self.thread is not None or not self.job or item.column() != 4:
            return
        segment = self.job.segments[item.row()]
        segment.translated = item.text().strip()
        segment.status = "완료" if segment.translated else "대기"
        segment.error = ""
        self.job.status = "완료" if all(s.status == "완료" for s in self.job.segments) else "일부 미완료"
        try:
            self.job_repository.save(self.job)
            self.window.table.blockSignals(True)
            self.window.table.item(item.row(), 5).setText(segment.status)
        except OSError as exc:
            self.alert(f"수정 내용 저장 실패: {exc}")
        finally:
            self.window.table.blockSignals(False)

    def confirm_paths(self, paths):
        existing = [str(p) for p in paths if p.exists()]
        if existing:
            return QMessageBox.question(self.window, "파일 덮어쓰기", "다음 파일을 덮어쓸까요?\n" + "\n".join(existing)) == QMessageBox.StandardButton.Yes
        return True

    def export(self, kind):
        if not self.job or not self.job.stt_complete or not self.job.segments:
            self.alert("STT가 완료된 원본 자막이 없습니다.")
            return
        partial = False
        if kind in {"translated", "both"}:
            failed = sum(s.status != "완료" for s in self.job.segments)
            if failed:
                if QMessageBox.question(self.window, "부분 자막 저장", f"미완료 {failed}개 구간을 제외하고 번역 자막을 저장할까요?") != QMessageBox.StandardButton.Yes:
                    return
                partial = True
            if not any(s.status == "완료" for s in self.job.segments):
                self.alert("저장할 번역 완료 구간이 없습니다.")
                return
        ext = self.window.format.currentText()
        output = Path(self.window.output.text())
        video_id = self.job.metadata["id"]
        originals = output / f"{video_id}.{self.job.language}.stt.{ext}"
        translated = output / f"{video_id}.{self.job.settings['target_language']}.translated.{ext}"
        paths = [originals, translated] if kind == "both" else [translated if kind == "translated" else originals]
        if not self.confirm_paths(paths):
            return
        try:
            exporter = SubtitleExporter()
            for path in paths:
                exporter.export(self.job.segments, path, translated=(path == translated), partial=partial)
            self.window.status.setText("저장 완료: " + " / ".join(map(str, paths)))
        except Exception as exc:
            self.alert(f"자막 저장 실패: {exc}")

    def download_caption(self):
        selection = self.window.captions.currentData()
        if not selection or not self.metadata:
            self.alert("영상 정보를 조회한 뒤 기존 자막을 선택하세요.")
            return
        if self.window.url.text().strip() != self.metadata.url:
            # URLs can have aliases; require a fresh lookup to avoid using stale captions.
            if self.window.url.text().strip() != getattr(self, "inspected_url", ""):
                self.alert("URL이 변경되었습니다. 영상 정보를 다시 조회하세요.")
                return
        ext = self.window.format.currentText()
        if ext == "txt":
            self.alert("기존 YouTube 자막은 SRT 또는 VTT를 선택하세요.")
            return
        filename = f"{self.metadata.id}.{selection['language']}.youtube-{selection['source']}.{ext}"
        destination, _ = QFileDialog.getSaveFileName(self.window, "기존 YouTube 자막 저장", str(Path(self.window.output.text()) / filename), f"자막 (*.{ext})")
        if not destination:
            return
        self.start("subtitle", {**selection, "destination": destination})

    def open_folder(self):
        path = Path(self.window.output.text())
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def request_close(self):
        if self.thread is not None:
            if not self.closing and QMessageBox.question(self.window, "종료", "작업을 취소하고 종료할까요? 모델 로딩 중에는 종료까지 시간이 걸릴 수 있습니다.") == QMessageBox.StandardButton.Yes:
                self.closing = True
                self.cancel()
            return False
        self.settings_repository.save(self.window.settings())
        return True
