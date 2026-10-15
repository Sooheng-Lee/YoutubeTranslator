import os
import pytest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from app.controller import AppController
from app.models import AppSettings, SubtitleSegment, TranslationJob
from app.repositories.job_repository import JobRepository
from app.repositories.settings_repository import SettingsRepository
from app.ui.main_window import MainWindow


def test_ui_edit_persists_and_original_export(tmp_path):
    application = QApplication.instance() or QApplication([])
    settings = AppSettings(output_dir=str(tmp_path))
    window = MainWindow(settings)
    controller = AppController(window, SettingsRepository(tmp_path), JobRepository(tmp_path / "jobs"))
    window.access_token.setText("hf_ephemeral_secret")
    from PyQt6.QtWidgets import QLineEdit
    assert window.access_token.echoMode() == QLineEdit.EchoMode.Password
    controller.settings_repository.save(window.settings())
    assert "hf_ephemeral_secret" not in controller.settings_repository.path.read_text(encoding="utf-8")
    job = TranslationJob("fixture", {"target_language": "ko"}, {"id": "fixture"}, language="en",
                         segments=[SubtitleSegment(1, 0, 1, "Hello", "안녕", "완료")], stt_complete=True)
    controller.update_job(job)
    assert not window.table.item(0, 3).flags() & Qt.ItemFlag.ItemIsEditable
    window.table.item(0, 4).setText("수정된 번역")
    restored = controller.job_repository.load("fixture")
    assert restored.segments[0].translated == "수정된 번역"
    controller.export("both")
    assert "수정된 번역" in (tmp_path / "fixture.ko.translated.srt").read_text(encoding="utf-8")
    assert "Hello" in (tmp_path / "fixture.en.stt.srt").read_text(encoding="utf-8")
    window.set_busy(True)
    assert window.original_button.isEnabled()
    assert not window.start_button.isEnabled()
    assert window.cancel_button.isEnabled()
    window.set_busy(False)
    window.close()


def test_start_button_reports_missing_preparation_without_starting_pipeline(tmp_path, monkeypatch):
    from PyQt6.QtCore import QEventLoop, QTimer
    from app.services.preflight import PreparationError, TranslationPreflight

    def fail(*args):
        raise PreparationError("Access 토큰을 입력하세요.")

    monkeypatch.setattr(TranslationPreflight, "check", fail)
    monkeypatch.setattr("app.worker.TranslationPipeline", lambda **kwargs: pytest.fail("Must not start pipeline"))
    application = QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings(url="https://youtu.be/fixture", output_dir=str(tmp_path)))
    controller = AppController(window, SettingsRepository(tmp_path), JobRepository(tmp_path / "jobs"))
    errors = []
    controller.alert = errors.append
    window.start_button.click()
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: loop.quit() if controller.thread is None else None)
    timer.start(10)
    QTimer.singleShot(3000, loop.quit)
    loop.exec()
    timer.stop()
    assert controller.thread is None
    assert errors and "Access 토큰" in errors[0]
    assert window.start_button.isEnabled()
    assert not window.cancel_button.isEnabled()
    window.close()


def test_model_button_downloads_without_url_or_video_output(tmp_path, monkeypatch):
    from PyQt6.QtCore import QEventLoop, QTimer
    from app.services.llm_model_downloader import LLMModelDownloader
    seen = []
    monkeypatch.setattr("app.worker.YouTubeDownloader", lambda: pytest.fail("Model download must not initialize video services"))

    def download(self, model, access_token, cancellation, report):
        seen.append((model, access_token))
        report("LLM 다운로드 완료", 100)
        return tmp_path / "model"

    monkeypatch.setattr(LLMModelDownloader, "download", download)
    application = QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings(url="", output_dir="", device="cpu"))
    window.access_token.setText("hf_fixture")
    controller = AppController(window, SettingsRepository(tmp_path), JobRepository(tmp_path / "jobs"))
    errors = []
    controller.alert = errors.append
    window.model_download_button.click()
    assert not window.model_download_button.isEnabled()
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: loop.quit() if controller.thread is None else None)
    timer.start(10)
    QTimer.singleShot(3000, loop.quit)
    loop.exec()
    timer.stop()
    assert controller.thread is None and not errors
    assert seen == [(window.model.text(), "hf_fixture")]
    assert window.model_download_button.isEnabled()
    assert "LLM 모델 다운로드 완료" in window.status.text()
    assert window.progress.value() == 100
    assert "hf_fixture" not in controller.settings_repository.path.read_text(encoding="utf-8")
    window.close()
