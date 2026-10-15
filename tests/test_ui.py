import os
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
