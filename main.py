import logging
from logging.handlers import RotatingFileHandler
import os
import sys


def configure_runtime():
    # Keep torch DLLs discoverable without importing/loading either AI model at startup.
    root = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(root):
        os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
    # PyAV 18 / torch on Windows can deadlock loading AV filter DLLs if torch
    # initializes first. Import only the media library here, never an AI model.
    import av  # noqa: F401


def main():
    configure_runtime()
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from app.controller import AppController
    from app.repositories.settings_repository import SettingsRepository, data_directory
    from app.ui.main_window import MainWindow

    handler = RotatingFileHandler(data_directory() / "app.log", maxBytes=2_000_000,
                                  backupCount=2, encoding="utf-8")
    logging.basicConfig(handlers=[handler], level=logging.WARNING)
    application = QApplication(sys.argv)
    application.setApplicationName("YouTubeTranslator")
    repository = SettingsRepository()
    window = MainWindow(repository.load())
    controller = AppController(window, repository)

    def exception_hook(kind, value, traceback):
        # Do not write source content or URL parameters into logs by default.
        logging.error("Unhandled exception: %s", kind.__name__)
        QMessageBox.critical(window, "오류", str(value))

    sys.excepthook = exception_hook
    window.show()
    # Optional deterministic startup check, with the regular event loop.
    if "--smoke-test" in sys.argv:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(500, application.quit)
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
