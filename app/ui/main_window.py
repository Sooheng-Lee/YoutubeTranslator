from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
                            QHBoxLayout, QLabel, QLineEdit, QMainWindow, QProgressBar,
                            QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
                            QHeaderView)

from app.models import AppSettings, LANGUAGES


class MainWindow(QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.setWindowTitle("YouTube Translator · TranslateGemma 12B")
        self.resize(1180, 850)
        self.busy = False
        self.close_handler = None
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        inputs = QGroupBox("영상 및 번역 설정")
        form = QFormLayout(inputs)
        self.url = QLineEdit(settings.url)
        self.url.setPlaceholderText("https://www.youtube.com/watch?v=...")
        form.addRow("YouTube URL", self.url)
        self.source, self.target = QComboBox(), QComboBox()
        self.source.addItem("자동 감지", "auto")
        for name, code in LANGUAGES.items():
            self.source.addItem(f"{name} ({code})", code)
            self.target.addItem(f"{name} ({code})", code)
        self.select(self.source, settings.source_language)
        self.select(self.target, settings.target_language)
        languages = QHBoxLayout()
        languages.addWidget(QLabel("원본"))
        languages.addWidget(self.source)
        languages.addWidget(QLabel("목표"))
        languages.addWidget(self.target)
        form.addRow("언어", languages)
        self.whisper = QComboBox()
        self.whisper.addItems(["tiny", "base", "small", "medium", "large-v3", "turbo"])
        self.whisper.setCurrentText(settings.whisper_model)
        form.addRow("STT 모델 (CUDA 자동 / CPU 대체)", self.whisper)
        self.model = QLineEdit(settings.model_path)
        self.access_token = QLineEdit()
        self.access_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.access_token.setPlaceholderText("hf_... (이 세션에서만 사용)")
        self.access_token.setToolTip("모델 다운로드용 Hugging Face 읽기 토큰. 설정 파일에 저장하지 않습니다. 기존 로그인 또는 완전한 로컬 모델이 있으면 생략할 수 있습니다.")
        model_row = QHBoxLayout()
        model_row.addWidget(self.model, 3)
        model_row.addWidget(QLabel("Access 토큰"))
        model_row.addWidget(self.access_token, 2)
        self.model_download_button = QPushButton("LLM 모델 다운로드")
        self.model_download_button.setToolTip("LLM ID Path의 모델을 다운로드합니다. 영상 URL이나 GPU는 필요하지 않습니다.")
        model_row.addWidget(self.model_download_button)
        form.addRow("LLM ID Path", model_row)
        self.device, self.precision = QComboBox(), QComboBox()
        self.device.addItems(["cuda", "cpu"])
        self.precision.addItems(["4bit", "float16", "float32"])
        self.device.setCurrentText(settings.device)
        self.precision.setCurrentText(settings.precision)
        runtime = QHBoxLayout()
        runtime.addWidget(self.device)
        runtime.addWidget(self.precision)
        form.addRow("번역 실행 장치 / 정밀도", runtime)
        self.output = QLineEdit(settings.output_dir)
        browse = QPushButton("폴더 선택")
        browse.clicked.connect(self.choose_folder)
        output = QHBoxLayout()
        output.addWidget(self.output)
        output.addWidget(browse)
        form.addRow("저장 폴더", output)
        self.quality = QComboBox()
        self.quality.addItem("최고 화질", "best")
        for height in [2160, 1440, 1080, 720, 480, 360]:
            self.quality.addItem(f"최대 {height}p", str(height))
        self.select(self.quality, settings.quality)
        self.save_video = QCheckBox("번역 시 영상도 함께 저장")
        self.save_video.setChecked(settings.save_video)
        quality = QHBoxLayout()
        quality.addWidget(self.quality)
        quality.addWidget(self.save_video)
        form.addRow("영상 다운로드", quality)
        layout.addWidget(inputs)
        self.inputs = inputs
        actions = QHBoxLayout()
        self.inspect_button = QPushButton("영상 정보 / 기존 자막 조회")
        self.start_button = QPushButton("번역 시작")
        self.retry_button = QPushButton("미완료 구간 재시도")
        self.video_button = QPushButton("영상만 다운로드")
        self.cancel_button = QPushButton("취소")
        self.cancel_button.setEnabled(False)
        for button in [self.inspect_button, self.start_button, self.retry_button, self.video_button, self.cancel_button]:
            actions.addWidget(button)
        layout.addLayout(actions)
        self.info = QLabel("영상만 다운로드할 때는 번역 모델을 준비하지 않습니다.")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        captions = QHBoxLayout()
        self.captions = QComboBox()
        self.captions.addItem("기존 자막: 먼저 영상 정보를 조회하세요.", None)
        self.caption_button = QPushButton("선택한 YouTube 자막 저장")
        captions.addWidget(self.captions, 1)
        captions.addWidget(self.caption_button)
        layout.addLayout(captions)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ID", "시작 (초)", "종료 (초)", "원문", "번역문 (수정 가능)", "상태"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.setWordWrap(True)
        layout.addWidget(self.table, 1)
        exports = QHBoxLayout()
        self.format = QComboBox()
        self.format.addItems(["srt", "vtt", "txt"])
        self.original_button = QPushButton("원본 자막 저장")
        self.translated_button = QPushButton("번역 자막 저장")
        self.both_button = QPushButton("두 자막 함께 저장")
        self.folder_button = QPushButton("저장 폴더 열기")
        for widget in [self.format, self.original_button, self.translated_button, self.both_button, self.folder_button]:
            exports.addWidget(widget)
        layout.addLayout(exports)
        self.progress = QProgressBar()
        self.status = QLabel("대기")
        self.status.setWordWrap(True)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(QLabel("STT: CUDA 사용 가능 시 GPU, 그 외 CPU/int8 · 번역 기본: CUDA/4bit · 모델 다운로드 및 접근 동의 필요"))
        self.setStyleSheet("QGroupBox { font-weight: bold; } QPushButton { padding: 7px; } QLineEdit, QComboBox { min-height: 25px; }")

    @staticmethod
    def select(combo, data):
        index = combo.findData(data)
        if index >= 0:
            combo.setCurrentIndex(index)

    def choose_folder(self):
        selected = QFileDialog.getExistingDirectory(self, "저장 폴더", self.output.text())
        if selected:
            self.output.setText(selected)

    def settings(self):
        return AppSettings(self.url.text().strip(), self.source.currentData(), self.target.currentData(),
                           self.whisper.currentText(), self.model.text().strip(), self.device.currentText(),
                           self.precision.currentText(), self.output.text().strip(), self.quality.currentData(),
                           self.save_video.isChecked())

    def set_busy(self, busy):
        self.busy = busy
        self.inputs.setEnabled(not busy)
        for button in [self.inspect_button, self.start_button, self.retry_button, self.video_button, self.caption_button]:
            button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        # Original subtitles may be exported during translation; editing requires idle state.
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers if busy else
                                   QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.EditKeyPressed)

    def show_job(self, job):
        self.table.blockSignals(True)
        self.table.setRowCount(len(job.segments))
        for row, segment in enumerate(job.segments):
            for col, value in enumerate([segment.id, f"{segment.start:.3f}", f"{segment.end:.3f}", segment.original,
                                         segment.translated, segment.status]):
                item = QTableWidgetItem(str(value))
                if col != 4:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if segment.error:
                    item.setToolTip(segment.error)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)

    def closeEvent(self, event):
        if self.close_handler and not self.close_handler():
            event.ignore()
        else:
            event.accept()
