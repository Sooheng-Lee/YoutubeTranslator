from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

MODEL_ID = "google/translategemma-12b-it"
# A conservative subset of the officially supported languages.
LANGUAGES = {"한국어": "ko", "영어": "en", "일본어": "ja", "중국어": "zh",
             "독일어": "de", "프랑스어": "fr", "스페인어": "es", "이탈리아어": "it",
             "포르투갈어": "pt", "러시아어": "ru", "아랍어": "ar", "힌디어": "hi"}


class JobStatus(str, Enum):
    WAITING = "대기"
    PREPARING = "준비"
    DOWNLOADING = "다운로드"
    TRANSCRIBING = "STT"
    TRANSLATING = "번역"
    COMPLETED = "완료"
    CANCELLED = "취소"
    FAILED = "실패"


@dataclass
class AppSettings:
    url: str = ""
    source_language: str = "auto"
    target_language: str = "ko"
    whisper_model: str = "small"
    model_path: str = MODEL_ID
    device: str = "cuda"
    precision: str = "4bit"
    output_dir: str = str(Path.home() / "Downloads" / "YouTubeTranslator")
    quality: str = "best"
    save_video: bool = False


@dataclass
class SubtitleSegment:
    id: int
    start: float
    end: float
    original: str
    translated: str = ""
    status: str = "대기"
    error: str = ""


@dataclass
class VideoMetadata:
    id: str
    url: str
    title: str
    duration: float = 0
    subtitles: list[dict] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)


@dataclass
class TranslationJob:
    id: str
    settings: dict
    metadata: dict
    language: str = ""
    segments: list[SubtitleSegment] = field(default_factory=list)
    stt_complete: bool = False
    status: str = JobStatus.WAITING.value
    video_path: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["segments"] = [SubtitleSegment(**s) for s in data.get("segments", [])]
        return cls(**data)


@dataclass
class DownloadResult:
    kind: str
    path: str
    language: str = ""
    source: str = ""
