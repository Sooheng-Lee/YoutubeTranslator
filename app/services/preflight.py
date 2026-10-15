"""Validate translation prerequisites before any YouTube download or STT work."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

from app.models import LANGUAGES, MODEL_ID
from app.repositories.settings_repository import data_directory


class PreparationError(ValueError):
    pass


class TranslationPreflight:
    def check(self, settings, access_token, cancellation, report):
        report("번역 준비사항 확인 중", -1)
        cancellation.check()
        if not settings.model_path.strip():
            raise PreparationError("LLM ID Path를 입력하세요.")
        if settings.device not in {"cpu", "cuda"} or settings.precision not in {"4bit", "bfloat16", "float16", "float32"}:
            raise PreparationError("실행 장치와 정밀도 설정을 확인하세요.")
        if settings.device == "cpu" and settings.precision != "float32":
            raise PreparationError("CPU 번역은 float32를 선택하세요. 이 PC에서는 CUDA / 4bit를 권장합니다.")
        if settings.target_language not in LANGUAGES.values() or settings.source_language not in {"auto", *LANGUAGES.values()}:
            raise PreparationError("지원하는 원본 및 목표 언어를 선택하세요.")
        self._check_dependencies()
        self._check_media(cancellation)
        report("LLM 모델 파일 및 접근 권한 확인 중", -1)
        self._check_model(settings, access_token, cancellation)
        cancellation.check()
        report("실행 장치와 메모리 확인 중", -1)
        self._check_runtime(settings)
        cancellation.check()
        report("준비사항 확인 완료", 100)

    @staticmethod
    def _check_dependencies():
        required = {"torch": "torch", "transformers": "transformers", "accelerate": "accelerate",
                    "faster_whisper": "faster-whisper", "av": "av", "huggingface_hub": "huggingface-hub",
                    "bitsandbytes": "bitsandbytes", "yt_dlp": "yt-dlp", "imageio_ffmpeg": "imageio-ffmpeg",
                    "psutil": "psutil", "sentencepiece": "sentencepiece"}
        missing = [name for module, name in required.items() if importlib.util.find_spec(module) is None]
        if missing:
            raise PreparationError("필수 모듈이 없습니다: " + ", ".join(missing) + "\nsetup.bat을 실행하세요.")

    @staticmethod
    def _check_media(cancellation):
        from app.services.audio_processor import AudioProcessor
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            subprocess.run([AudioProcessor().ffmpeg, "-version"], check=True, capture_output=True,
                           timeout=10, creationflags=flags)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            raise PreparationError("FFmpeg를 실행할 수 없습니다. setup.bat으로 다시 설치하세요.") from exc
        cancellation.check()
        deno = Path(__file__).resolve().parents[2] / ".tools" / "deno.exe"
        if not deno.is_file() and not shutil.which("deno") and not shutil.which("node"):
            raise PreparationError("YouTube 다운로드용 JavaScript 런타임이 없습니다. setup.bat을 실행하세요.")

    @staticmethod
    def _local_files(path):
        """Check all referenced weight shards and tokenizer files without loading weights."""
        if not (path / "config.json").is_file() or not (path / "tokenizer_config.json").is_file():
            raise PreparationError("로컬 모델의 config.json 또는 tokenizer_config.json이 없습니다.")
        if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
            raise PreparationError("로컬 모델의 tokenizer.json 또는 tokenizer.model이 없습니다.")
        try:
            config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            text = config.get("text_config", config)
            if text.get("hidden_size") != 3840 or text.get("num_hidden_layers") != 48:
                raise PreparationError("현재 번역 엔진은 TranslateGemma 12B를 지원합니다. 해당 모델을 지정하세요.")
            index = path / "model.safetensors.index.json"
            if index.is_file():
                weights = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            else:
                weights = {"model.safetensors"}
            if not weights:
                raise PreparationError("로컬 모델의 가중치 목록이 비어 있습니다.")
            for name in weights:
                file = (path / name).resolve()
                if not file.is_relative_to(path.resolve()) or not file.is_file() or file.stat().st_size == 0:
                    raise PreparationError("로컬 모델의 가중치 파일이 누락되었거나 비어 있습니다. 다운로드를 완료하세요.")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, PreparationError):
                raise
            raise PreparationError("로컬 모델 설정 또는 가중치 인덱스를 읽을 수 없습니다.") from exc

    def _check_model(self, settings, access_token, cancellation):
        model = settings.model_path.strip()
        if model != MODEL_ID:
            path = Path(model).expanduser()
            if not path.is_dir():
                raise PreparationError("LLM ID Path가 올바르지 않습니다. 현재는 google/translategemma-12b-it 또는 해당 모델의 로컬 폴더를 지원합니다.")
            self._local_files(path)
            return
        cache = data_directory() / "models" / "translategemma-12b-it"
        try:
            self._local_files(cache)
            return  # Fully downloaded model can be used without network/authentication.
        except PreparationError:
            pass
        from huggingface_hub import HfApi, get_hf_file_metadata, get_token, hf_hub_url
        credential = access_token or get_token()
        if not credential:
            raise PreparationError("모델 다운로드 준비가 필요합니다. Access 토큰을 입력하거나 login_huggingface.bat으로 로그인하세요.\nHugging Face에서 TranslateGemma 모델 이용 조건에도 동의해야 합니다.")
        try:
            info = HfApi().model_info(MODEL_ID, token=credential, files_metadata=True, timeout=15)
            cancellation.check()
            weights = [f for f in info.siblings if f.rfilename.endswith(".safetensors")]
            if not weights:
                raise PreparationError("다운로드 가능한 모델 가중치를 찾을 수 없습니다.")
            # Repository metadata alone can be public even when weights are gated.
            get_hf_file_metadata(hf_hub_url(MODEL_ID, weights[0].rfilename), token=credential, timeout=15)
            cancellation.check()
            required = sum(f.size or 0 for f in weights if not (cache / f.rfilename).is_file()
                           or (cache / f.rfilename).stat().st_size != f.size)
            if shutil.disk_usage(data_directory()).free < required + 2 * 1024**3:
                raise PreparationError(f"모델 저장 드라이브 공간이 부족합니다. 약 {required / 1024**3 + 2:.1f}GB 이상의 여유 공간을 확보하세요.")
        except PreparationError:
            raise
        except Exception as exc:
            from app.cancellation import CancelledError
            if isinstance(exc, CancelledError):
                raise
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {401, 403}:
                raise PreparationError("LLM 접근 권한이 없습니다. Access 토큰의 읽기 권한과 TranslateGemma 모델 이용 조건 동의 여부를 확인하세요.") from exc
            raise PreparationError("LLM 다운로드 준비를 확인할 수 없습니다. 네트워크 연결과 Access 토큰을 확인한 뒤 재시도하세요.") from exc

    @staticmethod
    def _check_runtime(settings):
        import av  # noqa: F401 -- initialize media DLLs before torch
        import torch
        import psutil
        if settings.device == "cuda":
            if not torch.cuda.is_available():
                raise PreparationError("CUDA GPU를 사용할 수 없습니다. GPU 드라이버 및 CUDA PyTorch 설치를 확인하세요.")
            if settings.precision == "4bit":
                import bitsandbytes  # noqa: F401 -- verify native library import
            required = {"4bit": 8, "bfloat16": 27, "float16": 27, "float32": 54}[settings.precision]
            free, _ = torch.cuda.mem_get_info()
            if free < required * 1024**3:
                raise PreparationError(f"GPU 여유 메모리가 부족합니다. {settings.precision} 모드는 약 {required}GB 이상이 필요하며 현재 {free / 1024**3:.1f}GB입니다. 다른 GPU 프로그램을 종료하거나 4bit를 선택하세요.")
            if settings.precision != "float32" and not torch.cuda.is_bf16_supported():
                raise PreparationError("TranslateGemma CUDA 실행에는 BF16 지원 GPU가 필요합니다. GPU와 PyTorch 설정을 확인하세요.")
        elif psutil.virtual_memory().available < 56 * 1024**3:
            raise PreparationError("TranslateGemma 12B CPU/float32 실행에는 약 56GB 이상의 여유 RAM이 필요합니다. CUDA / 4bit를 선택하세요.")
