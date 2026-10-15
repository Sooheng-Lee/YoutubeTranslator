import gc
import ctypes
import os
from pathlib import Path
import sys

from app.models import SubtitleSegment
from app.repositories.settings_repository import data_directory

_dll_handles = []
_cuda_libraries_loaded = False


class WhisperTranscriber:
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"

    @staticmethod
    def _cuda_configuration():
        global _cuda_libraries_loaded
        import av  # noqa: F401 -- initialize media DLLs before torch on Windows
        import torch
        import ctranslate2

        if not torch.cuda.is_available() or ctranslate2.get_cuda_device_count() == 0:
            return None
        if os.name == "nt" and not _cuda_libraries_loaded:
            directory = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
            if directory.is_dir() and not _dll_handles:
                _dll_handles.append(os.add_dll_directory(str(directory)))
                os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            # Check cuBLAS/cuDNN before native inference; missing DLLs can abort a process.
            for name in ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll",
                         "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_graph64_9.dll",
                         "cudnn_engines_precompiled64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
                         "cudnn_heuristic64_9.dll"):
                _dll_handles.append(ctypes.WinDLL(name))
            _cuda_libraries_loaded = True
        supported = ctranslate2.get_supported_compute_types("cuda", device_index=0)
        for compute_type in ("float16", "int8_float16", "float32"):
            if compute_type in supported:
                return "cuda", compute_type
        return None

    def transcribe(self, path, settings, token, report, on_segment):
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
        token.check()
        report("Whisper 모델 및 CUDA 실행 환경 확인 중", -1)
        model_path = download_model(settings.whisper_model, output_dir=str(
            data_directory() / "models" / "whisper" / settings.whisper_model))
        token.check()
        try:
            configuration = self._cuda_configuration()
        except (RuntimeError, OSError, ImportError, ValueError):
            configuration = None
            report("Whisper CUDA 실행 라이브러리를 사용할 수 없어 CPU/int8로 전환", -1)
        self.device, self.compute_type = configuration or ("cpu", "int8")
        while True:
            emitted = 0
            token.check()
            report(f"Whisper 모델 로딩 ({self.device}/{self.compute_type})", -1)
            try:
                self.model = WhisperModel(model_path, device=self.device, compute_type=self.compute_type)
                token.check()
                segments, info = self.model.transcribe(str(path), language=None if settings.source_language == "auto" else settings.source_language,
                                                       task="transcribe", vad_filter=True)
                for index, segment in enumerate(segments, 1):
                    token.check()
                    text = segment.text.strip()
                    if text and segment.end > segment.start:
                        on_segment(SubtitleSegment(index, segment.start, segment.end, text))
                        emitted += 1
                    report(f"음성 인식 ({self.device}/{self.compute_type})",
                           min(100, int(segment.end / max(info.duration, 1) * 100)))
                return info.language
            except (RuntimeError, OSError) as exc:
                token.check()
                if self.device != "cuda":
                    raise
                self.unload()
                if emitted:
                    raise RuntimeError("Whisper CUDA 전사 도중 오류가 발생했습니다. GPU 메모리와 실행 환경을 확인한 뒤 재시도하세요.") from exc
                self.device, self.compute_type = "cpu", "int8"
                report("Whisper CUDA 초기 실행에 실패하여 CPU/int8로 전환", -1)

    def unload(self):
        self.model = None
        gc.collect()
