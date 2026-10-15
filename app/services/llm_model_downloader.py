"""Download model files without loading the model or requiring a GPU."""
from fnmatch import fnmatch
from pathlib import Path
import shutil

from app.models import MODEL_ID
from app.repositories.settings_repository import data_directory
from app.services.preflight import PreparationError, TranslationPreflight


class LLMModelDownloader:
    PATTERNS = ("*.json", "*.safetensors", "*.model", "*.jinja", "*.txt")

    def download(self, model_id, access_token, cancellation, report):
        from huggingface_hub import HfApi, get_token, hf_hub_download

        cancellation.check()
        model_id = model_id.strip()
        if not model_id:
            raise PreparationError("LLM ID Path에 다운로드할 모델 ID를 입력하세요.")
        if model_id != MODEL_ID:
            local = Path(model_id).expanduser()
            if local.is_dir():
                TranslationPreflight._local_files(local)
                report("로컬 LLM 모델 파일 준비 완료", 100)
                return local.resolve()
            raise PreparationError("현재 다운로드 지원 모델은 google/translategemma-12b-it입니다. LLM ID Path를 확인하세요.")
        destination = data_directory() / "models" / "translategemma-12b-it"
        try:
            TranslationPreflight._local_files(destination)
            report("이미 다운로드된 LLM 모델 확인 완료", 100)
            return destination
        except PreparationError:
            pass
        credential = access_token or get_token()
        if not credential:
            raise PreparationError("Access 토큰을 입력하거나 Hugging Face에 로그인하세요. TranslateGemma 모델 이용 조건 동의도 필요합니다.")
        report("LLM 다운로드 권한 및 파일 목록 확인 중", -1)
        info = HfApi().model_info(model_id, token=credential, files_metadata=True, timeout=15)
        cancellation.check()
        files = [f for f in info.siblings if any(fnmatch(f.rfilename, p) for p in self.PATTERNS)]
        if not any(f.rfilename.endswith(".safetensors") for f in files):
            raise PreparationError("다운로드할 모델 가중치를 찾을 수 없습니다.")
        # Complete each file before moving to the next; allow cancellation between files.
        files.sort(key=lambda f: (f.rfilename.endswith(".safetensors"), f.rfilename))
        required = 0
        for file in files:
            path = (destination / file.rfilename).resolve()
            if not path.is_relative_to(destination.resolve()):
                raise PreparationError("모델 파일 경로가 올바르지 않습니다.")
            if not path.is_file() or path.stat().st_size != file.size:
                required += file.size or 0
        if shutil.disk_usage(data_directory()).free < required + 2 * 1024**3:
            raise PreparationError(f"모델 저장 공간이 부족합니다. 약 {required / 1024**3 + 2:.1f}GB 이상의 여유 공간이 필요합니다.")
        destination.mkdir(parents=True, exist_ok=True)
        for index, file in enumerate(files):
            cancellation.check()
            report(f"LLM 파일 {index + 1}/{len(files)} 다운로드: {file.rfilename} (진행률은 완료 파일 수 기준)",
                   int(index / len(files) * 100))
            hf_hub_download(model_id, file.rfilename, revision=info.sha, token=credential,
                            local_dir=str(destination))
            cancellation.check()
            downloaded = destination / file.rfilename
            if not downloaded.is_file() or downloaded.stat().st_size == 0 or (
                    file.size is not None and downloaded.stat().st_size != file.size):
                raise PreparationError("다운로드한 모델 파일이 불완전합니다. 다운로드를 다시 실행하세요.")
            report(f"LLM 파일 {index + 1}/{len(files)} 완료", int((index + 1) / len(files) * 100))
        TranslationPreflight._local_files(destination)
        report("LLM 모델 다운로드 완료", 100)
        return destination
