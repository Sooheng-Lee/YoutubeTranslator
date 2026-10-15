import json
import sys
from types import SimpleNamespace

import pytest

from app.cancellation import CancellationToken
from app.models import AppSettings, MODEL_ID
from app.services.preflight import PreparationError, TranslationPreflight
from app.worker import PipelineWorker


def make_model(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 3840, "num_hidden_layers": 48}}))
    (path / "tokenizer_config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}))
    (path / "part-1.safetensors").write_bytes(b"fixture")
    (path / "part-2.safetensors").write_bytes(b"fixture")


def test_local_missing_shard_rejected(tmp_path):
    make_model(tmp_path)
    (tmp_path / "part-2.safetensors").unlink()
    with pytest.raises(PreparationError, match="가중치"):
        TranslationPreflight._local_files(tmp_path)


def test_complete_local_model_needs_no_hf_token(tmp_path, monkeypatch):
    make_model(tmp_path)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: pytest.fail("Local model must not request login"))
    TranslationPreflight()._check_model(AppSettings(model_path=str(tmp_path)), "", CancellationToken())


def test_complete_download_cache_works_offline(tmp_path, monkeypatch):
    make_model(tmp_path / "models" / "translategemma-12b-it")
    monkeypatch.setattr("app.services.preflight.data_directory", lambda: tmp_path)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: pytest.fail("Complete cache must work offline"))
    TranslationPreflight()._check_model(AppSettings(), "", CancellationToken())


def test_missing_token_fails_before_download(tmp_path, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr("app.services.preflight.data_directory", lambda: tmp_path)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    with pytest.raises(PreparationError, match="Access 토큰"):
        TranslationPreflight()._check_model(AppSettings(), "", CancellationToken())


def test_entered_token_used_for_weight_access(tmp_path, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr("app.services.preflight.data_directory", lambda: tmp_path)
    calls = []

    def info(self, repo_id, **kwargs):
        calls.append((repo_id, kwargs["token"]))
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename="model.safetensors", size=100)])

    def metadata(url, **kwargs):
        calls.append((url, kwargs["token"]))

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", info)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", metadata)
    monkeypatch.setattr("app.services.preflight.shutil.disk_usage", lambda path: SimpleNamespace(free=10 * 1024**3))
    TranslationPreflight()._check_model(AppSettings(), "hf_fixture", CancellationToken())
    assert len(calls) == 2 and all(token == "hf_fixture" for _, token in calls)


def test_rejected_access_shows_actionable_message(tmp_path, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr("app.services.preflight.data_directory", lambda: tmp_path)

    def denied(*args, **kwargs):
        exc = RuntimeError("denied")
        exc.response = SimpleNamespace(status_code=403)
        raise exc

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", denied)
    with pytest.raises(PreparationError, match="이용 조건 동의"):
        TranslationPreflight()._check_model(AppSettings(), "hf_fixture", CancellationToken())


def test_worker_blocks_pipeline_and_redacts_token(monkeypatch):
    def fail(*args):
        raise PreparationError("invalid hf_fixture_secret")

    monkeypatch.setattr(TranslationPreflight, "check", fail)
    monkeypatch.setattr("app.worker.TranslationPipeline", lambda **kwargs: pytest.fail("Pipeline must not start"))
    worker = PipelineWorker("translate", AppSettings(), access_token="hf_fixture_secret")
    errors, finished = [], []
    worker.error.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))
    worker.run()
    assert "준비가 완료되지" in errors[0]
    assert "hf_fixture_secret" not in errors[0]
    assert worker.access_token is None and finished == [True]


def test_video_download_skips_llm_preflight(monkeypatch):
    monkeypatch.setattr(TranslationPreflight, "check", lambda *args: pytest.fail("Video needs no model"))

    class Downloader:
        def inspect(self, *args):
            return SimpleNamespace(id="fixture")

        def download_video(self, *args):
            return "fixture.mp4"

    monkeypatch.setattr("app.worker.YouTubeDownloader", Downloader)
    worker = PipelineWorker("video", AppSettings())
    results = []
    worker.result.connect(results.append)
    worker.run()
    assert results == ["fixture.mp4"]


def test_invalid_cpu_precision_rejected():
    with pytest.raises(PreparationError, match="float32"):
        TranslationPreflight().check(AppSettings(device="cpu", precision="4bit"), "", CancellationToken(), lambda *args: None)


@pytest.mark.parametrize("available,free,message", [(False, 16, "CUDA GPU"), (True, 2, "여유 메모리")])
def test_unavailable_gpu_or_insufficient_memory(monkeypatch, available, free, message):
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: available, mem_get_info=lambda: (free * 1024**3, 12 * 1024**3))))
    with pytest.raises(PreparationError, match=message):
        TranslationPreflight._check_runtime(AppSettings())
