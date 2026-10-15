import json
from types import SimpleNamespace

import pytest

from app.cancellation import CancellationToken, CancelledError
from app.models import AppSettings, MODEL_ID
from app.services.llm_model_downloader import LLMModelDownloader
from app.worker import PipelineWorker


def mock_hub(monkeypatch, tmp_path, fail_after=None):
    import huggingface_hub
    payloads = {"config.json": json.dumps({"text_config": {"hidden_size": 3840, "num_hidden_layers": 48}}).encode(),
                "tokenizer_config.json": b"{}", "tokenizer.json": b"{}", "model.safetensors": b"fixture"}
    siblings = [SimpleNamespace(rfilename=k, size=len(v)) for k, v in payloads.items()]
    monkeypatch.setattr("app.services.llm_model_downloader.data_directory", lambda: tmp_path)
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda *args, **kwargs: SimpleNamespace(siblings=siblings, sha="fixed-revision"))
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    monkeypatch.setattr("app.services.llm_model_downloader.shutil.disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    calls = []

    def download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        path = tmp_path / "models" / "translategemma-12b-it" / filename
        path.write_bytes(payloads[filename])
        if fail_after:
            fail_after()
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls


def test_download_uses_entered_token_pinned_revision_and_reuses_completed_model(tmp_path, monkeypatch):
    calls = mock_hub(monkeypatch, tmp_path)
    progress = []
    downloader = LLMModelDownloader()
    path = downloader.download(MODEL_ID, "hf_fixture", CancellationToken(), lambda *args: progress.append(args))
    assert (path / "model.safetensors").exists()
    assert len(calls) == 4
    assert all(c[2]["token"] == "hf_fixture" and c[2]["revision"] == "fixed-revision" for c in calls)
    assert progress[-1][1] == 100
    downloader.download(MODEL_ID, "", CancellationToken(), lambda *args: None)
    assert len(calls) == 4
    assert not any("hf_fixture" in p.read_text(errors="ignore") for p in path.iterdir())


def test_cancel_between_files_keeps_completed_file(tmp_path, monkeypatch):
    token = CancellationToken()
    calls = mock_hub(monkeypatch, tmp_path, fail_after=token.cancel)
    with pytest.raises(CancelledError):
        LLMModelDownloader().download(MODEL_ID, "hf_fixture", token, lambda *args: None)
    assert len(calls) == 1
    assert (tmp_path / "models" / "translategemma-12b-it" / "config.json").exists()


def test_model_worker_needs_neither_video_nor_preflight(monkeypatch):
    monkeypatch.setattr("app.worker.YouTubeDownloader", lambda: pytest.fail("Model download needs no FFmpeg/video"))
    monkeypatch.setattr("app.worker.TranslationPreflight.check", lambda *args: pytest.fail("Download needs no GPU"))
    seen = []
    monkeypatch.setattr(LLMModelDownloader, "download", lambda self, model, access, *args: seen.append((model, access)) or "model-cache")
    worker = PipelineWorker("model", AppSettings(url="", device="cpu"), access_token="hf_fixture")
    results = []
    worker.result.connect(results.append)
    worker.run()
    assert seen == [(MODEL_ID, "hf_fixture")]
    assert results[0].kind == "llm" and results[0].path == "model-cache"
    assert worker.access_token is None
