import json
import os
import shutil
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import tempfile
from threading import Thread
from urllib.parse import urlparse

from app.models import VideoMetadata
from app.services.audio_processor import AudioProcessor


class YouTubeDownloader:
    def __init__(self):
        self.ffmpeg = AudioProcessor().ffmpeg

    @staticmethod
    def validate_url(url):
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}:
            raise ValueError("YouTube 영상의 https URL을 입력하세요.")

    def _run(self, url, mode, options, token, report):
        self.validate_url(url)
        token.check()
        options = {"noplaylist": True, "quiet": True, "no_warnings": True,
                   "socket_timeout": 20, "retries": 3, "ffmpeg_location": self.ffmpeg,
                   "match_filter": None, **options}
        deno = Path(__file__).resolve().parents[2] / ".tools" / "deno.exe"
        if deno.exists():
            options["js_runtimes"] = {"deno": {"path": str(deno)}}
        elif shutil.which("node"):
            options["js_runtimes"] = {"node": {"path": shutil.which("node")}}
        elif not shutil.which("deno"):
            raise RuntimeError("YouTube 추출용 JavaScript 런타임이 없습니다. setup.bat을 실행하세요.")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with tempfile.TemporaryDirectory(prefix="yttranslator-") as directory:
            config = Path(directory) / "request.json"
            config.write_text(json.dumps({"url": url, "mode": mode, "options": options}), encoding="utf-8")
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            process = subprocess.Popen([sys.executable, "-m", "app.download_runner", str(config)],
                                       cwd=str(Path(__file__).resolve().parents[2]), env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", creationflags=flags)
            queue = Queue()

            def read_output():
                for line in process.stdout:
                    queue.put(line)
                queue.put(None)

            reader = Thread(target=read_output, daemon=True)
            reader.start()
            result, error = None, "다운로드가 실패했습니다. 네트워크와 영상 접근 가능 여부를 확인하세요."
            try:
                while True:
                    token.check()
                    try:
                        line = queue.get(timeout=0.15)
                    except Empty:
                        continue
                    if line is None:
                        break
                    if not line.startswith("YTAPP:"):
                        continue
                    event = json.loads(line[6:])
                    if event["kind"] == "progress":
                        report(*event["value"])
                    elif event["kind"] == "result":
                        result = event["value"]
                    elif event["kind"] == "error":
                        error = event["value"]
                if process.wait() != 0 or result is None:
                    raise RuntimeError(error)
                token.check()
                return result
            finally:
                if process.poll() is None:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       capture_output=True, creationflags=flags)
                    else:
                        process.terminate()
                    process.wait()
                reader.join(timeout=3)
                process.stdout.close()

    def inspect(self, url, token, report):
        report("영상 정보 및 자막 목록 조회", -1)
        return VideoMetadata(**self._run(url, "inspect", {"skip_download": True}, token, report))

    def download_audio(self, url, directory, token, report):
        directory = Path(directory)
        result = self._run(url, "audio", {"format": "bestaudio/best",
                           "outtmpl": str(directory / "audio.%(ext)s")}, token, report)
        candidates = [p for p in directory.glob("audio.*") if p.suffix not in {".part", ".ytdl", ".wav"}]
        path = Path(result["filepath"])
        if not path.is_file():
            path = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
        if path is None:
            raise RuntimeError("다운로드한 오디오 파일을 찾을 수 없습니다.")
        return path

    def download_video(self, url, video_id, output, quality, token, report):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        cap = "" if quality == "best" else f"[height<={int(quality)}]"
        self._run(url, "video", {
            "format": f"bv{cap}[ext=mp4]+ba[ext=m4a]/b{cap}[ext=mp4]/bv{cap}+ba/b{cap}",
            "outtmpl": str(output / f"{video_id}.%(ext)s"),
            "merge_output_format": "mp4/mkv", "overwrites": False,
        }, token, report)
        candidates = [p for p in output.glob(f"{video_id}.*") if p.suffix in {".mp4", ".mkv", ".webm"} and ".f" not in p.stem]
        if not candidates:
            raise RuntimeError("최종 영상 파일을 찾을 수 없습니다.")
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def download_subtitle(self, url, language, source, directory, token, report):
        directory = Path(directory)
        self._run(url, "subtitle", {
            "skip_download": True, "writesubtitles": source == "manual",
            "writeautomaticsub": source == "auto", "subtitleslangs": [language],
            "subtitlesformat": "vtt/srt", "outtmpl": str(directory / "subtitle.%(ext)s"),
        }, token, report)
        candidates = list(directory.glob("subtitle.*.vtt")) + list(directory.glob("subtitle.*.srt"))
        if not candidates:
            raise RuntimeError("선택한 기존 자막을 다운로드할 수 없습니다.")
        return max(candidates, key=lambda p: p.stat().st_mtime)
