"""Isolated yt-dlp process so cancellation can also terminate its FFmpeg children."""
import json
from pathlib import Path
import sys
import time

from yt_dlp import YoutubeDL


def emit(kind, value):
    print("YTAPP:" + json.dumps({"kind": kind, "value": value}, ensure_ascii=True), flush=True)


class QuietLogger:
    def debug(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


def run(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    opts = config["options"]
    opts["logger"] = QuietLogger()
    last_progress = [0.0]

    def progress(data):
        now = time.monotonic()
        if now - last_progress[0] < 0.2 and data.get("status") != "finished":
            return
        last_progress[0] = now
        total = data.get("total_bytes") or data.get("total_bytes_estimate")
        percent = int(data.get("downloaded_bytes", 0) / total * 100) if total else -1
        emit("progress", ["다운로드", min(percent, 100)])

    opts["progress_hooks"] = [progress]
    opts["postprocessor_hooks"] = [lambda data: emit("progress", ["미디어 병합/후처리", -1])]
    with YoutubeDL(opts) as downloader:
        info = downloader.extract_info(config["url"], download=config["mode"] != "inspect")
    if not info or info.get("_type") == "playlist" or info.get("is_live"):
        raise ValueError("단일 녹화 영상만 지원합니다.")
    if config["mode"] == "inspect":
        subtitles = []
        for key, source in (("subtitles", "manual"), ("automatic_captions", "auto")):
            for language, formats in info.get(key, {}).items():
                if any(f.get("ext") in {"vtt", "srt"} for f in formats):
                    subtitles.append({"language": language, "source": source})
        value = {"id": info["id"], "url": info.get("webpage_url", config["url"]),
                 "title": info.get("title", info["id"]), "duration": info.get("duration") or 0,
                 "heights": sorted({int(f["height"]) for f in info.get("formats", []) if f.get("height")}, reverse=True),
                 "subtitles": subtitles}
    else:
        value = {"id": info["id"], "filepath": info.get("filepath") or info.get("_filename", "")}
    emit("result", value)


if __name__ == "__main__":
    try:
        run(sys.argv[1])
    except Exception as exc:
        emit("error", str(exc))
        sys.exit(1)
