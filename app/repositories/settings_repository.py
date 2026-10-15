import json
from dataclasses import asdict, fields
from pathlib import Path

from platformdirs import user_data_dir

from app.models import AppSettings


def data_directory():
    path = Path(user_data_dir("YouTubeTranslator", appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class SettingsRepository:
    def __init__(self, root=None):
        self.path = Path(root or data_directory()) / "settings.json"

    def load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            allowed = {f.name for f in fields(AppSettings)}
            return AppSettings(**{k: v for k, v in data.items() if k in allowed})
        except (OSError, ValueError, TypeError):
            return AppSettings()

    def save(self, settings):
        write_json(self.path, asdict(settings))
