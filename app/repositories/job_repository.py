import json
from pathlib import Path

from app.models import TranslationJob
from app.repositories.settings_repository import data_directory, write_json


class JobRepository:
    def __init__(self, root=None):
        self.root = Path(root or data_directory() / "jobs")

    def save(self, job):
        write_json(self.root / f"{job.id}.json", job.to_dict())

    def load(self, job_id):
        path = self.root / f"{job_id}.json"
        try:
            return TranslationJob.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def workspace(self, job_id):
        path = self.root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path
