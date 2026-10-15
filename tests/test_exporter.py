import pytest
import pysubs2

from app.models import SubtitleSegment
from app.services.subtitle_exporter import SubtitleExporter


@pytest.mark.parametrize("ext", ["srt", "vtt"])
def test_export_preserves_timestamps_and_unicode(tmp_path, ext):
    segment = SubtitleSegment(1, 1.234, 3.456, "Hello", "안녕하세요\n반갑습니다", "완료")
    path = SubtitleExporter().export([segment], tmp_path / f"sub.{ext}", translated=True)
    parsed = pysubs2.load(str(path), encoding="utf-8")
    assert parsed[0].start == 1234
    assert parsed[0].end == 3456
    assert parsed[0].plaintext == "안녕하세요\n반갑습니다"
    if ext == "vtt":
        assert path.read_text(encoding="utf-8").startswith("WEBVTT")


def test_failed_translation_does_not_block_original(tmp_path):
    segments = [SubtitleSegment(1, 0, 1, "original", status="실패"),
                SubtitleSegment(2, 1, 2, "next", "다음", "완료")]
    exporter = SubtitleExporter()
    exporter.export(segments, tmp_path / "original.srt")
    assert len(pysubs2.load(str(tmp_path / "original.srt"))) == 2
    with pytest.raises(ValueError, match="미완료"):
        exporter.export(segments, tmp_path / "translated.srt", translated=True)
    exporter.export(segments, tmp_path / "partial.srt", translated=True, partial=True)
    parsed = pysubs2.load(str(tmp_path / "partial.srt"))
    assert len(parsed) == 1 and parsed[0].start == 1000


def test_invalid_interval_rejected(tmp_path):
    with pytest.raises(ValueError, match="시간"):
        SubtitleExporter().export([SubtitleSegment(1, 2, 1, "bad")], tmp_path / "bad.srt")
