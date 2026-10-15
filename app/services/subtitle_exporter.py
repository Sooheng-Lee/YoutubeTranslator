from pathlib import Path

import pysubs2


class SubtitleExporter:
    def export(self, segments, destination, translated=False, partial=False):
        destination = Path(destination)
        if destination.suffix.lower() not in {".srt", ".vtt", ".txt"}:
            raise ValueError("SRT, VTT, TXT 형식을 선택하세요.")
        if not segments:
            raise ValueError("저장할 자막이 없습니다.")
        if translated and not partial and any(s.status != "완료" for s in segments):
            raise ValueError("미완료 번역이 있습니다. 부분 저장을 선택하세요.")
        subs = pysubs2.SSAFile()
        lines = []
        for segment in segments:
            if segment.start < 0 or segment.end <= segment.start:
                raise ValueError(f"잘못된 자막 시간: 구간 {segment.id}")
            if translated and segment.status != "완료":
                continue
            text = segment.translated if translated else segment.original
            # SSA uses braces for formatting; escape literal braces for safe plain text output.
            text = text.replace("{", "｛").replace("}", "｝")
            subs.append(pysubs2.SSAEvent(start=round(segment.start * 1000),
                                        end=round(segment.end * 1000),
                                        text=text.replace("\n", r"\N")))
            lines.append(f"[{segment.start:.3f} - {segment.end:.3f}]\n{segment.original}\n{segment.translated}\n")
        if not subs:
            raise ValueError("저장할 완료 구간이 없습니다.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + ".tmp")
        if destination.suffix.lower() == ".txt":
            temp.write_text("\n".join(lines), encoding="utf-8")
        else:
            subs.save(str(temp), format_=destination.suffix[1:].lower(), encoding="utf-8")
        temp.replace(destination)
        return destination

    def convert(self, source, destination):
        subs = pysubs2.load(str(source), encoding="utf-8")
        destination = Path(destination)
        temp = destination.with_name(destination.name + ".tmp")
        subs.save(str(temp), format_=destination.suffix[1:].lower(), encoding="utf-8")
        temp.replace(destination)
        return destination
