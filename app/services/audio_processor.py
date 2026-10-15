import os
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg


class AudioProcessor:
    def __init__(self):
        self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    def normalize(self, source, destination, token):
        destination = Path(destination)
        command = [self.ffmpeg, "-y", "-nostdin", "-i", str(source), "-vn",
                   "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors,
                                       creationflags=flags)
            try:
                while True:
                    token.check()
                    try:
                        code = process.wait(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                if code:
                    errors.seek(0)
                    raise RuntimeError("오디오 변환 실패: " + errors.read().decode(errors="replace")[-1200:])
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        return destination
