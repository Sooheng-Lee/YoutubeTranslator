"""Install the official portable Deno runtime used by yt-dlp's EJS solver."""
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile


def read_url(url):
    request = Request(url, headers={"User-Agent": "YouTubeTranslator-setup"})
    with urlopen(request, timeout=90) as response:
        return response.read()


def main():
    root = Path(__file__).resolve().parent / ".tools"
    if (root / "deno.exe").exists():
        print("Portable Deno already installed.")
        return
    release = json.loads(read_url("https://api.github.com/repos/denoland/deno/releases/latest"))
    name = "deno-x86_64-pc-windows-msvc.zip"
    asset = next(a for a in release["assets"] if a["name"] == name)
    archive = read_url(asset["browser_download_url"])
    digest = hashlib.sha256(archive).hexdigest()
    expected = asset.get("digest", "").removeprefix("sha256:")
    if not expected:
        checksum_asset = next(a for a in release["assets"] if a["name"] == name + ".sha256sum")
        expected = read_url(checksum_asset["browser_download_url"]).decode().split()[0]
    if digest != expected:
        raise RuntimeError("Deno download checksum mismatch.")
    root.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        executable = zipped.read("deno.exe")
    temp = root / "deno.exe.tmp"
    temp.write_bytes(executable)
    temp.replace(root / "deno.exe")
    (root / "version.json").write_text(json.dumps({"deno": release["tag_name"], "sha256": digest}, indent=2))
    print(f"Installed Deno {release['tag_name']} (SHA-256 verified).")


if __name__ == "__main__":
    main()
