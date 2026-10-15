"""Check installed packages and GPU kernels without downloading AI weights."""
import subprocess
import sys

from main import configure_runtime


def main():
    configure_runtime()
    import av
    import bitsandbytes as bnb
    import imageio_ffmpeg
    import torch
    import faster_whisper
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from huggingface_hub import get_token

    print("Python:", sys.version.split()[0])
    print("Environment:", sys.prefix)
    print("PyAV:", av.__version__)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float16, quant_type="nf4").to("cuda")
        with torch.inference_mode():
            output = layer(torch.randn(1, 64, device="cuda", dtype=torch.float16))
        print("NF4 GPU operation:", tuple(output.shape), "OK")
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-version"], capture_output=True, text=True)
    print("FFmpeg:", result.stdout.splitlines()[0])
    print("Hugging Face token configured:", bool(get_token()))
    print("No model weights downloaded by this check.")


if __name__ == "__main__":
    main()
