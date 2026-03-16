#!/usr/bin/env python3
"""
Download Piper TTS voice models for offline use.

Usage:
  python setup_models.py                          # downloads en_US-lessac-medium
  python setup_models.py en_US-amy-medium         # downloads a different voice
  python setup_models.py --list                   # show all available voices
"""

import argparse
import os
import sys
import urllib.request

MODELS_DIR = "models/piper"

# fmt: off
VOICES: dict[str, dict[str, str]] = {
    "en_US-lessac-medium": {
        "description": "US English, female, natural (recommended default)",
        "model":  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    },
    "en_US-amy-medium": {
        "description": "US English, female, clear",
        "model":  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    },
    "en_US-ryan-medium": {
        "description": "US English, male",
        "model":  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
    },
    "en_GB-alan-medium": {
        "description": "British English, male",
        "model":  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
    },
}
# fmt: on


def _progress_hook(count, block_size, total_size):
    if total_size <= 0:
        return
    done = count * block_size
    pct = min(100, int(done * 100 / total_size))
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    mb_done = done / 1_048_576
    mb_total = total_size / 1_048_576
    print(f"\r    [{bar}] {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return
    print(f"  Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
    print()  # newline after progress bar


def download_voice(voice_name: str) -> None:
    if voice_name not in VOICES:
        print(f"Unknown voice: {voice_name}")
        print(f"Available: {', '.join(VOICES)}")
        sys.exit(1)

    info = VOICES[voice_name]
    print(f"\nDownloading Piper voice: {voice_name}")
    print(f"Description: {info['description']}\n")

    model_dest = os.path.join(MODELS_DIR, f"{voice_name}.onnx")
    config_dest = os.path.join(MODELS_DIR, f"{voice_name}.onnx.json")

    _download(info["model"], model_dest)
    _download(info["config"], config_dest)

    print(f"\nVoice ready: {model_dest}")
    print("\nEnsure config.yaml has:")
    print(f'  audio:\n    tts:\n      model_path: "{model_dest}"')


def main() -> None:
    p = argparse.ArgumentParser(description="Download Piper TTS voice models")
    p.add_argument("voice", nargs="?", default="en_US-lessac-medium",
                   help="Voice name to download (default: en_US-lessac-medium)")
    p.add_argument("--list", action="store_true", help="List available voices")
    args = p.parse_args()

    if args.list:
        print("Available Piper voices:\n")
        for name, info in VOICES.items():
            marker = " (default)" if name == "en_US-lessac-medium" else ""
            print(f"  {name}{marker}")
            print(f"    {info['description']}")
        return

    download_voice(args.voice)


if __name__ == "__main__":
    main()
