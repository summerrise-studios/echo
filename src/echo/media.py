"""Audio and video in and out, through ffmpeg."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from . import EchoError


def _tool(name: str) -> str:
    env = os.environ.get(f"ECHO_{name.upper()}")
    if env:
        return env
    found = shutil.which(name)
    if found:
        return found
    if name == "ffprobe":  # often installed next to ffmpeg
        ff = _tool("ffmpeg")
        sibling = Path(ff).with_name("ffprobe" + Path(ff).suffix)
        if sibling.exists():
            return str(sibling)
    raise EchoError(f"{name} not found: install FFmpeg (https://ffmpeg.org) or set ECHO_{name.upper()}")


def ffmpeg(*args: str) -> None:
    cmd = [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise EchoError(f"ffmpeg failed: {p.stderr.strip()[-500:]}")


def probe(path: str | Path) -> dict:
    """Duration in seconds and whether the file has a video stream."""
    cmd = [_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration:stream=codec_type",
           "-of", "json", str(path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise EchoError(f"could not read {path}: {p.stderr.strip()[-300:]}")
    data = json.loads(p.stdout or "{}")
    kinds = {s.get("codec_type") for s in data.get("streams", [])}
    if "audio" not in kinds:
        raise EchoError(f"{path} has no audio track")
    return {"duration": float(data.get("format", {}).get("duration", 0.0)), "video": "video" in kinds}


def extract_audio(src: str | Path, dst: str | Path, sr: int = 44100, channels: int = 2) -> Path:
    ffmpeg("-i", src, "-vn", "-ac", channels, "-ar", sr, "-c:a", "pcm_f32le", dst)
    return Path(dst)


def atempo_chain(factor: float) -> str:
    """ffmpeg's atempo takes 0.5 to 2.0 per filter; chain filters for anything outside."""
    if factor <= 0:
        raise ValueError("tempo factor must be positive")
    parts = []
    while factor > 2.0:
        parts.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        parts.append(0.5)
        factor /= 0.5
    parts.append(factor)
    return ",".join(f"atempo={p:.4f}" for p in parts)


def change_tempo(src: str | Path, dst: str | Path, factor: float) -> Path:
    """Speed speech up (factor > 1) or slow it down without changing its pitch."""
    ffmpeg("-i", src, "-filter:a", atempo_chain(factor), "-c:a", "pcm_f32le", dst)
    return Path(dst)


def mux(video: str | Path, audio: str | Path, dst: str | Path) -> Path:
    """Copy the picture, replace the sound."""
    ffmpeg("-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
           "-c:a", "aac", "-b:a", "192k", "-shortest", dst)
    return Path(dst)


def encode(audio: str | Path, dst: str | Path) -> Path:
    ffmpeg("-i", audio, dst)
    return Path(dst)
