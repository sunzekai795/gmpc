from __future__ import annotations

from pathlib import Path
from typing import List

from .subprocess_utils import ensure_executable, run_command


class FFmpegError(RuntimeError):
    pass


def _ensure_ffmpeg() -> None:
    ensure_executable("ffmpeg")
    ensure_executable("ffprobe")


def cut_video_segment(in_video: Path, out_video: Path, start_s: float, duration_s: float) -> None:
    _ensure_ffmpeg()
    out_video.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_s),
            "-t",
            str(duration_s),
            "-i",
            str(in_video),
            "-c",
            "copy",
            str(out_video),
        ]
    )


def extract_keyframes(in_video: Path, out_dir: Path, fps: float) -> None:
    _ensure_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%06d.jpg")
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(in_video),
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            pattern,
        ]
    )


def resample_audio(in_wav: Path, out_wav: Path, sample_rate: int) -> None:
    _ensure_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(in_wav),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(out_wav),
        ]
    )


def crop_square_and_scale(in_video: Path, out_video: Path, resolution: int, fps: int = 25) -> None:
    _ensure_ffmpeg()
    out_video.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={resolution}:{resolution}:force_original_aspect_ratio=increase,crop={resolution}:{resolution},fps={fps}"
    run_command(["ffmpeg", "-y", "-i", str(in_video), "-vf", vf, "-an", str(out_video)])


def concat_videos(in_list: List[Path], out_video: Path) -> None:
    _ensure_ffmpeg()
    if not in_list:
        raise ValueError("empty input list")

    out_video.parent.mkdir(parents=True, exist_ok=True)
    concat_txt = out_video.parent / (out_video.stem + "_concat.txt")
    lines = [f"file '{p.resolve()}'" for p in in_list]
    concat_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    run_command(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-c",
            "copy",
            str(out_video),
        ]
    )


def make_synthetic_reference_video(out_video: Path, resolution: int = 512, fps: int = 25, seconds: int = 10) -> None:
    """Create a simple reference video for local validation."""

    _ensure_ffmpeg()
    out_video.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={resolution}x{resolution}:r={fps}:d={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_video),
        ]
    )
