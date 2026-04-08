from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import LatentSyncConfig
from ..utils.subprocess_utils import CommandError, ensure_executable, run_command
from .base import VideoBackend, VideoSynthesisRequest, VideoSynthesisResult


class LatentSyncSubprocessBackend:
    name = "latentsync_subprocess"

    def __init__(self, cfg: LatentSyncConfig):
        self._cfg = cfg

    def synthesize(self, request: VideoSynthesisRequest, out_mp4: Path) -> VideoSynthesisResult:
        if self._cfg.repo_dir is None:
            raise ValueError("latentsync.repo_dir is required")

        python_exe = self._cfg.python or Path("python")
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        request.reference_video.exists() or (_ for _ in ()).throw(FileNotFoundError(request.reference_video))
        request.audio_wav.exists() or (_ for _ in ()).throw(FileNotFoundError(request.audio_wav))

        cmd = [
            str(python_exe),
            "-m",
            "scripts.inference",
            "--unet_config_path",
            self._cfg.unet_config_path,
            "--inference_ckpt_path",
            self._cfg.inference_ckpt_path,
            "--inference_steps",
            str(self._cfg.inference_steps),
            "--guidance_scale",
            str(self._cfg.guidance_scale),
            "--video_path",
            str(request.reference_video),
            "--audio_path",
            str(request.audio_wav),
            "--video_out_path",
            str(out_mp4),
            "--temp_dir",
            str(self._cfg.temp_root / request.session_id / f"{request.turn_id:04d}"),
        ]
        if self._cfg.enable_deepcache:
            cmd.append("--enable_deepcache")

        try:
            run_command(cmd, cwd=self._cfg.repo_dir)
        except CommandError as e:
            raise RuntimeError(f"LatentSync failed for {request.session_id}/{request.turn_id}") from e

        return VideoSynthesisResult(session_id=request.session_id, turn_id=request.turn_id, video_path=out_mp4)


class DummyVideoBackend:
    """A lightweight backend that muxes input audio onto a looped reference video."""

    name = "dummy_video"

    def __init__(self, fps: int = 25):
        self._fps = fps
        ensure_executable("ffmpeg")

    def synthesize(self, request: VideoSynthesisRequest, out_mp4: Path) -> VideoSynthesisResult:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        # -stream_loop -1 makes the video infinite; -shortest cuts to audio duration.
        cmd = [
            "ffmpeg",
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            str(request.reference_video),
            "-i",
            str(request.audio_wav),
            "-shortest",
            "-r",
            str(self._fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
        run_command(cmd)
        return VideoSynthesisResult(session_id=request.session_id, turn_id=request.turn_id, video_path=out_mp4)


def build_video_backend(cfg: LatentSyncConfig) -> VideoBackend:
    if cfg.backend == "dummy":
        return DummyVideoBackend()
    if cfg.backend == "subprocess":
        return LatentSyncSubprocessBackend(cfg)
    raise NotImplementedError(f"Unsupported video backend: {cfg.backend}")
