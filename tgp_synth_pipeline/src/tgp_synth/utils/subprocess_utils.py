from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union


class CommandError(RuntimeError):
    """Raised when an external command fails."""

    def __init__(self, cmd: Sequence[str], returncode: int, output: str):
        super().__init__(f"Command failed ({returncode}): {' '.join(map(str, cmd))}\n{output}")
        self.cmd = list(cmd)
        self.returncode = returncode
        self.output = output


@dataclass
class CommandResult:
    cmd: List[str]
    returncode: int
    output: str
    duration_s: float


def run_command(
    cmd: Sequence[Union[str, os.PathLike[str]]],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> CommandResult:
    """Run a command and capture combined stdout/stderr.

    This helper is intentionally conservative:
    - captures stdout+stderr for later inspection
    - provides a structured result including timing
    """

    cmd_str = [str(x) for x in cmd]
    start = time.time()
    proc = subprocess.run(
        cmd_str,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    duration_s = time.time() - start

    result = CommandResult(cmd=cmd_str, returncode=proc.returncode, output=proc.stdout, duration_s=duration_s)
    if check and proc.returncode != 0:
        raise CommandError(cmd_str, proc.returncode, proc.stdout)
    return result


def which(executable: str) -> Optional[str]:
    """A small wrapper around `shutil.which` (without importing shutil at module import time)."""

    import shutil

    return shutil.which(executable)


def ensure_executable(executable: str) -> str:
    """Ensure an executable is available in PATH."""

    found = which(executable)
    if not found:
        raise FileNotFoundError(f"Required executable not found in PATH: {executable}")
    return found
