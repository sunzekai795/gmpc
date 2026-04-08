from __future__ import annotations

from pathlib import Path
from typing import Optional


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_under(base: Path, maybe_relative: Optional[Path]) -> Optional[Path]:
    if maybe_relative is None:
        return None
    return maybe_relative if maybe_relative.is_absolute() else (base / maybe_relative).resolve()


def assert_file_exists(path: Path, hint: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{hint}: {path}")


def assert_dir_exists(path: Path, hint: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{hint}: {path}")
