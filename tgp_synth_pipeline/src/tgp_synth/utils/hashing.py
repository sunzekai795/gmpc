from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def sha1_text(text: str) -> str:
    return sha1_bytes(text.encode("utf-8"))


def sha1_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha1_json(obj: Dict[str, Any]) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha1_bytes(blob)
