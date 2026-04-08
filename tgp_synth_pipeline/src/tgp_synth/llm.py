from __future__ import annotations

import importlib
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple


class LLMError(RuntimeError):
    pass


def _split_callable_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"LLM callable spec must be '<module-or-file>:<func>', got: {spec}")
    left, func_name = spec.split(":", 1)
    left = left.strip()
    func_name = func_name.strip()
    if not left or not func_name:
        raise ValueError(f"Invalid callable spec: {spec}")
    return left, func_name


def load_llm_callable(spec: str) -> Callable[[str], str]:
    """Load a user-provided LLM function.

    Supported forms:
    - "some.module:generate"
    - "/abs/path/to/file.py:generate"
    - "relative/path/to/file.py:generate" (resolved relative to CWD)

    The loaded function must accept `prompt: str` and return `str`.
    """

    target, func_name = _split_callable_spec(spec)

    if target.endswith(".py") or "/" in target or "\\" in target:
        file_path = Path(target).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"LLM file not found: {file_path}")

        module_name = f"tgp_user_llm_{file_path.stem}_{abs(hash(str(file_path)))}"
        module_spec = importlib.util.spec_from_file_location(module_name, file_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Failed to import module from file: {file_path}")

        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)  # type: ignore[attr-defined]
    else:
        module = importlib.import_module(target)

    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"LLM callable not found or not callable: {spec}")

    return func


@dataclass
class LLMCallMetrics:
    duration_s: float


@dataclass
class LLMClient:
    """A small reliability wrapper over a user-owned callable."""

    generate: Callable[[str], str]
    max_retries: int = 2
    timeout_s: Optional[float] = None

    def complete(self, prompt: str) -> str:
        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                resp = self.generate(prompt)
                duration = time.time() - start

                if self.timeout_s is not None and duration > self.timeout_s:
                    raise TimeoutError(f"LLM call exceeded timeout_s={self.timeout_s}")
                if not isinstance(resp, str):
                    raise TypeError(f"LLM callable must return str, got {type(resp)}")
                if not resp.strip():
                    raise ValueError("LLM returned empty response")

                return resp
            except BaseException as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 8))

        raise LLMError(str(last_err))
