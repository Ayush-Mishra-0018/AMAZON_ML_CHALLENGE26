"""Small, dependency-free helpers for reproducible stage reports."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def write_json_atomic(path: str | Path, payload: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


@contextmanager
def stage_timer(report: dict[str, object]) -> Iterator[None]:
    started = time.monotonic()
    report["started_unix"] = time.time()
    try:
        yield
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
