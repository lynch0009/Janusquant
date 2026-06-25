"""Output directory allocation and transactional publication."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
from uuid import uuid4


def resolve_output_dir(target: Path) -> Path:
    target = Path(target)
    if not target.exists():
        return target
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    candidate = target.with_name(f"{target.name}_{stamp}")
    if candidate.exists():
        candidate = target.with_name(f"{target.name}_{stamp}_{uuid4().hex[:8]}")
    return candidate


class StagedOutput:
    def __init__(self, target: Path):
        self.final = resolve_output_dir(Path(target))
        self.staging = self.final.with_name(f"{self.final.name}.__staging__{uuid4().hex[:8]}")

    def __enter__(self) -> Path:
        self.final.parent.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=False)
        return self.staging

    @property
    def output_dir(self) -> Path:
        return self.final

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None:
            shutil.rmtree(self.staging, ignore_errors=True)
            return False
        if self.final.exists():
            self.final = resolve_output_dir(self.final)
        self.staging.replace(self.final)
        return False
