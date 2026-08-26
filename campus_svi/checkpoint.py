"""Resumable per-cell progress logging.

Every grid cell is an atomic query unit. Its outcome is appended to a JSONL
file as soon as it resolves, so an interrupted run — a dropped connection, a
rate-limit wall, a closed Colab tab — never costs more than the cell in
flight. Re-running a fetch script skips cells already marked ``done``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class Checkpoint:
    """Append-only JSONL log of per-cell fetch outcomes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        self._failed: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn final line from a hard kill
                gid, status = rec.get("grid_id"), rec.get("status")
                if gid is None:
                    continue
                if status == "done":
                    self._done.add(gid)
                    self._failed.discard(gid)
                elif status == "failed":
                    self._failed.add(gid)

    # -- queries -----------------------------------------------------------

    def is_done(self, grid_id: str) -> bool:
        return grid_id in self._done

    @property
    def done(self) -> set[str]:
        return set(self._done)

    @property
    def failed(self) -> set[str]:
        """Cells that failed and were never subsequently completed."""
        return self._failed - self._done

    def pending(self, grid_ids: Iterable[str], retry_failed: bool = True) -> list[str]:
        """Grid ids still needing a fetch, in input order."""
        out = []
        for gid in grid_ids:
            if gid in self._done:
                continue
            if not retry_failed and gid in self._failed:
                continue
            out.append(gid)
        return out

    # -- writes ------------------------------------------------------------

    def _write(self, grid_id: str, status: str, **extra) -> None:
        rec = {"grid_id": grid_id, "status": status, **extra}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    def mark_done(self, grid_id: str, n_records: int = 0, **extra) -> None:
        self._done.add(grid_id)
        self._failed.discard(grid_id)
        self._write(grid_id, "done", n_records=n_records, **extra)

    def mark_failed(self, grid_id: str, error: str = "", **extra) -> None:
        self._failed.add(grid_id)
        self._write(grid_id, "failed", error=str(error)[:400], **extra)

    def summary(self) -> str:
        return f"{len(self._done)} done, {len(self.failed)} failed ({self.path.name})"
