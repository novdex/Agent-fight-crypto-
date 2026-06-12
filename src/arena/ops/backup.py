"""Rolling SQLite backups and config fingerprinting."""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def backup_db(db_path: str, *, keep: int = 14) -> Optional[str]:
    """Copy ``db_path`` to ``<db>.bak.<UTCstamp>`` and prune old backups.

    Keeps the ``keep`` newest backups (timestamped names sort
    chronologically); with ``keep <= 0`` nothing is pruned. Returns the new
    backup path, or ``None`` when the source file does not exist.
    """
    src = Path(db_path)
    if not src.is_file():
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    dest = src.with_name(f"{src.name}.bak.{stamp}")
    suffix = 0
    while dest.exists():  # same-microsecond collision; keep names sortable
        suffix += 1
        dest = src.with_name(f"{src.name}.bak.{stamp}-{suffix}")
    shutil.copy2(src, dest)

    if keep > 0:
        backups = sorted(src.parent.glob(f"{src.name}.bak.*"))
        for old in backups[:-keep]:
            try:
                old.unlink()
            except OSError:  # already gone / permission: never fail a backup
                pass

    return str(dest)


def config_hash(config_path: str) -> str:
    """sha256 of the config file bytes, first 12 hex chars ('' when absent)."""
    p = Path(config_path)
    if not p.is_file():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]
