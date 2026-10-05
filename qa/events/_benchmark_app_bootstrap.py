"""Add the application source and sibling script directory to import paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path

EVENTS_DIR = Path(__file__).resolve().parent
SOURCE_DIR = Path(os.environ.get("GW_BENCHMARK_SOURCE_DIR", "/work/server/src"))

for _path in (EVENTS_DIR, SOURCE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
