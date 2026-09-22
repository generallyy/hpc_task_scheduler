"""Shared test configuration for repository scripts that are not Python packages."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source_directory in (ROOT / "scripts", ROOT / "src"):
    sys.path.insert(0, str(source_directory))
