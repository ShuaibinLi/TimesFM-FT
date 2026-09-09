"""Test path setup for editable source and the vendored TimesFM submodule."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for source_path in (ROOT / "src", ROOT / "3rdparty" / "timesfm" / "src"):
    sys.path.insert(0, str(source_path))
