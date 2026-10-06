import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from subjectmate import config  # noqa: E402

requires_index = pytest.mark.skipif(
    not (config.INDEX_DIR / "meta.json").exists(),
    reason="index not built (run scripts/build_index.py)",
)
