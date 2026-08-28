"""Shared path setup for the offline test suite.

Puts the project root on sys.path so tests can `import src.<module>` regardless
of where pytest is invoked from. The modules under test (sql_normalization,
execution_eval, preprocessing) are stdlib-only, so the suite runs offline.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
