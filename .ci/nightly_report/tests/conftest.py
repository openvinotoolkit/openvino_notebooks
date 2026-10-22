import sys
from pathlib import Path

# Make `nightly_report` importable as a top-level package (scripts in .ci are run directly, not installed).
CI_DIR = Path(__file__).resolve().parents[2]
if str(CI_DIR) not in sys.path:
    sys.path.insert(0, str(CI_DIR))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
