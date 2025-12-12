from pathlib import Path
import sys

# Ensure project root is on sys.path so pytest can import the single-file module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
