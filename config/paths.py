import os
import sys
from pathlib import Path

def get_project_root() -> Path:
    path = Path(__file__).resolve().parent
    while not (path / "phase3").exists() and not (path / "progress.md").exists():
        parent = path.parent
        if parent == path:
            break
        path = parent
    return path

PROJECT_ROOT = get_project_root()

# Ensure project root is in sys.path for absolute imports
root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

# Directory Mappings
DATA_PHASE1 = PROJECT_ROOT / "phase1"
DATA_PHASE2 = PROJECT_ROOT / "phase2"
PHASE3_DIR = PROJECT_ROOT / "phase3"
PHASE4_DIR = PROJECT_ROOT / "phase4"
PHASE4B_DIR = PROJECT_ROOT / "phase4b"
PHASE5_DIR = PROJECT_ROOT / "phase5"
PHASE6_DIR = PROJECT_ROOT / "phase6"
PHASE7_DIR = PROJECT_ROOT / "phase7"
UTILS_DIR = PROJECT_ROOT / "utils"
