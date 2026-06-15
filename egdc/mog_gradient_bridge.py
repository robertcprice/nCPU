#!/usr/bin/env python3
import sys
from pathlib import Path

# Add the repo root to sys.path so the egdc package imports correctly
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from egdc.mog.solvers.gradient_bridge import main

if __name__ == "__main__":
    sys.exit(main())
