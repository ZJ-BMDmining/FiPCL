#!/usr/bin/env python3
"""Train the release FiPCL configuration on one protected fold."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'FiPCL'), str(ROOT)]

from trainer import main


if __name__ == '__main__':
    main()
