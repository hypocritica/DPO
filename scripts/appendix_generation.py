#!/usr/bin/env python3
"""Run the fixed SD3.5-M generation appendix on existing SLAI allocations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from self_correct_dpo.evaluation.appendix_generation import main


if __name__ == '__main__':
    raise SystemExit(main())
