#!/usr/bin/env python3
"""Plan, inventory, run and report the complete multiseed appendix."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from self_correct_dpo.evaluation.multiseed_generation import main
if __name__=='__main__':raise SystemExit(main())
