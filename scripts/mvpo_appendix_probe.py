#!/usr/bin/env python3
"""Measure denoising margins and raw LoRA gradients on the appendix checkpoints."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from self_correct_dpo.evaluation.appendix_probe import main


if __name__ == "__main__":
    main()
