"""Fixed eleven-version SD1.5 generation, scoring and report workflow."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from self_correct_dpo.evaluation.sd15_main import main

if __name__ == '__main__':
    raise SystemExit(main())
