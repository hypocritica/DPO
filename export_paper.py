"""Export verified benchmark results into versioned paper tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from self_correct_dpo.evaluation.paper import main

if __name__ == '__main__':
    raise SystemExit(main())
