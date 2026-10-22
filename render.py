import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from self_correct_dpo.evaluation.render import main

if __name__ == "__main__":
    main()
