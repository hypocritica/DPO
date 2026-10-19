"""Train from a source checkout, without requiring an editable installation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from self_correct_dpo.train import main

if __name__ == "__main__":
    main()
