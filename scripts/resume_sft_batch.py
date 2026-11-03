"""Compatibility entry for the original SFT batch-resume command."""
from resume_batch import main
import subprocess

if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode if exc.returncode > 0 else 128 - exc.returncode)
