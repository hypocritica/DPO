"""ZImage H100 single-GPU B4 probe; preserves FP32 transformer weights."""
import sys

from probe_sd35_b2 import main


if __name__ == '__main__':
    sys.exit(main(family='zimage', profile='h100', batch_size=4, steps=20))
