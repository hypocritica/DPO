"""Prepare optional Pick-a-Pic, PartiPrompt, HPDv2/v3 and ImageReward test panels."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from self_correct_dpo.benchmarks import prepare_panels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pickapic', help='Official test_unique Parquet/HF directory, or its caption TXT/JSON export')
    p.add_argument('--hpdv2', help='HPDv2 test.json or Linear-DPO hpsv2_test400.txt; not the 3200-prompt HPS benchmark')
    p.add_argument('--train-pairs', required=True, help='Training manifest for caption-overlap exclusion')
    p.add_argument('--parti', help='Official PartiPrompts.tsv or a text/JSON export')
    p.add_argument('--hpdv3', help='Local HPDv3 test.json metadata or test caption export')
    p.add_argument('--imagereward', help='Optional ImageReward test metadata/captions, distinct from the scorer')
    p.add_argument('--monitor-count', type=int, default=128)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--out', required=True)
    a=p.parse_args()
    if not any((a.pickapic,a.hpdv2,a.parti,a.hpdv3,a.imagereward)):
        p.error('Provide at least one benchmark input')
    prepare_panels(a.pickapic, a.hpdv2, a.out, train_pairs=a.train_pairs,
                   parti=a.parti,hpdv3=a.hpdv3,imagereward=a.imagereward,count=a.monitor_count, seed=a.seed)


if __name__ == '__main__':
    main()
