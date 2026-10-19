"""Estimate optimization time from real metrics; excludes preflight/evaluation/checkpoint I/O."""
import argparse
import json
from pathlib import Path
import statistics


def estimate(run,*,warmup=10,targets=(2000,6000)):
    samples={}
    for file in sorted(Path(run).glob('metrics-*.jsonl')):
        with file.open() as handle:
            for line in handle:
                if not line.strip():continue
                try:row=json.loads(line)
                except json.JSONDecodeError:
                    if handle.read().strip():raise  # Only an incomplete final line is tolerated.
                    break
                seconds=row.get('cost',{}).get('training_seconds')
                if seconds is not None:samples[row['step']]=(seconds,row)
    ordered=sorted(samples.items())
    durations=[b[1][0]-a[1][0] for a,b in zip(ordered,ordered[1:])
               if b[0]==a[0]+1 and a[0]>=warmup and b[1][0]>a[1][0]]
    if not durations:raise ValueError('Need consecutive completed updates after warmup')
    median=statistics.median(durations)
    last_step,(_,last)=ordered[-1]
    config_file=Path(run)/'training.json'
    identity=json.loads(config_file.read_text()) if config_file.exists() else {}
    t=identity.get('config',{}).get('training',{})
    return {'run':str(Path(run).resolve()),'gpu':identity.get('gpu'),'world_size':identity.get('world_size'),
        'method':{'noise_views':t.get('noise_views'),'rollout':t.get('independent_rollout')},
        'effective_pair_batch':last.get('effective_batch_size_pairs'),'samples_after_warmup':len(durations),
        'median_seconds_per_update':median,'min_seconds_per_update':min(durations),'max_seconds_per_update':max(durations),
        'peak_rank0_allocated_GiB':max(r[1][1].get('peak_cuda_bytes',0) for r in ordered)/1024**3,
        'optimization_hours':{str(n):median*n/3600 for n in targets},
        'remaining_optimization_hours':{str(n):max(0,n-last_step)*median/3600 for n in targets},
        'excludes':'Startup, full dataset validation, diagnostics, checkpoints, generation/scoring, queue/download time. Rank0 memory is not whole-node memory.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--steps',nargs='+',type=int,default=[2000,6000])
    a=p.parse_args()
    if a.warmup<0 or any(n<1 for n in a.steps):p.error('warmup >= 0 and steps >= 1 required')
    print(json.dumps(estimate(a.run,warmup=a.warmup,targets=a.steps),indent=2))


if __name__=='__main__':main()
