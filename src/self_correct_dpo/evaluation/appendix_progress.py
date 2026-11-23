"""Quiet subprocess logs with journal-based progress and measured stage ETA."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from self_correct_dpo.runtime.checkpoints import atomic_json


def committed_count(path):
    try:
        with Path(path).open('rb') as handle:
            return sum(line.endswith(b'\n') and bool(line.strip()) for line in handle)
    except FileNotFoundError:return 0


def run_stage(command, *, log, journal, total, label, **kwargs):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    progress=Path(os.environ.get('MVPO_PROGRESS_FILE',str(log.with_suffix('.progress.json'))))
    started=time.monotonic();initial=committed_count(journal);proc=None
    try:
        with log.open('a') as handle:
            proc=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,**kwargs)
            while True:
                now=time.monotonic();n=committed_count(journal);elapsed=now-started
                rate=(n-initial)/elapsed if elapsed>0 and n>initial else None
                atomic_json(progress,dict(stage=label,completed=min(n,total),total=total,remaining=max(0,total-n),
                    eta_seconds=max(0,total-n)/rate if rate else None,elapsed_seconds=elapsed,log=str(log)))
                code=proc.poll()
                if code is not None:break
                time.sleep(1)
        if code:
            tail='\n'.join(log.read_text(errors='replace').splitlines()[-12:])
            raise RuntimeError(f'{label} failed ({code}); log={log}\n{tail}')
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()


def run_workers(commands,output,env,progress_files):
    """Two/four independent GPUs; terminate the entire subprocess group on failure."""
    logs=Path(output)/'logs';logs.mkdir(parents=True,exist_ok=True);workers=[];old={}
    def stop(sig,frame):raise KeyboardInterrupt
    for sig in (signal.SIGINT,signal.SIGTERM):old[sig]=signal.signal(sig,stop)
    try:
        for gpu,name,cmd in commands:
            log=logs/f'{name}-{time.time_ns()}.log';progress=Path(progress_files[name]);progress.parent.mkdir(parents=True,exist_ok=True)
            atomic_json(progress,dict(stage='loading / checking reused results',completed=0,total=None,remaining=None,eta_seconds=None))
            with log.open('w') as handle:
                proc=subprocess.Popen(cmd,env={**env,'CUDA_VISIBLE_DEVICES':gpu,'MVPO_PROGRESS_FILE':str(progress)},
                                      stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append((proc,gpu,name,log,progress))
        last=0
        while workers:
            if time.monotonic()-last>=15:
                for _,gpu,name,_,path in workers:
                    try:r=json.loads(path.read_text())
                    except (OSError,ValueError):continue
                    eta=f"{r['eta_seconds']/60:.1f} min" if r.get('eta_seconds') is not None else 'estimating after first batch'
                    counts=f"{r['completed']}/{r['total']}" if r['total'] is not None else 'checking inputs'
                    remaining=r['remaining'] if r['remaining'] is not None else '-'
                    print(f"GPU {gpu} | {name} | {r['stage']} | {counts} | remaining={remaining} | ETA={eta}",flush=True)
                last=time.monotonic()
            for item in workers[:]:
                proc,gpu,name,log,_=item;code=proc.poll()
                if code is None:continue
                if code:raise RuntimeError(f'Worker {name} failed ({code}); {log}\n'+ '\n'.join(log.read_text(errors='replace').splitlines()[-12:]))
                print(f'GPU {gpu} | {name} | completed',flush=True);workers.remove(item)
            if workers:time.sleep(1)
    finally:
        for proc,*_ in workers:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=8)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for sig,handler in old.items():signal.signal(sig,handler)
