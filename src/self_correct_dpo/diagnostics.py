"""Repeated denoising diagnostics, distinct from generated image quality."""
import json
import random
from pathlib import Path
from self_correct_dpo.common import digest


def dpo_diagnostic(backend, rows, config, out, step, device, attempt, *, split="train"):
    import torch
    import numpy as np
    from self_correct_dpo.losses import preference_loss
    from self_correct_dpo.training import image_batch
    settings = config['training']['diagnostic']
    chosen = sorted(rows, key=lambda r: digest([settings['seed'], r['pair_id']]))[:settings['pairs']]
    prompts = [[next(p['prompt'] for p in r['prompts'] if p['variant_id'] == 'canonical')] for r in chosen]
    devices = [torch.device(device).index or torch.cuda.current_device()] if str(device).startswith('cuda') else []
    metrics_sum, all_timesteps = {}, []
    # Isolate diagnostic posterior/noise draws and any module RNG from training.
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(settings['seed'])
        generator = torch.Generator(device=device).manual_seed(settings['seed'])
        backend.eval()
        try:
            with torch.no_grad():
                for row, text in zip(chosen, prompts):
                    values = getattr(backend, "ordinary_pair_errors", backend.pair_errors)(image_batch([row], config['training']['resolution'], device), [text], generator)
                    _, metrics = preference_loss(values['policy_errors'], values['reference_errors'], 'dpo', config['training']['beta'])
                    for key, value in metrics.items(): metrics_sum[key] = metrics_sum.get(key, 0.) + float(value) / len(chosen)
                    if 'timesteps' in values: all_timesteps.append(values['timesteps'].cpu().tolist())
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            backend.train()
    record = {'step': step, 'split': split, 'attempt': attempt, 'pair_ids': [r['pair_id'] for r in chosen],
              'seed': settings['seed'], 'timesteps': all_timesteps, 'metrics': metrics_sum,
              'state_distribution': 'ordinary_uniform_noise',
              'interpretation': 'Ordinary uniform-noise diagnostic: fixed VAE posterior, Gaussian noise and timestep replay; optimization diagnostic, not generation gain.'}
    with open(Path(out) / 'dpo-diagnostic.jsonl', 'a') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')
    return record
