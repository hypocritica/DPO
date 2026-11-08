"""Freeze and run one flow LoRA experiment on already allocated GPUs."""
import argparse
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from self_correct_dpo.common import read_json
from self_correct_dpo.config import load_config
from self_correct_dpo.resources import load_resources
from self_correct_dpo.training import validate_config
from self_correct_dpo.runtime.checkpoints import atomic_json
from self_correct_dpo.train import training_lock
from run_sdxl_h100 import prepare_output


def build_config(family, algorithm, *, world=2, views=2, budget_unit=None, budget=128,
                 profile='h100', beta=None, steps=3200, stagewise=False, boundary=500,
                 resources=None, smoke=False, recipe=None, batch_size=None, weight_dtype=None, aggregation='C', smoke_steps=2,
                 smoke_batch_size=1, smoke_accumulation_steps=1):
    if type(smoke_steps) is not int or not 1 <= smoke_steps <= 200:
        raise ValueError('smoke-steps must be an integer in 1..200')
    if any(type(v) is not int or v < 1 for v in (smoke_batch_size, smoke_accumulation_steps)):
        raise ValueError('smoke batch-size and accumulation must be positive integers')
    if not smoke and (smoke_batch_size != 1 or smoke_accumulation_steps != 1):
        raise ValueError('smoke batch-size/accumulation require --smoke')
    if not smoke and smoke_steps != 2:
        raise ValueError('smoke-steps requires --smoke')
    if recipe not in (None, 'paper-pilot', 'paper-v2'): raise ValueError('Unknown flow recipe')
    if recipe and smoke: raise ValueError('paper-pilot is an effect experiment, not a smoke test')
    budget_unit = budget_unit or ('pairs' if recipe == 'paper-pilot' or algorithm == 'sft' else 'pair_views')
    if budget_unit not in ('pairs', 'pair_views'): raise ValueError('Unknown comparison budget')
    if world not in ((1,2,4) if smoke else (2,4)): raise ValueError('Formal flow recipes support 2/4 GPUs; single GPU requires --smoke')
    if family == 'zimage' and profile not in ('h100', 'a800', '5090'):
        raise ValueError('Unknown ZImage GPU profile')
    if family == 'zimage' and profile == '5090' and weight_dtype != 'bfloat16':
        raise ValueError('ZImage 5090 requires explicit --weight-dtype bfloat16; this is a separate precision experiment')
    if weight_dtype not in (None, 'float32', 'bfloat16'):
        raise ValueError('weight_dtype must be float32 or bfloat16')
    if aggregation not in ('B', 'C') or (algorithm != 'mvpo' and aggregation != 'C'):
        raise ValueError('B aggregation applies to MVPO only')
    if family == 'sd35_medium' and profile not in ('h100', 'a800', '5090'):
        raise ValueError('Unknown SD35 GPU profile')
    if type(views) is not int or views < 1: raise ValueError('views must be a positive integer')
    if algorithm != 'mvpo' and (views != 2 or stagewise):
        raise ValueError('Explicit views/stagewise apply to MVPO only')
    c = load_config(ROOT / 'configs/flow' / f'{family}-{algorithm}.json', resources=resources)
    t = c['training']
    if weight_dtype is not None:
        if family == 'zimage': c['model']['transformer_dtype'] = weight_dtype
        else: t['dtype'] = weight_dtype
    if algorithm == 'mvpo':
        c['method'] = f'multiview-{aggregation}-gaussian'
        t['formal_arm'] = aggregation
    k = views if algorithm == 'mvpo' else 1
    if batch_size is None:
        batch_size = (4 if family == 'sd35_medium' else 2) if recipe == 'paper-pilot' and profile in ('h100', 'a800') else 1
    if type(batch_size) is not int or batch_size < 1: raise ValueError('batch-size must be a positive integer')
    denominator = world * batch_size * (k if budget_unit == 'pair_views' else 1)
    if budget < 1 or budget % denominator: raise ValueError('Budget must divide GPUs * batch-size * K exactly')
    t.update(world_size=world, batch_size=batch_size, noise_views=k, gradient_accumulation_steps=budget//denominator,
             comparison_budget={'unit':budget_unit,'per_update':budget}, max_steps=steps)
    if recipe in ('paper-pilot', 'paper-v2'):
        # Borrow the published SD3-M recipe as a starting point; this is a
        # LoRA/backbone transfer, not an exact full-weight reproduction.
        t.update(learning_rate=5e-6, beta=500, warmup_steps=200,
                 offload_frozen=False, vae_batch_size=1 if profile == '5090' else 4,
                 lora_checkpoint_every=100, checkpoint_every=200, full_checkpoint_every=400, preflight='once')
        # Keep every evaluated adapter without retaining intervening snapshots.
        points = [s for s in (200, 400, 800, 1300, 1800, 2300, 2800) if s <= steps]
        points = sorted(set(points + [steps]))
        t['lora_checkpoint_steps'] = points
        c['evaluation'].update(expected_prompts=64, steps=points,
                               keep_images_every=100, include_base=True)
        c['evaluation']['prompts'] = load_resources(resources)['MONITOR_PROMPTS_FLOW_PILOT']
        c['experiment_note'] += (' paper-pilot: Linear-DPO SD3-M hyperparameters transferred to LoRA; '
            'same pairs/update by default, MVPO K views incur extra compute; 64-prompt validation only.')
    if recipe == 'paper-v2':
        # Preparation validates bytes once. New runs consume passed receipts and
        # skip repeated image/schema/weight inventories at training startup.
        c['data'].update(validation='none', validation_report=str(Path(c['data']['pairs']).parent/'validation-report.json'))
        c['model']['validation_report'] = str(Path(c['model']['path'])/'flow-validation-report.json')
        t.update(preflight='once', offload_frozen=profile == '5090', vae_batch_size=1,
                 checkpoint_every=200, full_checkpoint_every=400, lora_checkpoint_every=400)
        # Keep the held-out panel/sampling recipe for a separate evaluation GPU.
        # Saving adapters does not require ending the training process.
        c['evaluation']['enabled'] = False
        c['experiment_note'] += (' paper-v2: certified prepared inputs; no policy EMA. '
            'Linear-DPO alone retains its algorithmic EMA reference. '
            f'Aggregation={aggregation}; weight_dtype={c["model"].get("transformer_dtype", t["dtype"])}.')
    if beta is not None: t['beta'] = beta
    if stagewise:
        c['method'] = f'stagewise-{aggregation}-gaussian'
        t['stagewise'] = {'boundary':boundary,'objective':'dpo','target_margin':1.,'huber_delta':1.,'huber_weight':1.}
    if smoke:
        # Deliberately small compute smoke, separate config/run identity.
        t.update(max_steps=smoke_steps, warmup_steps=0, batch_size=smoke_batch_size,
                 gradient_accumulation_steps=smoke_accumulation_steps,
                 checkpoint_every=1 if smoke_steps<=2 else 10, logging='jsonl')
        t['comparison_budget'] = {'unit':'pair_views','per_update':world*k*smoke_batch_size*smoke_accumulation_steps}
        c['evaluation'] = {'enabled':False}
    if algorithm == 'sft':
        points = sorted({p for p in t['lora_checkpoint_steps'] if p <= t['max_steps']} | {t['max_steps']})
        t['lora_checkpoint_steps'] = points
        if c.get('evaluation', {}).get('enabled'): c['evaluation']['steps'] = points
        winners = t['batch_size'] * t['gradient_accumulation_steps'] * world
        c['experiment_note'] += (f' Winner-only SFT: {winners} winner images/update, K1; '
            'no loser/reference/EMA. Pair-row exposure is matched; DPO compute is higher.')
    c['experiment_note'] += f' Resource profile={profile}; static shifted schedule index sampling.'
    validate_config(c, files=False)
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', required=True, choices=['sd35_medium','zimage'])
    p.add_argument('--algorithm', required=True, choices=['dpo','mvpo','linear_dpo','sft'])
    p.add_argument('--profile', default='h100', choices=['5090','h100','a800'])
    p.add_argument('--world-size', type=int, default=2, choices=[1,2,4])
    p.add_argument('--views', type=int, default=2)
    p.add_argument('--aggregation', choices=['B','C'], default='C')
    p.add_argument('--weight-dtype', choices=['float32','bfloat16'], help='Explicit weight precision variant; required BF16 for ZImage 5090')
    p.add_argument('--data-report', type=Path, help='Reusable passed data validation report')
    p.add_argument('--model-report', type=Path, help='Reusable passed model validation report')
    p.add_argument('--recipe', choices=['paper-pilot','paper-v2'], help='Published-recipe starting point; run all 3200 steps by default')
    p.add_argument('--batch-size', type=int, help='Pairs per GPU; paper-v2 defaults 1; legacy paper-pilot SD35 H100/A800=4, ZImage=2')
    p.add_argument('--budget-unit', choices=['pairs','pair_views'], help='Default pairs for paper-pilot/SFT, pair_views otherwise')
    p.add_argument('--budget', type=int, default=128)
    p.add_argument('--beta', type=float)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--seed', type=int)
    p.add_argument('--ema-decay', type=float)
    p.add_argument('--linear-eta', type=float)
    p.add_argument('--frozen-placement', choices=['resident','offload'])
    p.add_argument('--latest-checkpoint-every', type=int,
                   help='Save full rolling resume state every N optimizer updates; permanent snapshots keep their recipe schedule')
    evaluation = p.add_mutually_exclusive_group()
    evaluation.add_argument('--no-evaluation', action='store_true', help='Continuous training; paper-v2 default')
    evaluation.add_argument('--inline-evaluation', action='store_true', help='Opt into the old pause/evaluate/reload workflow')
    p.add_argument('--steps', type=int, default=3200)
    p.add_argument('--stagewise', action='store_true')
    p.add_argument('--stage-boundary', type=int, default=500)
    p.add_argument('--run-tag', required=True)
    p.add_argument('--resources')
    p.add_argument('--model', type=Path)
    p.add_argument('--model-revision')
    p.add_argument('--pairs', type=Path)
    p.add_argument('--monitor-prompts', type=Path)
    p.add_argument('--scorers-config', type=Path)
    p.add_argument('--runs-root', type=Path)
    p.add_argument('--config-dir', type=Path, default=ROOT/'configs')
    p.add_argument('--config-only', action='store_true')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-steps', type=int, default=2, help='1..200 updates for --smoke; default 2; debug launcher uses 50')
    p.add_argument('--smoke-batch-size', type=int, default=1, help='Pairs per GPU for --smoke only; use a new run tag')
    p.add_argument('--smoke-accumulation-steps', type=int, default=1, help='Accumulation for --smoke only; match B*accumulation in throughput comparisons')
    p.add_argument('--stop-after-step', type=int)
    p.add_argument('--execution', default='slurm', choices=['slurm','standalone'])
    p.add_argument('--run-label')
    p.add_argument('--report-to', choices=['tensorboard', 'wandb', 'all', 'none', 'jsonl'])
    p.add_argument('--wandb-project', default='mvpo-flow')
    p.add_argument('--wandb-group')
    p.add_argument('--wandb-name')
    p.add_argument('--wandb-mode', choices=['offline', 'online', 'disabled'], default='offline')
    p.add_argument('--wandb-images', type=int, default=8, help='Evaluation images per point, 0 disables media logging')
    a = p.parse_args()
    if a.latest_checkpoint_every is not None and a.latest_checkpoint_every < 1:
        p.error('--latest-checkpoint-every must be a positive integer')
    if not 0 <= a.wandb_images <= 64: p.error('--wandb-images must be between 0 and 64')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',a.run_tag): p.error('Invalid run-tag')
    a.budget_unit = a.budget_unit or ('pairs' if a.recipe == 'paper-pilot' or a.algorithm == 'sft' else 'pair_views')
    c = build_config(a.family,a.algorithm,world=a.world_size,views=a.views,budget_unit=a.budget_unit,
        budget=a.budget,profile=a.profile,beta=a.beta,steps=a.steps,stagewise=a.stagewise,
        boundary=a.stage_boundary,resources=a.resources,smoke=a.smoke,recipe=a.recipe,batch_size=a.batch_size,
        weight_dtype=a.weight_dtype,aggregation=a.aggregation,smoke_steps=a.smoke_steps,
        smoke_batch_size=a.smoke_batch_size,smoke_accumulation_steps=a.smoke_accumulation_steps)
    if a.latest_checkpoint_every is not None:
        c['training']['checkpoint_every'] = a.latest_checkpoint_every
    for option, key in ((a.learning_rate,'learning_rate'),(a.ema_decay,'ema_decay'),(a.linear_eta,'linear_eta')):
        if option is not None: c['training'][key] = option
    if a.seed is not None: c['seed'] = a.seed
    if a.frozen_placement: c['training']['offload_frozen'] = a.frozen_placement == 'offload'
    if a.no_evaluation: c['evaluation']['enabled'] = False
    if a.inline_evaluation:
        if a.smoke: p.error('--inline-evaluation is not supported by smoke runs')
        c['evaluation']['enabled'] = True
    if a.report_to or a.recipe == 'paper-v2':
        c['training']['logging'] = a.report_to or 'all'
        c['training']['wandb'] = {
            'project': a.wandb_project, 'mode': a.wandb_mode,
            'image_count': a.wandb_images,
            'group': a.wandb_group or f'{a.family}-hpdv3-{a.budget_unit}{a.budget}-seed{c["seed"]}',
            'tags': [a.family, a.algorithm, f'k{c["training"]["noise_views"]}',
                     'stagewise' if a.stagewise else 'global', a.profile, a.recipe or 'custom'],
        }
        if a.wandb_name: c['training']['wandb']['name'] = a.wandb_name
    validate_config(c,files=False)
    if a.model: c['model']['path'] = str(a.model.expanduser().resolve())
    report_path = a.model_report or (Path(c['model']['path'])/'flow-validation-report.json' if a.recipe == 'paper-v2' else None)
    if a.model_revision: c['model']['revision'] = a.model_revision
    elif report_path and report_path.is_file():
        c['model']['revision'] = read_json(report_path)['revision']
    else:
        receipt = Path(c['model']['path'])/'.scd-download.json'
        if receipt.is_file():
            info=read_json(receipt)
            if not info.get('complete'): raise ValueError('Model download incomplete')
            c['model']['revision']=info['revision']
    if a.pairs: c['data']['pairs'] = str(a.pairs.expanduser().resolve())
    if a.recipe == 'paper-v2' or a.data_report:
        c['data']['validation_report'] = str((a.data_report or Path(c['data']['pairs']).parent/'validation-report.json').expanduser().resolve())
    if a.recipe == 'paper-v2' or a.model_report:
        c['model']['validation_report'] = str((a.model_report or Path(c['model']['path'])/'flow-validation-report.json').expanduser().resolve())
    if a.monitor_prompts and not a.smoke: c['evaluation']['prompts'] = str(a.monitor_prompts.expanduser().resolve())
    if a.scorers_config and c.get('evaluation', {}).get('sampling'):
        c['evaluation']['scorers_config'] = str(a.scorers_config.expanduser().resolve())
    k = c['training']['noise_views']
    name_steps = a.smoke_steps if a.smoke and a.smoke_steps != 2 else a.steps
    name = f'{a.family}-{a.algorithm}-k{k}-{a.budget_unit}{a.budget}-{a.world_size}{a.profile}-b{c["training"]["batch_size"]}-{name_steps}'
    if a.recipe == 'paper-v2':
        precision = c['model'].get('transformer_dtype', c['training']['dtype'])
        name += f'-{a.aggregation}-{precision}'
        if not c['evaluation'].get('enabled'):
            name += '-trainonly'
        print(f'Profile: {a.world_size}x{a.profile}; weights: {precision}; policy EMA: disabled; reference: {c["training"]["reference_mode"]}', flush=True)
    name += f'-stage{a.stage_boundary}' if a.stagewise else ''
    if a.smoke:
        name += '-smoke' + (str(a.smoke_steps) if a.smoke_steps != 2 else '')
        if a.smoke_batch_size != 1 or a.smoke_accumulation_steps != 1:
            name += f'-pv{c["training"]["comparison_budget"]["per_update"]}-acc{a.smoke_accumulation_steps}'
    name += ('-' + a.recipe if a.recipe else '') + '-' + a.run_tag
    output = (a.runs_root or Path(load_resources(a.resources)['RUNS_ROOT'])).resolve()/name
    frozen = a.config_dir.resolve()/f'local-{name}.json'
    with training_lock(str(output)+'.launcher'):
        if frozen.exists():
            if read_json(frozen) != c: raise ValueError('Frozen configuration differs; choose a new run tag')
        else:
            if (output/'training.json').is_file() and read_json(output/'training.json')['config'] != c:
                raise ValueError('Existing run configuration differs')
            frozen.parent.mkdir(parents=True,exist_ok=True)
            atomic_json(frozen,c)
        print(f'Config: {frozen}\nOutput: {output}',flush=True)
        if a.config_only: return
        if not a.check_only:
            if not (output/'training.json').is_file():
                from check_flow_environment import check_environment
                check_environment(a.family)
            if prepare_output(output,c): return
        command=[sys.executable,'-u',str(ROOT/'train.py'),'--config',str(frozen),'--out',str(output),
                 '--execution',a.execution]
        if a.resources: command += ['--resources',a.resources]
        if a.run_label: command += ['--run-label',a.run_label]
        if a.check_only: command += ['--check-only']
        elif (output/'checkpoints/latest.json').is_file(): command += ['--resume_from_checkpoint','latest']
        if a.stop_after_step is not None: command += ['--stop-after-step',str(a.stop_after_step)]
        from self_correct_dpo.evaluation.runner import run_command
        run_command(command)


if __name__ == '__main__': main()
