"""Create a fresh experiment config without changing any archived template."""
import argparse
import copy
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from self_correct_dpo.training import validate_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=['sd15','sdxl'], help='Select family defaults; omitted: SD1.5 or the supplied template family')
    p.add_argument("--template", help='Override the family template')
    p.add_argument("--out", required=True)
    p.add_argument("--model", help="Local Diffusers pipeline; otherwise SD15_MODEL/SDXL_MODEL from paths.sh")
    p.add_argument("--model-revision", help='Verified base checkpoint revision/provenance')
    p.add_argument("--vae", help='Optional local SDXL AutoencoderKL directory, e.g. sdxl-vae-fp16-fix')
    p.add_argument("--vae-revision", help='Required provenance when --vae is used')
    p.add_argument("--pairs", help='Prepared pair manifest; otherwise model-specific path from paths.sh')
    p.add_argument('--portable-paths', action='store_true', help='Keep paths.sh variable references in the new config; source paths.sh on every run')
    p.add_argument('--dataset',choices=['pickapic','hpdv3'],default='pickapic',help='Model-specific prepared manifest from paths.sh')
    p.add_argument("--diagnostic-suite", help="Required with historical diagnostic-suite templates")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--accumulation", type=int)
    p.add_argument("--world-size", type=int, help='Single-node GPU/process count, e.g. 2, 4 or 8')
    p.add_argument("--effective-batch",type=int,help='Global pair batch; recompute accumulation for the chosen GPU and per-rank batch counts')
    p.add_argument("--steps", type=int)
    p.add_argument("--resolution", type=int, help='Square training and monitoring resolution; defaults: SD1.5=512, SDXL=1024')
    p.add_argument("--arm", choices=['B','C'])
    p.add_argument('--method',choices=['dpo','k2-noise','k2-rollout'],
                   help='dpo: K1/no rollout; k2-noise: two ordinary noise views; k2-rollout: two model perturbation views')
    p.add_argument('--timestep-sampling',choices=['original','uniform_final'],help='Explicit model rollout timestep law')
    p.add_argument('--benchmark',action='store_true',help='100 updates by default, no evaluation/diagnostics, checkpoint at the final update')
    p.add_argument("--monitor-prompts", help='Frozen Pick-a-Pic test monitoring JSONL')
    p.add_argument("--scorers-config", help='Local scorer assets JSON')
    p.add_argument("--smoke", action="store_true", help="batch1, accumulation1, two updates; retain family resolution")
    args = p.parse_args()
    family=args.family or 'sd15'
    default=ROOT/('configs/sdxl/model-K2-C.json' if family=='sdxl' else 'configs/model-K2-C.json')
    c = copy.deepcopy(json.loads(Path(args.template or default).read_text()))
    if args.family and c['model']['family']!=args.family:p.error('--family differs from the supplied template')
    family=c['model']['family']
    model_variable='SDXL_MODEL' if family=='sdxl' else 'SD15_MODEL'
    prepared_variable=('HPDV3' if args.dataset=='hpdv3' else 'PICKAPIC')+('_SDXL_PREPARED' if family=='sdxl' else '_SD15_PREPARED')
    model_path=args.model or os.environ.get(model_variable)
    prepared=os.environ.get(prepared_variable)
    pairs_path=args.pairs or (str(Path(prepared)/'pairs.jsonl') if prepared else os.environ.get('PAIRS_PATH'))
    if not model_path or not pairs_path:p.error('Provide --model and --pairs, or edit and source paths.sh')
    if 'CHANGE_ME' in model_path or 'CHANGE_ME' in pairs_path:p.error('Replace CHANGE_ME in paths.sh first')
    if args.portable_paths and (args.model or args.pairs or not prepared or not os.environ.get(model_variable)):
        p.error('--portable-paths uses model-specific paths.sh variables; omit --model/--pairs and source paths.sh')
    c["model"]["path"] = str(Path(model_path).expanduser().resolve())
    receipt=Path(c['model']['path'])/'.scd-download.json'
    if receipt.is_file() and not args.model_revision:
        identity=json.loads(receipt.read_text())
        if not identity.get('complete'):p.error('Model download is incomplete')
        c['model']['revision']=identity['revision']
    if args.model_revision:c['model']['revision']=args.model_revision
    if args.vae:
        if family!='sdxl' or not args.vae_revision:p.error('--vae requires --family sdxl (or an SDXL template) and --vae-revision')
        c['model']['vae']={'path':str(Path(args.vae).expanduser().resolve()),'revision':args.vae_revision}
    elif args.vae_revision:
        if not c['model'].get('vae'):p.error('--vae-revision requires --vae or a template VAE')
        c['model']['vae']['revision']=args.vae_revision
    c["data"]["pairs"] = str(Path(pairs_path).expanduser().resolve())
    if args.diagnostic_suite:
        c["data"]["diagnostic_suite"] = str(Path(args.diagnostic_suite).expanduser().resolve())
    elif c["data"].get("diagnostic_suite"):
        p.error("Historical template requires --diagnostic-suite; use the main template for new datasets")
    t = c["training"]
    original_effective=t['batch_size']*t['gradient_accumulation_steps']*t.get('world_size',1)
    t['formal_arm']=args.arm or t.get('formal_arm','C')
    if args.method:
        t['noise_views']=1 if args.method=='dpo' else 2
        if args.method in ('dpo','k2-noise'):
            t.pop('independent_rollout',None)
        else:
            t['independent_rollout']=dict(t.get('independent_rollout',
                {'stride':33,'min_anchor_t':20,'cfg_scale':1.0}))
            t['independent_rollout']['state_source']='model'
        c['experiment_note']=f"Method selected by make_config: {args.method}; optimizer step counts complete global pair batches."
    if args.timestep_sampling:
        if 'independent_rollout' not in t:p.error('--timestep-sampling requires a rollout method')
        t['independent_rollout']['timestep_sampling']=args.timestep_sampling
    if args.smoke and args.benchmark:p.error('Choose --smoke or --benchmark')
    if (args.monitor_prompts or args.scorers_config) and not c.get('evaluation'):
        template='configs/sdxl/model-K2-C.json' if family=='sdxl' else 'configs/model-K2-C.json'
        c['evaluation']=json.loads((ROOT/template).read_text())['evaluation']
    if args.monitor_prompts:
        c['evaluation']['prompts']=str(Path(args.monitor_prompts).expanduser().resolve())
    if args.scorers_config:
        c['evaluation']['scorers_config']=str(Path(args.scorers_config).expanduser().resolve())
    if args.resolution is not None:
        t['resolution']=args.resolution
        if c.get('evaluation',{}).get('enabled'):
            c['evaluation']['sampling'].update(height=args.resolution,width=args.resolution)
    if args.smoke:
        t.update(batch_size=1, gradient_accumulation_steps=1, max_steps=2,
                 warmup_steps=0, checkpoint_every=1, checkpoint_fractions=[1.],
                 max_lora_checkpoints=1, logging="jsonl", diagnostic={})
        c['evaluation']={'enabled':False}
        t.pop('lora_checkpoint_every',None)
    if args.benchmark:
        t.update(max_steps=100,warmup_steps=0,checkpoint_every=100,diagnostic={},logging='jsonl')
        c['evaluation']={'enabled':False}
    for arg, key in ((args.batch_size,"batch_size"), (args.accumulation,"gradient_accumulation_steps"), (args.steps,"max_steps")):
        if arg is not None:
            t[key] = arg
    if args.world_size is not None:t['world_size']=args.world_size
    if args.benchmark:t['checkpoint_every']=t['max_steps']
    if args.effective_batch is not None or (args.world_size is not None and args.accumulation is None and not args.smoke):
        if args.effective_batch is not None and args.accumulation is not None:p.error('Choose --effective-batch or --accumulation')
        target=args.effective_batch if args.effective_batch is not None else original_effective
        unit=t['batch_size']*t.get('world_size',1)
        if target<1 or unit<1 or target%unit:p.error('Global effective batch must be positive and divisible by batch-size * world-size')
        t['gradient_accumulation_steps']=target//unit
    validate_config(c, files=False)
    if args.portable_paths:
        c['model']['path']='${'+model_variable+'}'
        c['data']['pairs']='${'+prepared_variable+'}/pairs.jsonl'
        if c['model'].get('vae'):
            if not os.environ.get('SDXL_VAE') or c['model']['vae']['path']!=str(Path(os.environ['SDXL_VAE']).expanduser().resolve()):
                p.error('--portable-paths external VAE must match SDXL_VAE in paths.sh')
            c['model']['vae']['path']='${SDXL_VAE}'
        if c.get('evaluation',{}).get('enabled'):
            if args.monitor_prompts or args.scorers_config:
                p.error('--portable-paths uses MONITOR_PROMPTS/SCORERS_CONFIG from paths.sh; omit the corresponding flags')
            c['evaluation']['prompts']='${MONITOR_PROMPTS}'
            c['evaluation']['scorers_config']='${SCORERS_CONFIG}'
    with open(args.out, "x") as f:
        json.dump(c, f, indent=2); f.write("\n")
    print(json.dumps({"config": args.out, "world_size":t.get('world_size',1),"effective_batch": t["batch_size"] * t["gradient_accumulation_steps"] * t.get('world_size',1), "steps":t["max_steps"]}))


if __name__ == "__main__":
    main()
