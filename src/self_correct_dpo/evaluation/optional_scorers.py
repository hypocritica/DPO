"""Opt-in official HPSv3 and ImageReward adapters with local assets only.

Each scorer can run in its own Python environment. No private scoring service,
download wrapper or replacement reward is used.
"""
import importlib
import os
from pathlib import Path
import sys
import tempfile
import weakref


def validate_spec(metric,spec):
    package='hpsv3' if metric=='hpsv3' else 'ImageReward'
    source=Path(spec['source']).resolve(strict=True)
    if not (source/package/'__init__.py').is_file():
        raise ValueError(f'{metric}.source must contain the official {package} package')
    for name in ('config','checkpoint'):
        if not Path(spec[name]).is_file(): raise ValueError(f'{metric}.{name} must be a local file')
    directory='model' if metric=='hpsv3' else 'tokenizer'
    if not Path(spec[directory]).is_dir():
        raise ValueError(f'{metric}.{directory} must be a local directory')
    if spec.get('python') and (not Path(spec['python']).is_file() or not os.access(spec['python'],os.X_OK)):
        raise ValueError('Scorer python must be an executable in the chosen project environment')
    if type(spec.get('batch_size',1)) is not int or spec.get('batch_size',1)<1:
        raise ValueError('Scorer batch_size must be positive')
    if metric=='hpsv3': hps_config(spec)


def hps_config(spec):
    import yaml
    config=yaml.safe_load(Path(spec['config']).read_text())
    if not isinstance(config,dict): raise ValueError('HPSv3 configuration must be a YAML mapping')
    config['model_name_or_path']=str(Path(spec['model']).resolve(strict=True))
    if type(spec.get('disable_flash_attn2',True)) is not bool:
        raise ValueError('disable_flash_attn2 must be boolean')
    # Use the official model, with a local base and the portable SDPA path.
    config['disable_flash_attn2']=spec.get('disable_flash_attn2',True)
    return config


def source_package(name,source):
    package=Path(source).resolve()/name
    if name in sys.modules:
        module=sys.modules[name]
        if not Path(module.__file__).resolve().is_relative_to(package):
            raise RuntimeError(f'{name} is already imported from a different source')
        return module
    sys.path.insert(0,str(package.parent))
    module=importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(package):
        raise RuntimeError(f'{name} did not load from the configured official source')
    return module


def load_optional_scorer(metric,spec,device='cuda'):
    validate_spec(metric,spec)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    if metric=='hpsv3':return hpsv3_scorer(spec,device)
    if metric=='image_reward':return image_reward_scorer(spec,device)
    raise ValueError(metric)


def hpsv3_scorer(spec,device):
    import torch
    import yaml
    api=source_package('hpsv3',spec['source'])
    temporary=tempfile.TemporaryDirectory(prefix='self-correct-hpsv3-')
    config=hps_config(spec)
    config['output_dir']=str(Path(temporary.name)/'inference')
    path=Path(temporary.name)/'inference.yaml'
    path.write_text(yaml.safe_dump(config))
    try:
        inferencer=api.HPSv3RewardInferencer(config_path=str(path),
            checkpoint_path=str(Path(spec['checkpoint']).resolve()),device=device,differentiable=False)
    except BaseException:
        temporary.cleanup()
        raise

    @torch.inference_mode()
    def score(images,prompts):
        # Keep the generated local configuration alive throughout this worker.
        _=temporary.name
        rewards=inferencer.reward(prompts=prompts,image_paths=images)
        values=rewards if isinstance(rewards,torch.Tensor) else torch.stack([torch.as_tensor(v) for v in rewards])
        if values.ndim!=2 or values.shape!=(len(prompts),2):
            raise ValueError('Official HPSv3 must return two columns per image, with mu in column zero')
        return values[:,0].detach().float().cpu().tolist()
    weakref.finalize(score,temporary.cleanup)
    return score


def _load_image_reward_state(model,state):
    """Require all model weights; only deterministic BERT position buffers may differ."""
    import torch
    expected=model.state_dict()
    state={k.removeprefix('module.'):v for k,v in state.items()}
    for name in list(state):
        if name not in expected and name.endswith('.position_ids'):
            value=state[name]
            if not isinstance(value,torch.Tensor) or value.ndim not in (1,2):
                raise ValueError('Invalid ImageReward position buffer')
            wanted=torch.arange(value.shape[-1]).expand_as(value)
            if not torch.equal(value.cpu(),wanted):raise ValueError('Noncanonical ImageReward position buffer')
            del state[name]
    for name,value in expected.items():
        if name not in state and name.endswith('.position_ids'):state[name]=value
    model.load_state_dict(state,strict=True)


def image_reward_scorer(spec,device):
    import torch
    from transformers import BertTokenizer
    source_package('ImageReward',spec['source'])
    module=importlib.import_module('ImageReward.ImageReward')
    blip=importlib.import_module('ImageReward.models.BLIP.blip_pretrain')
    # The upstream constructor hardcodes bert-base-uncased. Replace only its
    # tokenizer factory during construction, preserving the official special tokens.
    def tokenizer():
        value=BertTokenizer.from_pretrained(spec['tokenizer'],local_files_only=True)
        value.add_special_tokens({'bos_token':'[DEC]'})
        value.add_special_tokens({'additional_special_tokens':['[ENC]']})
        value.enc_token_id=value.additional_special_tokens_ids[0]
        return value
    original=blip.init_tokenizer
    blip.init_tokenizer=tokenizer
    try:
        model=module.ImageReward(med_config=spec['config'],device=device)
    finally:
        blip.init_tokenizer=original
    payload=torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)
    state=payload.get('state_dict',payload.get('model',payload))
    _load_image_reward_state(model,state)
    model=model.to(device=device,dtype=torch.float32).eval()
    model.requires_grad_(False)

    @torch.inference_mode()
    def score(images,prompts):
        return [float(model.score(prompt,image)) for prompt,image in zip(prompts,images)]
    return score
