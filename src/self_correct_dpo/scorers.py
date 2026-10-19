"""Local-weight scoring, with explicitly recorded scales and versions.

The defaults follow Linear-DPO revision 663179c7: raw paired cosine for
PickScore, ViT-H/14 CLIP and HPS v2.0; improved LAION aesthetic MLP.
"""
import importlib.util
import sys
from pathlib import Path

from self_correct_dpo.common import artifact_identity, digest, sha256

DEFAULT_METRICS = ('pickscore', 'clip', 'hpsv2', 'aesthetic')
METRICS = (*DEFAULT_METRICS, 'hpsv3', 'image_reward')


def scorer_identity(metric, spec):
    if metric not in METRICS:
        raise ValueError(f'Unsupported scorer: {metric}')
    required = {'pickscore':('model','processor'), 'clip':('checkpoint','open_clip_source'),
                'hpsv2':('checkpoint','open_clip_source'),
                'aesthetic':('model','processor','checkpoint'),
                'hpsv3':('source','model','checkpoint','config'),
                'image_reward':('source','checkpoint','config','tokenizer')}[metric]
    if metric in ('hpsv3','image_reward'):
        from self_correct_dpo.optional_scorers import validate_spec
        validate_spec(metric,spec)
    if spec.get('python'):
        required=(*required,'python')
    assets = {}
    for key in required:
        path = Path(spec[key]).resolve(strict=True)
        if key == 'source':
            package=path/('hpsv3' if metric=='hpsv3' else 'ImageReward')
            assets[key]={str(p.relative_to(path)):sha256(p) for p in sorted(package.rglob('*'))
                         if p.is_file() and p.suffix in {'.py','.yaml','.json'}}
        elif key == 'open_clip_source':
            package = path / 'open_clip'
            if not (package / '__init__.py').is_file():
                raise ValueError('open_clip_source must contain the official HPSv2 open_clip package')
            assets[key] = {str(p.relative_to(path)):sha256(p) for p in sorted(package.rglob('*'))
                           if p.is_file() and p.suffix in {'.py','.json','.gz'}}
        elif path.is_file():
            stat=path.stat()
            assets[key] = {'path':str(path), 'bytes':stat.st_size, 'mtime_ns':stat.st_mtime_ns}
            if stat.st_size <= 64*1024**2:
                assets[key]['sha256']=sha256(path)
            else:
                assets[key]['fingerprint_kind']='inventory only; not a full weight hash'
        else:
            assets[key] = artifact_identity(path)
    version = spec.get('version', 'v2.0') if metric == 'hpsv2' else None
    if metric == 'hpsv2':
        if version not in ('v2.0', 'v2.1'):
            raise ValueError('HPS version must be v2.0 or v2.1')
        expected = 'HPS_v2_compressed.pt' if version == 'v2.0' else 'HPS_v2.1_compressed.pt'
        if Path(spec['checkpoint']).name != expected:
            raise ValueError(f'{version} requires the explicitly named {expected}; do not silently mix HPS versions')
    if metric=='hpsv3': version='v3'
    if metric=='image_reward': version='v1.0'
    scale={'aesthetic':'raw improved-LAION MLP output','hpsv3':'official HPSv3 mu (logits[:,0]); no sigmoid',
           'image_reward':'official ImageReward v1.0 standardized reward (raw-mean)/std'}.get(metric,'raw cosine, no softmax or temperature')
    result={'metric':metric, 'spec':spec, 'assets':assets, 'version':version, 'scale':scale,
            'text_truncation':'BERT max_length=35' if metric=='image_reward' else
                              'official HPSv3 multimodal processor' if metric=='hpsv3' else 'CLIP context length, benchmark convention',
            'source_sha256':sha256(__file__)}
    if metric in ('hpsv3','image_reward'):
        result['adapter_sha256']=sha256(Path(__file__).with_name('optional_scorers.py'))
    return result


def aesthetic_head():
    import torch.nn as nn
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            widths = (768,1024,128,64,16,1)
            layers = []
            for i, (a,b) in enumerate(zip(widths, widths[1:])):
                layers.append(nn.Linear(a,b))
                if i < 3:
                    layers.append(nn.Dropout((.2,.2,.1)[i]))
            self.layers = nn.Sequential(*layers)

        def forward(self, features):
            return self.layers(features).squeeze(-1)
    return Head()


def _open_clip(source):
    package = Path(source).resolve() / 'open_clip'
    name = '_self_correct_hps_open_clip_' + digest(str(package))[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, package / '__init__.py',
                                                    submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_scorer(metric, spec, device='cuda'):
    if metric in ('hpsv3','image_reward'):
        from self_correct_dpo.optional_scorers import load_optional_scorer
        return load_optional_scorer(metric,spec,device=device)
    import torch
    import torch.nn.functional as F
    if metric in ('pickscore','aesthetic'):
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(spec['model'], local_files_only=True,
                                          torch_dtype=torch.float32).eval().to(device)
        processor = AutoProcessor.from_pretrained(spec['processor'], local_files_only=True, use_fast=False)
        head = None
        if metric == 'aesthetic':
            if int(model.config.projection_dim) != 768:
                raise ValueError('Improved LAION aesthetic requires OpenAI CLIP ViT-L/14, projection_dim=768')
            head = aesthetic_head().eval().to(device)
            head.load_state_dict(torch.load(spec['checkpoint'], map_location='cpu', weights_only=True), strict=True)

        @torch.inference_mode()
        def score(images, prompts):
            inputs = processor(images=images, return_tensors='pt')
            features = F.normalize(model.get_image_features(**{k:v.to(device) for k,v in inputs.items()}).float(),dim=-1)
            if head is not None:
                return head(features).cpu().tolist()
            texts = processor(text=prompts, padding=True, truncation=True,
                              max_length=model.config.text_config.max_position_embeddings, return_tensors='pt')
            text = F.normalize(model.get_text_features(**{k:v.to(device) for k,v in texts.items()}).float(),dim=-1)
            return (features * text).sum(-1).cpu().tolist()
        return score
    if metric not in ('clip','hpsv2'):
        raise ValueError(metric)
    api = _open_clip(spec['open_clip_source'])
    # No pretrained identifier: instantiate locally and load the FULL state dict.
    # This avoids the upstream wrapper's automatic model downloads.
    model, _, preprocess = api.create_model_and_transforms('ViT-H-14', pretrained=None,
        precision='fp32', device=device, output_dict=True, light_augmentation=True,
        with_score_predictor=False, with_region_predictor=False, pretrained_image=False)
    payload = torch.load(spec['checkpoint'], map_location='cpu', weights_only=True)
    state = payload.get('state_dict', payload)
    state = {k.removeprefix('module.'):v for k,v in state.items()}
    model.load_state_dict(state, strict=True)
    del payload, state
    model.eval()
    tokenizer = api.get_tokenizer('ViT-H-14')

    @torch.inference_mode()
    def score(images, prompts):
        pixels = torch.stack([preprocess(im) for im in images]).to(device)
        texts = tokenizer(prompts).to(device)
        values = model(pixels, texts)
        image = F.normalize(values['image_features'].float(), dim=-1)
        text = F.normalize(values['text_features'].float(), dim=-1)
        return (image * text).sum(-1).cpu().tolist()
    return score
