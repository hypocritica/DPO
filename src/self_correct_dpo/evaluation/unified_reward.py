"""Local pointwise ACS evaluation with the pinned official UnifiedReward 2.0 model.

The three axes and prompt are official. Their arithmetic mean is our explicitly
named reporting convention, not an official learned overall reward head.
"""
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import statistics


MODEL_REPOSITORY = 'CodeGoat24/UnifiedReward-2.0-qwen-7b'
MODEL_REVISION = 'f14142a59d75cbdf8c0de75aa942f135e4bd3a70'
SOURCE_REVISION = 'ff6f520247a2d36e2aff7b2ba53a935afd437f66'
REFERENCE_SCRIPT = 'inference_qwen/UnifiedReward-2.0-inference/point_score_ACS_image_generation.py'
AGGREGATION = 'arithmetic_mean_of_alignment_coherence_style'
SCALE = 'mean of official pointwise Alignment/Coherence/Style, each 1-5; project aggregation, no normalization'
TEMPLATE = (
    'You are presented with a generated image and its associated text caption. '
    'Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n'
    'Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n'
    '- Alignment Score: How well the image matches the caption in terms of content.\n'
    '- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n'
    '- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n'
    'Output your evaluation using the format below:\n\n'
    'Alignment Score (1-5): X\n'
    'Coherence Score (1-5): Y\n'
    'Style Score (1-5): Z\n\n'
    'Your task is provided as follows:\n'
    'Text Caption: [{prompt}]'
)


def parse_acs(text):
    """Require exactly three finite, bounded scores; never impute a failed judge."""
    if not isinstance(text, str):
        raise ValueError('UnifiedReward output must be text')
    number = r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
    pattern = r'\s*' + r'\s*'.join(
        re.escape(label + ' Score (1-5):') + r'[ \t]*' + number
        for label in ('Alignment', 'Coherence', 'Style')) + r'\s*'
    match = re.fullmatch(pattern, text)
    if match is None:
        raise ValueError(f'UnifiedReward must return exactly Alignment, Coherence and Style scores: {text[:1000]!r}')
    values = [float(value) for value in match.groups()]
    if any(not math.isfinite(value) or not 1 <= value <= 5 for value in values):
        raise ValueError(f'UnifiedReward ACS scores must be finite and within [1, 5]: {values}')
    return {'score': statistics.mean(values),
            'components': dict(zip(('alignment', 'coherence', 'style'), values)),
            'raw_output': text, 'aggregation': AGGREGATION}


def validate_spec(spec):
    if spec.get('model_revision') != MODEL_REVISION or spec.get('source_revision') != SOURCE_REVISION:
        raise ValueError('Use the pinned UnifiedReward-2.0-qwen-7b model and reference source revisions')
    if spec.get('aggregation') != AGGREGATION:
        raise ValueError('UnifiedReward requires the explicitly named ACS arithmetic-mean convention')
    if spec.get('dtype', 'bfloat16') != 'bfloat16':
        raise ValueError('This UnifiedReward protocol uses bfloat16')
    if type(spec.get('batch_size', 1)) is not int or spec.get('batch_size', 1) != 1:
        raise ValueError('UnifiedReward pointwise evaluation uses batch_size=1')
    if type(spec.get('max_new_tokens', 4096)) is not int or spec.get('max_new_tokens', 4096) < 1:
        raise ValueError('UnifiedReward max_new_tokens must be a positive integer')
    if spec.get('python') and (not Path(spec['python']).is_file() or not os.access(spec['python'], os.X_OK)):
        raise ValueError('UnifiedReward python must be an existing executable')
    source, model = Path(spec['source']), Path(spec['model'])
    if not (source / REFERENCE_SCRIPT).is_file():
        raise ValueError('UnifiedReward source must contain the official 2.0 ACS reference script')
    if not (source / '.scd-source-revision').is_file() or (source / '.scd-source-revision').read_text().strip() != SOURCE_REVISION:
        raise ValueError('Prepare pinned UnifiedReward source with scripts/download_sources.sh unifiedreward OUT')
    receipt_path = model / '.scd-download.json'
    if not receipt_path.is_file():
        raise ValueError('Prepare the pinned UnifiedReward model with scripts/download_assets.py download unifiedreward')
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('repo_id') != MODEL_REPOSITORY or receipt.get('revision') != MODEL_REVISION
            or receipt.get('repo_type') != 'model' or receipt.get('complete') is not True):
        raise ValueError('UnifiedReward download receipt does not match the complete pinned model')
    config = json.loads((model / 'config.json').read_text())
    if config.get('model_type') != 'qwen2_5_vl' or 'Qwen2_5_VLForConditionalGeneration' not in config.get('architectures', []):
        raise ValueError('UnifiedReward 2.0 requires Qwen2.5-VL conditional-generation architecture')
    for name in ('preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json', 'chat_template.json',
                 'model.safetensors.index.json'):
        if not (model / name).is_file():
            raise ValueError(f'UnifiedReward local asset missing: {name}')
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    shards = set(index.get('weight_map', {}).values())
    if not shards:
        raise ValueError('UnifiedReward weight index is empty')
    for name in shards:
        path = model / name
        if (not path.resolve().is_relative_to(model.resolve()) or not path.is_file()
                or receipt.get('files', {}).get(name) != path.stat().st_size):
            raise ValueError(f'UnifiedReward weight shard is missing, incomplete or not in the pinned receipt: {name}')


def load_scorer(spec, device='cuda'):
    validate_spec(spec)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        spec['model'], local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': device}).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(spec['model'], local_files_only=True, use_fast=False)

    @torch.inference_mode()
    def score(images, prompts):
        if len(images) != len(prompts):
            raise ValueError('UnifiedReward needs exactly one caption for each image')
        results = []
        for image, prompt in zip(images, prompts):
            # The official 2.0 vLLM client sends JPEG quality=95. Preserve that
            # image convention while running the same model locally via HF.
            with BytesIO() as buffer:
                image.convert('RGB').save(buffer, format='JPEG', quality=95)
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    pixels = decoded.convert('RGB')
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'image': pixels},
                {'type': 'text', 'text': TEMPLATE.format(prompt=prompt)}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors='pt', truncation=False).to(model.device)
            generated = model.generate(**inputs, do_sample=False,
                                       max_new_tokens=spec.get('max_new_tokens', 4096))
            completion = generated[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(completion, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
            results.append(parse_acs(raw))
        return results
    return score
