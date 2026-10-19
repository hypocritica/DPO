"""CPU-only checks for the supported SDXL base and its optional local VAE."""
import math
from pathlib import Path

from self_correct_dpo.common import read_json


def validate_vae_config(config):
    if config.get('_class_name')!='AutoencoderKL' or config.get('latent_channels')!=4:
        raise ValueError('SDXL VAE must be AutoencoderKL with four latent channels')
    if len(config.get('block_out_channels',[]))!=4:
        raise ValueError('SDXL VAE requires an 8x spatial downsampling factor')
    if not math.isclose(float(config.get('scaling_factor',0)),0.13025,rel_tol=0,abs_tol=1e-8):
        raise ValueError('SDXL base requires VAE scaling_factor=0.13025')


def validate_vae_override(model,*,files=True):
    vae=model.get('vae')
    if vae is None:return
    if model['family']!='sdxl' or not isinstance(vae,dict) or not vae.get('path') or not vae.get('revision'):
        raise ValueError('An SDXL VAE override requires model.vae.path and a traceable revision')
    if files:
        if any(word in vae['revision'].upper() for word in ('FILL_','TODO','UNVERIFIED')):
            raise ValueError('Replace model.vae.revision with verified source provenance')
        validate_vae_config(read_json(Path(vae['path'])/'config.json'))


def validate_sdxl_base(path):
    path=Path(path)
    index=read_json(path/'model_index.json')
    if index.get('_class_name')!='StableDiffusionXLPipeline':
        raise ValueError('SDXL base requires StableDiffusionXLPipeline; refiner pipelines are unsupported')
    for name in ('unet','vae','text_encoder','text_encoder_2','tokenizer','tokenizer_2','scheduler'):
        declaration=index.get(name)
        if not declaration or declaration[-1] is None or not (path/name).is_dir():
            raise ValueError(f'SDXL base requires a complete local {name} component')
    unet=read_json(path/'unet/config.json')
    expected={'cross_attention_dim':2048,'in_channels':4,'out_channels':4,
              'addition_embed_type':'text_time','addition_time_embed_dim':256,
              'projection_class_embeddings_input_dim':2816}
    for name,value in expected.items():
        if unet.get(name)!=value:
            raise ValueError(f'SDXL base UNet requires {name}={value!r}; refiner/inpainting are unsupported')
    for name,hidden in (('text_encoder',768),('text_encoder_2',1280)):
        config=read_json(path/name/'config.json')
        if config.get('hidden_size')!=hidden or config.get('max_position_embeddings')!=77:
            raise ValueError(f'SDXL {name} requires hidden_size={hidden} and context length 77')
        if name=='text_encoder_2' and config.get('projection_dim')!=1280:
            raise ValueError('SDXL requires the second CLIP pooled projection with 1280 dimensions')
    validate_vae_config(read_json(path/'vae/config.json'))
    scheduler=read_json(path/'scheduler/scheduler_config.json')
    if scheduler.get('prediction_type') not in ('epsilon','v_prediction'):
        raise ValueError('SDXL base requires an epsilon/v_prediction diffusion scheduler')
    if type(scheduler.get('num_train_timesteps')) is not int or scheduler['num_train_timesteps']<2:
        raise ValueError('SDXL scheduler must declare num_train_timesteps')


def same_vae_revision(first,second):
    """Model relocation may change paths, but must keep the same traced VAE."""
    a,b=first.get('vae'),second.get('vae')
    return (a is None and b is None) or (isinstance(a,dict) and isinstance(b,dict)
        and bool(a.get('revision')) and a['revision']==b.get('revision'))
