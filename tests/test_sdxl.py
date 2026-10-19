"""Offline SDXL contracts plus real tiny Diffusers/CLIP/PEFT CPU components."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from self_correct_dpo.architecture import validate_sdxl_base, same_vae_revision
from self_correct_dpo.common import model_artifact_identity, read_json, write_json
from self_correct_dpo.config import load_config
from self_correct_dpo.formal import formal_loss, MultiViewBackend
from self_correct_dpo.independent import SingleRolloutBackend
from self_correct_dpo.joint import JointRolloutBackend
from self_correct_dpo.losses import preference_logits
from self_correct_dpo.models import PreferenceBackend, detach_condition
from self_correct_dpo.render import validate_config as validate_render, validate_adapter_backend, model_spec, load_pipeline
from self_correct_dpo.training import validate_config
from self_correct_dpo.common import sha256, write_jsonl


def make_pairs(root):
    from PIL import Image
    rows = []
    for i, split in enumerate(("train", "validation")):
        names = []
        hashes = []
        for j in range(2):
            name = f"{i}-{j}.png"
            Image.new("RGB", (4, 4), (i * 80, j * 80, 10)).save(root / name)
            names.append(name)
            hashes.append(sha256(root / name))
        caption = f"test scene {i}"
        rows.append({"pair_id":str(i), "scene_id":str(i), "split":split,
            "winner":names[0], "loser":names[1],
            "winner_sha256":hashes[0], "loser_sha256":hashes[1],
            "prompts":[{"variant_id":"canonical", "prompt":caption,
                        "template_family":"canonical", "transform_type":"canonical"}],
            "human_preference":{"source":"pickapic", "source_revision":"test-only",
                "labels":[1,0], "winner_index":0, "image_sha256":hashes,
                "original_prompt":caption}})
    write_jsonl(root / "pairs.jsonl", rows)
    return rows



def pipeline_metadata(root):
    root.mkdir()
    names=('unet','vae','text_encoder','text_encoder_2','tokenizer','tokenizer_2','scheduler')
    for name in names:(root/name).mkdir()
    write_json(root/'model_index.json',{'_class_name':'StableDiffusionXLPipeline',
        **{name:['fixture',name] for name in names}})
    write_json(root/'unet/config.json',{'cross_attention_dim':2048,'in_channels':4,'out_channels':4,
        'addition_embed_type':'text_time','addition_time_embed_dim':256,'projection_class_embeddings_input_dim':2816})
    write_json(root/'vae/config.json',{'_class_name':'AutoencoderKL','latent_channels':4,
        'block_out_channels':[128,256,512,512],'scaling_factor':.13025})
    for name,hidden in (('text_encoder',768),('text_encoder_2',1280)):
        write_json(root/name/'config.json',{'hidden_size':hidden,'projection_dim':hidden,'max_position_embeddings':77})
    write_json(root/'scheduler/scheduler_config.json',{'prediction_type':'epsilon','num_train_timesteps':1000})


class SdxlConfigTests(unittest.TestCase):
    def test_family_presets_and_existing_sd15_defaults(self):
        for arm in ('B','C'):
            c=read_json(ROOT/f'configs/sdxl/model-K2-{arm}.json')
            validate_config(c,files=False)
            self.assertEqual(c['model']['family'],'sdxl')
            self.assertEqual(c['training']['formal_arm'],arm)
            self.assertEqual(c['training']['resolution'],1024)
            self.assertEqual(c['training']['batch_size']*c['training']['gradient_accumulation_steps'],128)
            self.assertEqual(c['training']['beta'],500)
            self.assertEqual(c['training']['warmup_steps'],0)
            self.assertEqual(c['evaluation']['sampling']['scheduler'],'EulerDiscreteScheduler')
            self.assertEqual(c['evaluation']['first_step'],200)
            self.assertEqual(c['evaluation']['every'],500)
            self.assertEqual(c['training']['lora_checkpoint_every'],1000)
            c['training']['formal_arm']='D'
            with self.assertRaisesRegex(ValueError,'only B/C'):validate_config(c,files=False)
        c=read_json(ROOT/'configs/model-K2-C.json')
        self.assertEqual((c['model']['family'],c['training']['resolution'],c['training']['beta']),('sd15',512,5000))
        self.assertEqual(c['training']['batch_size']*c['training']['gradient_accumulation_steps'],256)

    def test_complete_sdxl_base_preflight_and_wrong_architectures_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'base';pipeline_metadata(root)
            c=read_json(ROOT/'configs/sdxl/model-K2-C.json')
            c['model'].update(path=str(root),revision='fixture-sdxl-base')
            validate_config(c)
            render={'model':c['model'],'sampling':read_json(ROOT/'configs/evaluate-sdxl.json')['sampling']}
            validate_render(render,'base')
            render['sampling']['scheduler']='DDIMScheduler';validate_render(render,'base')
            render['sampling']['scheduler']='PNDMScheduler'
            with self.assertRaisesRegex(ValueError,'scheduler'):validate_render(render,'base')
            path=root/'unet/config.json';original=read_json(path)
            for key,value in (('in_channels',9),('cross_attention_dim',1280),('projection_class_embeddings_input_dim',2560)):
                path.write_text(json.dumps({**original,key:value}))
                with self.assertRaisesRegex(ValueError,'SDXL base UNet'):validate_config(c)
            path.write_text(json.dumps(original))
            (root/'text_encoder_2').rename(root/'missing-text_encoder_2')
            with self.assertRaisesRegex(ValueError,'text_encoder_2'):validate_sdxl_base(root)

    def test_local_vae_paths_fingerprints_and_adapter_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);pipeline_metadata(r/'base');(r/'custom-vae').mkdir()
            (r/'custom-vae/config.json').write_bytes((r/'base/vae/config.json').read_bytes())
            model={'family':'sdxl','path':'base','revision':'base-revision',
                   'vae':{'path':'custom-vae','revision':'vae-revision'}}
            write_json(r/'config.json',{'model':model})
            resolved=load_config(r/'config.json')['model']
            self.assertEqual(resolved['vae']['path'],str(r/'custom-vae'))
            self.assertEqual(model_spec({'model':resolved})['vae'],resolved['vae'])
            before=model_artifact_identity(resolved)
            self.assertIn('vae',before)
            path=r/'custom-vae/config.json';config=read_json(path);path.write_text(json.dumps({**config,'fixture':'changed'}))
            self.assertNotEqual(before,model_artifact_identity(resolved))
            (r/'adapter').mkdir()
            write_json(r/'adapter/backend.json',{'family':'sdxl','component':'unet','model':resolved,
                'error_parameterization':'diffusion_epsilon_mse'})
            validate_adapter_backend(r/'adapter',resolved,{},'unet')
            moved=copy.deepcopy(resolved);moved['vae']['path']='/another-server/vae'
            self.assertTrue(same_vae_revision(resolved,moved))
            validate_adapter_backend(r/'adapter',moved,{},'unet')
            moved['vae']['revision']='wrong-vae'
            with self.assertRaisesRegex(ValueError,'VAE revision'):validate_adapter_backend(r/'adapter',moved,{},'unet')
            with self.assertRaisesRegex(ValueError,'VAE revision'):
                validate_adapter_backend(r/'adapter',{k:v for k,v in resolved.items() if k!='vae'},{},'unet')

    def test_sdxl_config_helper_and_portable_training_check_only(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);pipeline_metadata(r/'base');(r/'data').mkdir();make_pairs(r/'data')
            env={k:v for k,v in os.environ.items() if k not in
                 ('MODEL_PATH','PAIRS_PATH','MONITOR_PROMPTS','SCORERS_CONFIG','SD15_MODEL','SDXL_MODEL')}
            result=subprocess.run([sys.executable,'-I',str(ROOT/'scripts/make_config.py'),
                '--family','sdxl','--model',str(r/'base'),'--model-revision','fixture-sdxl-base',
                '--pairs',str(r/'data/pairs.jsonl'),'--monitor-prompts',str(r/'monitor.jsonl'),
                '--out',str(r/'normal.json')],env=env,cwd=r,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            normal=load_config(r/'normal.json')
            self.assertEqual(normal['model']['path'],str(r/'base'))
            self.assertEqual(normal['evaluation']['scorers_config'],str(ROOT/'configs/scorers.json'))
            command=[sys.executable,'-I',str(ROOT/'scripts/make_config.py'),'--family','sdxl','--model',str(r/'base'),
                '--model-revision','fixture-sdxl-base','--pairs',str(r/'data/pairs.jsonl'),'--smoke','--out',str(r/'config.json')]
            result=subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            config=read_json(r/'config.json')
            self.assertEqual(config['training']['resolution'],1024)
            self.assertEqual(config['training']['max_steps'],2)
            self.assertEqual(config['evaluation'],{'enabled':False})
            result=subprocess.run([sys.executable,'-I',str(ROOT/'train.py'),'--config',str(r/'config.json'),
                '--out',str(r/'run'),'--check-only'],cwd=r,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('"family": "sdxl"',result.stdout)
            self.assertFalse((r/'run').exists())


def tiny_sdxl(root):
    import torch
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from peft import LoraConfig, get_peft_model
    from self_correct_dpo.independent import RolloutConfig
    torch.manual_seed(82)
    write_json(root/'vocab.json',{'<|startoftext|>':0,'<|endoftext|>':1,'a</w>':2,'b</w>':3})
    (root/'merges.txt').write_text('#version: 0.2\n')
    tokenizer=CLIPTokenizer(str(root/'vocab.json'),str(root/'merges.txt'),model_max_length=8)
    def config(hidden):
        return CLIPTextConfig(vocab_size=4,hidden_size=hidden,intermediate_size=hidden*2,num_hidden_layers=2,
            num_attention_heads=2,max_position_embeddings=8,projection_dim=hidden,bos_token_id=0,eos_token_id=1,pad_token_id=1)
    unet=UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,block_out_channels=(16,),layers_per_block=1,
        down_block_types=('CrossAttnDownBlock2D',),up_block_types=('CrossAttnUpBlock2D',),
        cross_attention_dim=24,attention_head_dim=4,norm_num_groups=4,addition_embed_type='text_time',
        addition_time_embed_dim=4,projection_class_embeddings_input_dim=40)
    unet.register_to_config(_name_or_path=str(root/'unet'))
    unet.save_config(root/'unet')
    original=copy.deepcopy(unet)
    vae=AutoencoderKL(block_out_channels=(8,16),down_block_types=('DownEncoderBlock2D',)*2,
        up_block_types=('UpDecoderBlock2D',)*2,latent_channels=4,norm_num_groups=4,scaling_factor=.13025)
    pipe=StableDiffusionXLPipeline(vae=vae,unet=unet,text_encoder=CLIPTextModel(config(8)),
        text_encoder_2=CLIPTextModelWithProjection(config(16)),tokenizer=tokenizer,tokenizer_2=tokenizer,
        scheduler=DDPMScheduler(num_train_timesteps=1000,beta_schedule='scaled_linear'),add_watermarker=False)
    b=object.__new__(JointRolloutBackend)
    b.family='sdxl';b.resolution=16;b.device=torch.device('cpu');b.dtype=torch.float32
    b.noise_views=2;b.training_config={'noise_views':2,'vae_batch_size':1}
    b.model_config={'family':'sdxl','path':str(root),'revision':'tiny-test-fixture'}
    b.pipe=pipe;b.scheduler=pipe.scheduler;b.rollout_config=RolloutConfig()
    b.component_name='unet';b.error_parameterization='diffusion_epsilon_mse'
    b.frozen_modules=[pipe.vae,pipe.text_encoder,pipe.text_encoder_2]
    for module in b.frozen_modules:module.requires_grad_(False).eval()
    unet.requires_grad_(False)
    b.model=get_peft_model(unet,LoraConfig(r=2,lora_alpha=2,lora_dropout=0.,target_modules=['to_q','to_k','to_v','to_out.0']))
    pipe.unet=b.model
    b.train()
    return b,original


class TinySdxlTests(unittest.TestCase):
    def test_real_dual_clip_conditions_rollout_and_lora_roundtrip(self):
        import torch
        from peft import PeftModel
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}):
            root=Path(d);b,original=tiny_sdxl(root)
            condition=b._condition(['a','a'])
            self.assertEqual(tuple(condition['encoder_hidden_states'].shape),(2,8,24))
            self.assertEqual(tuple(condition['added_cond_kwargs']['text_embeds'].shape),(2,16))
            torch.testing.assert_close(condition['added_cond_kwargs']['time_ids'],torch.tensor([[16.,16.,0.,0.,16.,16.]]*2))
            image=torch.linspace(-.8,.8,2*3*16*16).reshape(1,2,3,16,16)
            ordinary=object.__new__(MultiViewBackend);ordinary.__dict__.update(b.__dict__)
            normal=ordinary.prepare_views(image,[['a','a']],torch.Generator().manual_seed(16))
            for view in normal['views']:
                torch.testing.assert_close(view['target'][::2],view['target'][1::2])
                torch.testing.assert_close(view['times'][::2],view['times'][1::2])
            self.assertEqual(tuple(ordinary.view_errors(normal,0).shape),(1,1,2))
            prepared=b.prepare_views(image,[['a','a']],torch.Generator().manual_seed(17))
            for value in prepared['condition']['added_cond_kwargs'].values():self.assertFalse(value.requires_grad)
            policy=torch.cat([b.view_errors(prepared,j) for j in range(2)],1)
            refs=torch.cat([b.view_errors(prepared,j,reference=True) for j in range(2)],1)
            torch.testing.assert_close(policy,refs)
            loss,_=formal_loss({'policy_errors':policy,'reference_errors':refs},'C',500.)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.trainable_parameters()))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in b.trainable_parameters()),0)
            self.assertTrue(all(p.grad is None for m in b.frozen_modules for p in m.parameters()))
            torch.optim.SGD(b.trainable_parameters(),lr=.001).step()
            b.eval()
            b.pipe.vae.save_pretrained(root/'custom-vae')
            b.pipe.unet=copy.deepcopy(original)
            b.pipe.save_pretrained(root/'base')
            b.pipe.unet=b.model
            b.model_config.update(path=str(root/'base'),vae={'path':str(root/'custom-vae'),'revision':'tiny-vae-fixture'})
            b.save_adapter(root/'adapter')
            v=prepared['views'][0]
            with torch.no_grad():
                expected=b._predict(v['noisy'],v['times'],prepared['condition'])
                restored=PeftModel.from_pretrained(original,str(root/'adapter'),is_trainable=False,local_files_only=True).eval()
                actual=restored(v['noisy'],v['times'],**prepared['condition'],return_dict=False)[0]
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
            with torch.no_grad():
                for p in b.trainable_parameters():p.add_(.1)
            b.load_adapter(root/'adapter')
            with torch.no_grad():torch.testing.assert_close(b._predict(v['noisy'],v['times'],prepared['condition']),expected,rtol=0,atol=0)
            config={'model':b.model_config,'sampling':{'scheduler':'EulerDiscreteScheduler','dtype':'float32'},
                    'aligned':{'kind':'peft','path':str(root/'adapter')}}
            loaded=load_pipeline(config,'aligned',torch,device='cpu')
            self.assertIsInstance(loaded.unet,PeftModel)
            self.assertEqual(loaded.vae.dtype,torch.float32)
            with torch.inference_mode():
                rendered=loaded(prompt='a',negative_prompt='',height=16,width=16,num_inference_steps=2,
                    guidance_scale=5.,generator=torch.Generator().manual_seed(17)).images[0]
            self.assertEqual(rendered.size,(16,16))

    def test_nested_detach_single_and_joint_k1_exactly_match(self):
        import torch
        from test_joint import backend
        b=backend(k=1);b.family='sdxl'
        b._condition=lambda texts:{'encoder_hidden_states':torch.ones(len(texts),2,requires_grad=True),
            'added_cond_kwargs':{'text_embeds':torch.ones(len(texts),3,requires_grad=True),
                                 'time_ids':torch.ones(len(texts),6,requires_grad=True)}}
        image=torch.zeros(2,2,3,64,64)
        single=SingleRolloutBackend.prepare_rollout(b,image,[['a']]*2,torch.Generator().manual_seed(2))
        joint=b.prepare_views(image,[['a']]*2,torch.Generator().manual_seed(2))['views'][0]
        for key in ('noisy','target','times','epsilon','x0'):torch.testing.assert_close(single[key],joint[key],rtol=0,atol=0)
        for tensor in single['condition']['added_cond_kwargs'].values():self.assertFalse(tensor.requires_grad)


if __name__=='__main__':unittest.main()
