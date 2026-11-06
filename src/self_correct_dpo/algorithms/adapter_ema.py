"""EMA in LoRA factor space, sharing the frozen base without policy mutation.

This is explicitly adapter-factor EMA, not EMA of the merged dense BA update.
functional_call preserves the policy Parameter objects and their version counters.
"""
import math
import torch


class AdapterEMA:
    def __init__(self, model, decay=0.9999):
        if isinstance(decay, bool) or not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError('EMA decay must be finite in [0,1)')
        self.decay, self.updates = float(decay), 0
        parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if not parameters or any('lora_' not in name for name in parameters):
            raise ValueError('AdapterEMA requires exclusively trainable LoRA parameters')
        self.shadow = {name: p.detach().float().clone() for name, p in parameters.items()}
        self._validate(model)

    def _validate(self, model):
        current = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if set(current) != set(self.shadow):
            raise ValueError('EMA adapter parameter names changed')
        for name, value in self.shadow.items():
            if (value.shape != current[name].shape or value.dtype != torch.float32
                    or value.requires_grad or not torch.isfinite(value).all()):
                raise ValueError(f'Invalid EMA tensor: {name}')
        return current

    @torch.no_grad()
    def update(self, model):
        # Shapes/dtypes are established at initialization/load; avoid one CUDA
        # synchronization per LoRA tensor on every optimizer update.
        parameters = dict(model.named_parameters())
        for name, shadow in self.shadow.items():
            shadow.lerp_(parameters[name].detach().float(), 1 - self.decay)
        self.updates += 1

    def state_dict(self):
        return {'kind': 'lora_factor_ema_v1', 'decay': self.decay, 'updates': self.updates,
                'shadow': {n: v.detach().cpu().clone() for n, v in self.shadow.items()}}

    def load_state_dict(self, state, model):
        if (not isinstance(state, dict) or state.get('kind') != 'lora_factor_ema_v1'
                or state.get('decay') != self.decay or type(state.get('updates')) is not int
                or state['updates'] < 0 or set(state.get('shadow', {})) != set(self.shadow)):
            raise ValueError('Missing or incompatible adapter EMA state')
        shadow = state['shadow']
        for name, value in shadow.items():
            if value.dtype != torch.float32 or value.shape != self.shadow[name].shape or not torch.isfinite(value).all():
                raise ValueError(f'Invalid saved EMA tensor: {name}')
        self.shadow = {n: v.detach().to(self.shadow[n].device).clone() for n, v in shadow.items()}
        self.updates = state['updates']
        self._validate(model)

    def forward(self, model, args, kwargs):
        # Unwrapped PEFT model: DDP only owns policy forwards/backwards.
        with torch.no_grad():
            return torch.func.functional_call(model, self.shadow, args, kwargs, strict=False)
