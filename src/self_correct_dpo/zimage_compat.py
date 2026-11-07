# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
"""Backport mixed-dtype Z-Image padding without modifying installed Diffusers.

Based on Diffusers' ZImageTransformer2DModel._prepare_sequence (v0.37.1),
with the torch.where padding operation from v0.40.0. Keep the older sequence
and attention-mask behavior so only padding precision changes.
"""
from types import MethodType

from packaging.version import Version
import torch
from torch.nn.utils.rnn import pad_sequence


def fix_zimage_padding(transformer):
    """Install the backport on one legacy transformer, before PEFT wrapping."""
    import diffusers
    if Version(diffusers.__version__) < Version('0.40.0'):
        transformer._prepare_sequence = MethodType(_prepare_sequence, transformer)
    return transformer


def _prepare_sequence(self, feats, pos_ids, inner_pad_mask, pad_token,
                      noise_mask=None, device=None):
    item_seqlens = [len(f) for f in feats]
    max_seqlen = max(item_seqlens)
    bsz = len(feats)

    # FP32 pad parameters and BF16 autocast features promote to FP32 here.
    # Casting pad_token down would round it and change this upstream behavior.
    feats_cat = torch.cat(feats, dim=0)
    mask = torch.cat(inner_pad_mask).unsqueeze(-1)
    feats_cat = torch.where(mask, pad_token, feats_cat)
    feats = list(feats_cat.split(item_seqlens, dim=0))

    freqs_cis = list(self.rope_embedder(torch.cat(pos_ids, dim=0)).split(
        [len(p) for p in pos_ids], dim=0))
    feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
    freqs_cis = pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, :feats.shape[1]]

    attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(item_seqlens):
        attn_mask[i, :seq_len] = 1

    noise_mask_tensor = None
    if noise_mask is not None:
        noise_mask_tensor = pad_sequence(
            [torch.tensor(m, dtype=torch.long, device=device) for m in noise_mask],
            batch_first=True, padding_value=0)[:, :feats.shape[1]]
    return feats, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor
