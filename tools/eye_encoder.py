import argparse
import math

import torch
from einops import rearrange
import copy
import torch.nn.functional as F

def hiCAEx_dict(encoder, domain='rgb', pretrain_weights=True):
    assert isinstance(domain, str), "domain must be a string"
    state_dict = {}
    for k,v in encoder.items():
        if k == 'global_tokens':
            state_dict['cls_token'] = v
        elif k == f'input_adapters.{domain}.pos_emb':
            state_dict['pos_embed'] = rearrange(v, 'b d h w -> b (h w) d')
            state_dict['pos_embed'] = F.pad(state_dict['pos_embed'], (0,0,1,0,0,0), mode='constant', value=0.0)
        elif k == f'input_adapters.{domain}.proj.weight':
            state_dict['patch_embed.proj.weight'] = v
        elif k == f'input_adapters.{domain}.proj.bias':
            state_dict['patch_embed.proj.bias'] = v
        elif 'encoder' in k:
            state_dict[k.replace('encoder', 'blocks')] = v

    if not pretrain_weights:
        state_dict['head.weight'] = encoder['output_adapters.cls.head.weight']
        state_dict['head.bias'] = encoder['output_adapters.cls.head.bias']
        state_dict['fc_norm.weight'] = encoder['output_adapters.cls.norm.weight']
        state_dict['fc_norm.bias'] = encoder['output_adapters.cls.norm.bias']

    return state_dict



