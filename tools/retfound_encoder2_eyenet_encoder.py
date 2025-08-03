import argparse
import math

import torch
from einops import rearrange
import copy
import torch.nn.functional as F


def retfound2eyenet(encoder, domain, pretrain_weights=True):
    """
    Converts timm ViT weights to MultiMAE weights.
    """
    assert isinstance(domain, list), "domain must be a list of strings"
    state_dict = {}
    state_dict['global_tokens'] = encoder['cls_token']
    for k,v in encoder.items():
        if k == 'pos_embed':
            n = int(math.sqrt(v.shape[1]))
            pos_embed = rearrange(v[:, 1:], 'b (n1 n2) d -> b d n1 n2', n1=n, n2=n)
            # state_dict['global_tokens'] += v[:, 0]
            # state_dict['pos_embed_global_tokens'] = v
            for dom in domain:
                state_dict[f'input_adapters.{dom}.pos_emb'] = pos_embed
        elif k == 'patch_embed.proj.weight':
            for dom in domain:
                state_dict[f'input_adapters.{dom}.proj.weight'] = v
        elif k == 'patch_embed.proj.bias':
            for dom in domain:
                state_dict[f'input_adapters.{dom}.proj.bias'] = v
        elif 'blocks.' in k and 'decoder_blocks.' not in k:
            state_dict[k.replace('blocks.', 'encoder.')] = v

    if not pretrain_weights:
        state_dict['output_adapters.cls.head.weight'] = encoder['head.weight']
        state_dict['output_adapters.cls.head.bias'] = encoder['head.bias']
        state_dict['output_adapters.cls.norm.weight'] = encoder['fc_norm.weight']
        state_dict['output_adapters.cls.norm.bias'] = encoder['fc_norm.bias']

    return state_dict


def eyenet2retfound(encoder, domain='rgb', pretrain_weights=True):
    """
    Converts MultiMAE weights to timm ViT weights.
    Assumes that there is only 1 global token in the MultiMAE.
    """
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


def compare_dicts(dict1, dict2):

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    different_keys = keys1.symmetric_difference(keys2)

    common_keys = keys1.intersection(keys2)
    unequal_value_keys = []

    for key in common_keys:
        if not torch.equal(dict1[key], dict2[key]):
            unequal_value_keys.append(key)

    return different_keys, unequal_value_keys



def convert_test():
    retfound_encoder = 'checkpoint-best.pth'
    print(f'Loading weights at {retfound_encoder}')
    ckpt_retfound_encoder = torch.load(retfound_encoder, map_location='cpu')

    # print('Converting from ViT weights to MultiMAE weights...')
    # ckpt_eyenet_encoder = dict()
    # ckpt_eyenet_encoder['model'] = retfound2eyenet(ckpt_retfound_encoder['model'])

    # ckpt_retfound_encoder2 = dict()
    # ckpt_retfound_encoder2['model'] = eyenet2retfound(ckpt_eyenet_encoder['model'])
    ckpt_retfound_encoder2 = torch.load('only_test_eyenet.pth', map_location='cpu')
    ckpt_retfound_encoder2['model'] = eyenet2retfound(ckpt_retfound_encoder2['model'])
    different_keys, unequal_value_keys = compare_dicts(ckpt_retfound_encoder2['model'], ckpt_retfound_encoder['model'])




if __name__ == '__main__':
    ckpt_retfound_encoder = torch.load(
        'checkpoint-299.pth',
        map_location='cpu')
    del ckpt_retfound_encoder['optimizer']
    del ckpt_retfound_encoder['epoch']
    del ckpt_retfound_encoder['scaler']
    del ckpt_retfound_encoder['args']
    del ckpt_retfound_encoder['loss_balancer']

    # print('Converting from ViT weights to MultiMAE weights...')
    ckpt_retfound_encoder['model'] = eyenet2retfound(ckpt_retfound_encoder['model'], domain='oct', pretrain_weights=True)
    torch.save(ckpt_retfound_encoder, 'checkpoint-oct.pth')
    # print(f'Saved converted weights at {eyenet_encoder}')
