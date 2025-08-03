# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math

import torch
from einops import rearrange
import copy


def vit_imagenet_to_eyenet(multimae_state_dict):
    """
    Converts timm ViT weights to MultiMAE weights.
    """
    state_dict = {}
    state_dict['global_tokens'] = multimae_state_dict['cls_token']
    for k,v in multimae_state_dict.items():
        if k == 'pos_embed':
            n = int(math.sqrt(v.shape[1]))
            pos_embed = rearrange(v[:,1:], 'b (n1 n2) d -> b d n1 n2', n1=n, n2=n)
            state_dict['global_tokens'] += v[:,0]
            state_dict['input_adapters.rgb.pos_emb'] = pos_embed
        elif k == 'patch_embed.proj.weight':
            state_dict['input_adapters.rgb.proj.weight'] = v
        elif k == 'patch_embed.proj.bias':
            state_dict['input_adapters.rgb.proj.bias'] = v
        elif 'blocks.' in k and 'decoder_blocks.' not in k:
            state_dict[k.replace('blocks.', 'encoder.')] = v

    state_dict['input_adapters.uwf.pos_emb'] = copy.deepcopy(state_dict['input_adapters.rgb.pos_emb'])
    state_dict['input_adapters.oct.pos_emb'] = copy.deepcopy(state_dict['input_adapters.rgb.pos_emb'])
    state_dict['input_adapters.eyephoto.pos_emb'] = copy.deepcopy(state_dict['input_adapters.rgb.pos_emb'])

    state_dict['input_adapters.uwf.proj.weight'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.weight'])
    state_dict['input_adapters.oct.proj.weight'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.weight'])
    state_dict['input_adapters.eyephoto.proj.weight'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.weight'])

    state_dict['input_adapters.uwf.proj.bias'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.bias'])
    state_dict['input_adapters.oct.proj.bias'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.bias'])
    state_dict['input_adapters.eyephoto.proj.bias'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.bias'])

    return state_dict


def vit_to_multimae(multimae_state_dict, ckpt_multimae_multivit):
    """
    Converts timm ViT weights to MultiMAE weights.
    """
    state_dict = {}
    state_dict['global_tokens'] = multimae_state_dict['cls_token']
    for k,v in multimae_state_dict.items():
        if k == 'pos_embed':
            n = int(math.sqrt(v.shape[1]))
            pos_embed = rearrange(v[:,1:], 'b (n1 n2) d -> b d n1 n2', n1=n, n2=n)
            state_dict['global_tokens'] += v[:,0]
            state_dict['input_adapters.rgb.pos_emb'] = pos_embed
        elif k == 'patch_embed.proj.weight':
            state_dict['input_adapters.rgb.proj.weight'] = v
        elif k == 'patch_embed.proj.bias':
            state_dict['input_adapters.rgb.proj.bias'] = v
        elif 'blocks.' in k and 'decoder_blocks.' not in k:
            state_dict[k.replace('blocks.', 'encoder.')] = v

    # state_dict['input_adapters.uwf.pos_emb'] = copy.deepcopy(state_dict['input_adapters.rgb.pos_emb'])
    # state_dict['input_adapters.oct.pos_emb'] = copy.deepcopy(state_dict['input_adapters.rgb.pos_emb'])
    # state_dict['input_adapters.eyephoto.pos_emb'] = copy.deepcopy(state_dict['input_adapters.rgb.pos_emb'])

    state_dict['input_adapters.uwf.proj.weight'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.weight'])
    state_dict['input_adapters.oct.proj.weight'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.weight'])
    state_dict['input_adapters.eyephoto.proj.weight'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.weight'])

    state_dict['input_adapters.uwf.proj.bias'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.bias'])
    state_dict['input_adapters.oct.proj.bias'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.bias'])
    state_dict['input_adapters.eyephoto.proj.bias'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.bias'])

    for kk, vv in ckpt_multimae_multivit.items():
        if 'pos_emb' in kk or 'mask_token' in kk:
            continue
        if 'output_adapters.rgb.' in kk and 'proj_context.weight' not in kk:
            if 'task_embeddings' not in kk:
                state_dict[kk] = vv
                state_dict[kk.replace('output_adapters.rgb.', 'output_adapters.norm_rgb.')] = copy.deepcopy(vv)
                state_dict[kk.replace('output_adapters.rgb.', 'output_adapters.uwf.')] = copy.deepcopy(vv)
                state_dict[kk.replace('output_adapters.rgb.', 'output_adapters.oct.')] = copy.deepcopy(vv)
                state_dict[kk.replace('output_adapters.rgb.', 'output_adapters.eyephoto.')] = copy.deepcopy(vv)
                state_dict[kk.replace('output_adapters.rgb.', 'output_adapters.norm_uwf.')] = copy.deepcopy(vv)
                state_dict[kk.replace('output_adapters.rgb.', 'output_adapters.norm_oct.')] = copy.deepcopy(vv)
                state_dict[kk.replace('output_adapters.rgb.', 'output_adapters.norm_eyephoto.')] = copy.deepcopy(vv)
            elif 'depth' not in kk and 'semseg' not in kk:
                for modal in ['norm_rgb', 'norm_oct', 'norm_uwf', 'norm_eyephoto','rgb', 'oct', 'uwf', 'eyephoto']:
                    for key in ['rgb', 'oct', 'uwf', 'eyephoto']:
                        state_dict[kk.replace('output_adapters.rgb.task_embeddings.rgb',
                                              f'output_adapters.{modal}.task_embeddings.{key}')] = copy.deepcopy(vv)

    return state_dict



def vit_to_multimae_decoder_depth_4(multimae_state_dict):
    """
    Converts timm ViT weights to MultiMAE weights.
    the decoder of eyenet is decoder_dim: 512, decoder_depth: 4, decoder_num_heads: 16
    """
    state_dict = {}
    state_dict['global_tokens'] = multimae_state_dict['cls_token']
    for k,v in multimae_state_dict.items():
        if k == 'pos_embed':
            n = int(math.sqrt(v.shape[1]))
            pos_embed = rearrange(v[:,1:], 'b (n1 n2) d -> b d n1 n2', n1=n, n2=n)
            state_dict['global_tokens'] += v[:,0]
            state_dict['input_adapters.rgb.pos_emb'] = pos_embed
        elif k == 'patch_embed.proj.weight':
            state_dict['input_adapters.rgb.proj.weight'] = v
        elif k == 'patch_embed.proj.bias':
            state_dict['input_adapters.rgb.proj.bias'] = v
        elif 'blocks.' in k and 'decoder_blocks.' not in k:
            state_dict[k.replace('blocks.', 'encoder.')] = v

    # state_dict['input_adapters.uwf.pos_emb'] = copy.deepcopy(state_dict['input_adapters.rgb.pos_emb'])
    # state_dict['input_adapters.oct.pos_emb'] = copy.deepcopy(state_dict['input_adapters.rgb.pos_emb'])
    # state_dict['input_adapters.eyephoto.pos_emb'] = copy.deepcopy(state_dict['input_adapters.rgb.pos_emb'])

    state_dict['input_adapters.uwf.proj.weight'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.weight'])
    state_dict['input_adapters.oct.proj.weight'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.weight'])
    state_dict['input_adapters.eyephoto.proj.weight'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.weight'])

    state_dict['input_adapters.uwf.proj.bias'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.bias'])
    state_dict['input_adapters.oct.proj.bias'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.bias'])
    state_dict['input_adapters.eyephoto.proj.bias'] = copy.deepcopy(state_dict['input_adapters.rgb.proj.bias'])

    # 'output_adapters.rgb.mask_token' (1, 1, 512)
    # 'output_adapters.rgb.pos_emb' (1, 512, 14, 14) | 'decoder_pos_embed' (1, 197, 512)
    # 'output_adapters.rgb.task_embeddings.eyephoto'(1, 1, 512)
    # 'output_adapters.rgb.task_embeddings.oct'(1, 1, 512)
    # 'output_adapters.rgb.task_embeddings.rgb'(1, 1, 512)
    # 'output_adapters.rgb.task_embeddings.uwf'(1, 1, 512)
    # 'output_adapters.rgb.decoder_transformer.0.norm1.weight' (512)
    # 'output_adapters.rgb.out_proj.weight' (768, 512) | 'decoder_pred.weight'
    # 'output_adapters.rgb.proj_context.weight' (512, 1024) | 'decoder_embed.weight'

    for domain in ['rgb', 'uwf', 'eyephoto', 'oct']:
        for k, v in multimae_state_dict.items():
            if 'decoder_blocks' in k:
                # if '4' in k or '5' in k or '6' in k or '7' in k:
                #     continue
                state_dict[k.replace(f'decoder_blocks', f'output_adapters.{domain}.decoder_transformer')] = copy.deepcopy(v)
            elif 'decoder_embed' in k:
                state_dict[k.replace('decoder_embed', f'output_adapters.{domain}.proj_context')] = copy.deepcopy(v)
            elif "decoder_pred" in k:
                state_dict[k.replace('decoder_pred', f'output_adapters.{domain}.out_proj')] = copy.deepcopy(v)
            # elif k == 'decoder_pos_embed':
                # state_dict[f'output_adapters.{domain}.mask_token'] = copy.deepcopy(multimae_state_dict['mask_token'])
                # n = int(math.sqrt(v.shape[1]))
                # pos_embed = rearrange(v[:, 1:], 'b (n1 n2) d -> b d n1 n2', n1=n, n2=n)
                # state_dict[f'output_adapters.{domain}.mask_token'] += v[:, 0]
                # state_dict[f'output_adapters.{domain}.pos_emb'] = pos_embed


    return state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="ViT to MultiMAE checkpoint converter")
    parser.add_argument(
        "--vit_ckpt_path", default='RETFound_oct_weights.pth', type=str,
        help="Path to converted ViT(MultiMAE) checkpoint"
    )
    parser.add_argument(
        "--multimae_ckpt_path", default='RETFound_oct_and_MultiMAE_to_EyeNet_no_proj_context_layer_no_pos_emb_mask_token.pth', type=str,
        help="Path to MultiMAE checkpoint"
    )
    args = parser.parse_args()

    print(f'Loading weights at {args.vit_ckpt_path}')
    ckpt = torch.load(args.vit_ckpt_path, map_location='cpu')
    ckpt_multimae_multivit = torch.load('multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth', map_location='cpu')

    del ckpt['optimizer']
    del ckpt['epoch']
    del ckpt['scaler']
    del ckpt['args']
    print('Converting from ViT weights to MultiMAE weights...')
    ckpt['model'] = vit_to_multimae(ckpt['model'], ckpt_multimae_multivit['model'])
    torch.save(ckpt, args.multimae_ckpt_path)
    print(f'Saved converted weights at {args.multimae_ckpt_path}')

