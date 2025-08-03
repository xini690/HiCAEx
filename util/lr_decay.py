# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------
import json

import numpy as np


def param_groups_lrd(args,model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    if args.images_number==1:
        if hasattr(model, 'blocks'):
            num_layers = len(model.blocks) + 1
        else:
            # use the number of layers in the ResNet model as a default value
            # num_layers = len(model.layer1) + len(model.layer2) + len(model.layer3) + len(model.layer4) + 1
            num_vit_blocks = len(model.model.blocks)
            num_layers = 1 + num_vit_blocks + 1

        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # no decay: all 1D parameters and model specific ones
            if p.ndim == 1 or n in no_weight_decay_list:
                g_decay = "no_decay"
                this_decay = 0.
            else:
                g_decay = "decay"
                this_decay = weight_decay

            layer_id = get_layer_id_for_vit(args,n, num_layers)
            group_name = "layer_%d_%s" % (layer_id, g_decay)

            if group_name not in param_group_names:
                this_scale = layer_scales[layer_id]

                param_group_names[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }
                param_groups[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }

            param_group_names[group_name]["params"].append(n)
            param_groups[group_name]["params"].append(p)

        # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

        return list(param_groups.values())
    elif args.images_number==2 :
        num_vit_blocks = len(model.cfp_model.blocks)
        num_layers = 1 + num_vit_blocks + 1

        layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # 判断是否使用 weight decay
            if p.ndim == 1 or n in no_weight_decay_list:
                g_decay = "no_decay"
                this_decay = 0.
            else:
                g_decay = "decay"
                this_decay = weight_decay

            # 获取参数所在层编号
            layer_id = get_layer_id_for_vit(args,n, num_layers)
            group_name = f"layer_{layer_id}_{g_decay}"

            if group_name not in param_group_names:
                this_scale = layer_scales[layer_id]
                param_group_names[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }
                param_groups[group_name] = {
                    "lr_scale": this_scale,
                    "weight_decay": this_decay,
                    "params": [],
                }

            param_group_names[group_name]["params"].append(n)
            param_groups[group_name]["params"].append(p)

        return list(param_groups.values())

def get_layer_id_for_vit(args,name, num_layers):
    if args.images_number==2:
        if name.startswith("cfp_model.patch_embed") or name.startswith("oct_model.patch_embed"):
            return 0
        if name.startswith("cfp_model.cls_token") or name.startswith("cfp_model.pos_embed"):
            return 0
        if name.startswith("oct_model.cls_token") or name.startswith("oct_model.pos_embed"):
            return 0

        if name.startswith("cfp_model.blocks") or name.startswith("oct_model.blocks"):
            block_id = int(name.split('.')[2])
            return 1 + block_id

        return num_layers
    elif args.images_number==1:
        if name.startswith("model.patch_embed"):
            return 0
        if name.startswith("model.cls_token"):
            return 0
        if name.startswith("model.cls_token"):
            return 0
        if name.startswith("model.blocks"):
            block_id = int(name.split('.')[2])
            return 1 + block_id

        return num_layers