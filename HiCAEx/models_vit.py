# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn

from util.cross_attention import CrossAttention

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool=False, return_all_tokens=False,nb_classes=3, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.global_pool = global_pool
        self.return_all_tokens = return_all_tokens

        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            # del self.norm
        self.mlp=Mlp(in_features=kwargs['embed_dim'], hidden_features=kwargs['embed_dim'] * 4, out_features=nb_classes, drop=0.1)

        self.cross_attn = CrossAttention(dim=kwargs['embed_dim'], num_heads=8)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(kwargs['embed_dim'], kwargs['embed_dim']),
            nn.GELU(),
            nn.Linear(kwargs['embed_dim'], kwargs['embed_dim'])
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)


        selected_tokens = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.return_all_tokens and (i + 1) % 3 == 0:
                selected_tokens.append(x.clone())

        # 跨层融合逻辑
        if self.return_all_tokens:
            out = selected_tokens[0]
            for tok in selected_tokens[1:]:
                out = self.cross_attn(out, tok) + tok
            x = out
            if self.global_pool:
                x = self.fc_norm(x)
            # x=self.fc_norm(x)
        else:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x



def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def RETFound_dinov2(args, **kwargs):
    model = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=True,
        img_size=224,
        **kwargs
    )
    return model

if __name__ == '__main__':

    model=vit_large_patch16(
        return_all_tokens=False
    )
    x = torch.randn(2, 3, 224, 224)
    tokens = model.forward(x)
    print(tokens)
    print(len(tokens))
    print(tokens[0].shape)


