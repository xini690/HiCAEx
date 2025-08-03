from types import SimpleNamespace

import torch
import torch.nn as nn
from util.load_checkpoint import get_pretrained_vit_model
from util.cross_attention import CrossAttention
import torch.nn.functional as F

class MultiModalViTMoE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.cfp_model = get_pretrained_vit_model(args, domain='cfp')
        self.oct_model = get_pretrained_vit_model(args, domain='oct')

        self.use_moe=args.use_moe

        self.fusion_proj = CrossAttention(dim=args.embed_dim)

        self.gating = nn.Sequential(
            nn.Linear(args.embed_dim, args.num_experts),
            nn.Softmax(dim=-1)
        )

        self.experts = nn.ModuleList([
            # embed_dim=1024   out_dim=512
            ExpertNetwork(args.embed_dim, args.embed_dim) for _ in range(args.num_experts)
        ])

        self.mlp = Mlp(in_features=args.embed_dim, hidden_features=args.embed_dim * 4, out_features=args.nb_classes, drop=0.1)

    def forward(self, cfp_img, oct_img):

        cfp_token = self.cfp_model(cfp_img)
        oct_token = self.oct_model(oct_img)
        # 提取 CLS token
        cfp_cls = cfp_token[:, 0, :]
        oct_cls = oct_token[:, 0, :]

        contrastive_loss=0
        if self.use_moe:

            cfp_attn = self.fusion_proj(cfp_token, oct_token)
            oct_attn = self.fusion_proj(oct_token, cfp_token)

            fusion = (cfp_attn + oct_attn) / 2
            fusion = fusion[:, 1:, :].mean(dim=1,keepdim=True)

            fusion_cls = fusion[:, 0, :]
            weights = self.gating(fusion_cls)

            contrastive_loss = nt_xent_loss(cfp_cls, oct_cls, temperature=0.07)

            expert_outs = torch.stack([expert(fusion_cls) for expert in self.experts], dim=1)

            weights = weights.unsqueeze(-1)
            fused_output = (expert_outs * weights).sum(dim=1)

            mean_cls = (cfp_cls + oct_cls) / 2
            fused_output=mean_cls + fused_output

            out = self.mlp(fused_output)
            return out, contrastive_loss
        else:

            fusion_cls = cfp_cls + oct_cls
            fusion_cls = F.normalize(fusion_cls, dim=-1)
            out = self.mlp(fusion_cls)

            return out, contrastive_loss

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



# 专家网络
class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class GatingNetwork(nn.Module):
    def __init__(self, in_dim, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, num_experts),
            nn.ReLU(),
            nn.Linear(in_dim, num_experts),
            nn.ReLU(),
            nn.Linear(in_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.gate(x)

def nt_xent_loss(z1, z2, temperature=0.07):

    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * B, dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask, torch.finfo(sim.dtype).min)

    positive_pairs = torch.cat([
        torch.arange(B, 2 * B),
        torch.arange(0, B)
    ]).to(z.device)

    loss = F.cross_entropy(sim, positive_pairs)
    return loss






