from torch import nn

from util.load_checkpoint import get_pretrained_vit_model

class SingleModalViTMoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = get_pretrained_vit_model(args, domain=args.in_domains)
        self.mlp = Mlp(in_features=args.embed_dim, hidden_features=args.embed_dim * 4, out_features=args.nb_classes, drop=0.1)
    def forward(self, x):
        x = self.model(x)
        x = x[:, 0]
        out = self.mlp(x)
        return out


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
