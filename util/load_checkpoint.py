import torch
from util.pos_embed import interpolate_pos_embed_vit
from timm.models.layers import trunc_normal_
import models_vit as models

def get_pretrained_vit_model(args, domain):

    model = models.__dict__['vit_large_patch16'](
        img_size=args.input_size,
        nb_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        return_all_tokens=args.return_all_tokens,
        global_pool=args.global_pool,
    )

    if domain == 'cfp':
        args.model=args.cfp_model
    elif domain == 'oct':
        args.model=args.oct_model

    if args.model and not args.eval:
        checkpoint = torch.load(args.model, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.model)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight',
                  'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed_vit(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        trunc_normal_(model.head.weight, std=2e-5)

    return model