import argparse
import ast
import datetime
import json

import numpy as np
import os
import time
from pathlib import Path

import pandas as pd
import yaml
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup

import models_vit as models
import util.lr_decay as lrd
import util.misc as misc
from ablation_Vit import SingleModalViTMoE
from multi_modal_Vit_MoE import MultiModalViTMoE
from util.datasets import build_dataset
from util.load_checkpoint import get_pretrained_vit_model
from util.multi_modal_Mixup import MultiModalMixup
from util.pos_embed import interpolate_pos_embed_vit
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from huggingface_hub import hf_hub_download, login
from engine_finetune import train_one_epoch, evaluate
from util.data_loader import EyeDiseaseDataset
import warnings
import faulthandler
from util.focal_loss import FocalLoss
from tools.retfound_encoder2_eyenet_encoder import eyenet2retfound
faulthandler.enable()
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)

    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    parser.add_argument('--scale', type=str, default='None', help='New: (0.6, 1.0), old: None')
    parser.add_argument('--vflip', type=float, default=0., help='New: 0.5, old: 0.')
    parser.add_argument('--hflip', type=float, default=0.5, help='New: 0.5, old: 0.5')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--task', default='', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--image_path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--csv_path', default='./data/', type=str,
                        help='csv path')
    parser.add_argument('--nb_classes', default=8, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # fine-tuning parameters
    parser.add_argument('--cfp_model', default='', help='finetune from checkpoint')
    parser.add_argument('--oct_model', default='', help='finetune from checkpoint')
    parser.add_argument('--label_column', default='', type=str)
    parser.add_argument('--loss_alpha', default=0.1, type=float)
    parser.add_argument('--savemodel', action='store_true', default=True,
                        help='Save model')
    parser.add_argument('--embed_dim', default=1024, type=int, help='Normalization method')
    parser.add_argument('--out_dim', default=512, type=int, help='Normalization method')
    parser.add_argument('--num_experts', default=4, type=int, help='Normalization method')
    parser.add_argument('--multi_label_threshold', default=0.5, type=float, help='Normalization method')
    parser.add_argument('--multi_label_classification', action='store_true')
    parser.add_argument('--use_moe', action='store_true')
    parser.add_argument('--base_alpha', default=0.2, type=float, help='Normalization method')
    parser.add_argument('--images_number', default=2, type=int, help='Normalization method')
    parser.add_argument('--return_all_tokens', action='store_true', help='Return tokens from multiple layers')

    parser.add_argument('--norm', default='IMAGENET', type=str, help='Normalization method')
    parser.add_argument('--enhance', action='store_true', default=False, help='Use enhanced data')
    parser.add_argument('--datasets_seed', default=2026, type=int)

    # config
    parser.add_argument('-c', '--config', default='mtf_eye.yaml', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    args_config, remaining = parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    if args.scale == 'None':
        args.scale = None
    else:
        try:
            args.scale = ast.literal_eval(args.scale)
        except (ValueError, SyntaxError):
            raise ValueError("Invalid format for --scale. Must be a tuple or 'None'.")
    return args

def main(args):

    if args.multi_label_classification:
        criterion = nn.BCEWithLogitsLoss()
    else:
        # criterion = nn.CrossEntropyLoss()
        criterion=FocalLoss(gamma=2)

    if args.resume and not args.eval:
        resume = args.resume
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Load checkpoint from: %s" % args.resume)
        args = checkpoint['args']
        args.resume = resume

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train='train', args=args)
    dataset_val = build_dataset(is_train='val', args=args)
    dataset_test = build_dataset(is_train='test', args=args)

    if args.images_number == 2:
        model = MultiModalViTMoE(args)
    elif args.images_number==1:
        # args.return_all_tokens=False
        model=SingleModalViTMoE(args)

    model.to(device)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if not args.eval:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print(
                        'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank,
                    shuffle=True)  # shuffle=True to reduce monitor bias
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))
    else:
        log_writer = None

    if not args.eval:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        print(f'len of train_set: {len(data_loader_train) * args.batch_size}')

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Total number of training set samples:{len(dataset_train)}")
    print(f"The number of batches in each epoch:{len(data_loader_train)}")

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.resume and args.eval:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Load checkpoint from: %s" % args.resume)
        model.load_state_dict(checkpoint['model'])

    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of model params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    no_weight_decay = model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, 'no_weight_decay') else []
    param_groups = lrd.param_groups_lrd(args,model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=no_weight_decay,
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        if 'epoch' in checkpoint:
            print("Test with the best model at epoch = %d" % checkpoint['epoch'])
        test_stats, auc_roc = evaluate(data_loader_test, model, device, args, epoch=0, mode='test',
                                       num_class=args.nb_classes, log_writer=log_writer)
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_score = 0.0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        val_stats, val_score = evaluate(data_loader_val, model, device, args, epoch, mode='val',
                                        num_class=args.nb_classes, log_writer=log_writer)
        if max_score < val_score:
            max_score = val_score
            best_epoch = epoch
            if args.output_dir and args.savemodel:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, mode='best')
        print("Best epoch = %d, Best score = %.4f" % (best_epoch, max_score))


        if epoch == (args.epochs - 1):
            checkpoint = torch.load(os.path.join(args.output_dir, args.task, 'checkpoint-best.pth'), map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            model.to(device)
            print("Test with the best model, epoch = %d:" % checkpoint['epoch'])
            test_stats, auc_roc = evaluate(data_loader_test, model, device, args, -1, mode='test',
                                           num_class=args.nb_classes, log_writer=None)

        if log_writer is not None:
            log_writer.add_scalar('loss/val', val_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, args.task, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


