# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter  # 기존 TensorBoard 로깅
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.datasets import ImageNetSubset
import timm.optim.optim_factory as optim_factory
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_pretrain import train_one_epoch
import models_ltrp
from PIL import ImageFile
from collections import OrderedDict
import wandb  # WandB 임포트 추가

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='ltrp_base_and_vs', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.9, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.06,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=3.0e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # ------------ importances
    parser.add_argument('--burning_in', default=0, type=int)
    parser.add_argument('--mask_all', action='store_true')
    parser.add_argument('--ltr_loss', default="list_mle", type=str,
                        help='')
    parser.add_argument('--list_mle_k', default=None, type=int,
                        help='')
    parser.add_argument('--rank_net_t', default=0.001, type=float,
                        help='')
    parser.add_argument('--rank_net_sigma', default=1, type=float,
                        help='')
    parser.add_argument('--focused_rank_k', default=10, type=int,
                        help='')
    parser.add_argument('--resume_from_mae', default='',
                        help='resume from checkpoint')
    parser.add_argument('--score_net', default='vit_small', type=str,
                        help='')
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--pretrained_from', default='')
    parser.add_argument('--asymmetric', action='store_true')
    parser.add_argument('--img_metric', default='', type=str)
    parser.add_argument('--resume_score_net', default='',
                        help='resume from checkpoint')
    parser.add_argument('--low_shot', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--score_net_depth', default=12, type=int, help='resume from checkpoint')
    return parser

def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # main process인 경우에만 WandB 초기화
    if misc.is_main_process():
        wandb.init(project="mae_pretraining", config=vars(args))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    if args.low_shot != '':
        print("pre-training with low shot: ", args.low_shot)
        dataset_train = ImageNetSubset(dataset_train, args.low_shot)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # 기존 TensorBoard 로그 설정 (WandB와 병행 가능)
    if misc.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = models_ltrp.__dict__[args.model](args)
    print("Model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    if total_params == 0:
        raise ValueError("No trainable parameters found in the model!")

    if args.pretrained_from:
        checkpoint = torch.load(args.pretrained_from, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        msg = model.score_net.load_state_dict(checkpoint_model, strict=False)
        print("pretrained_from ", msg)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node ", ngpus_per_node)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.resume:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    elif args.resume_from_mae:
        checkpoint = torch.load(args.resume_from_mae, map_location='cpu')
        print("INFO: checkpoint['model'].keys() = %s" % str(checkpoint['model'].keys()))
        msg = model_without_ddp.mim.load_state_dict(checkpoint['model'], strict=False)
        print("Resume from mae_origin checkpoint %s" % args.resume)
        print(msg)

    if args.resume_score_net:
        ckpt = torch.load(args.resume_score_net, map_location='cpu')
        new_ckpt = OrderedDict()
        ckpt = ckpt['model']
        for k, v in ckpt.items():
            if k.startswith('score_net.'):
                new_ckpt[k[len('score_net.'):]] = ckpt[k]
        state_dict = model_without_ddp.score_net.state_dict()
        for k in ['head.weight', 'head.bias', 'patch_embed.proj.weight', 'patch_embed.proj.bias']:
            if k in new_ckpt and new_ckpt[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del new_ckpt[k]
        msg = model_without_ddp.score_net.load_state_dict(new_ckpt, strict=False)
        print("Resume from score net checkpoint %s" % args.resume_score_net)
        print(msg)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,  # 기존 log_writer 유지 (WandB와 병행 가능)
            args=args
        )
        
        # WandB에 학습 메트릭 기록 (main process인 경우에만)
        if misc.is_main_process():
            wandb.log({**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch})

        if args.output_dir and misc.is_main_process():
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, name='checkpoint.pth')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    # main process인 경우에만 WandB 종료
    if misc.is_main_process():
        wandb.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
