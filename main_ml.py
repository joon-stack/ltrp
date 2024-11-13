# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.data.mixup import Mixup

import utils.lr_decay as lrd
import utils.misc as misc
from utils.pos_embed import interpolate_pos_embed
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import models_ltrp_vit
from engine_finetune import evaluate_offline, train_one_epoch_offline, \
    train_one_epoch_multi_label_coco, evaluate_multi_label_coco
from factory import get_score_net
from collections import OrderedDict
from multi_classification.helper_functions import CocoDetection, ModelEma, OTE_detection, NUS_WIDE_detection,COCO_CLS_NAME_2_CLS_ID_DICT
from utils.datasets import build_transform
from multi_classification.losses import AsymmetricLoss
from multi_classification.ml_decoder import add_ml_decoder_head


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0, metavar='PCT',
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
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=80, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
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
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # -----

    parser.add_argument('--keep_nums', default=147, type=int, )
    parser.add_argument('--score_net', default='', type=str)
    parser.add_argument('--finetune_ltrp', default='', help='finetune from checkpoint')
    parser.add_argument('--random_chose', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--finetune_scorenet', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--use_mask_idx', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dino_head_idx', default=-1, type=int,
                        help='Perform evaluation only')
    parser.add_argument('--ltrp_cluster_ratio', type=float, default=0.7, )
    # multi classifition
    parser.add_argument('--decoder_embedding', default=768, type=int, )
    parser.add_argument('--count_max', default=0, type=int)
    parser.add_argument('--filter_list', default='',type=str, help='finetune from checkpoint')
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    train_transform = build_transform(True, args)
    val_transform = build_transform(False, args)

    if 'voc' in args.data_path:
        # COCO Data loading
        instances_path_val = os.path.join(args.data_path, 'annotations/instances_val.json')
        instances_path_train = os.path.join(args.data_path, 'annotations/instances_train.json')

        data_path_train = f'{args.data_path}/train'  # args.data
        data_path_val = f'{args.data_path}/val'  # args.data

        dataset_train = OTE_detection(data_path_train,
                                      instances_path_train, train_transform)
        dataset_val = OTE_detection(data_path_val,
                                    instances_path_val, val_transform)
    elif 'NUS' in args.data_path:
        data_path = args.data_path.split('_TC')[0]
        if "TC10" in args.data_path:
            instances_path_val = os.path.join(data_path, 'val_tc10.txt')
            instances_path_train = os.path.join(data_path, 'train_tc10.txt')
        else:
            instances_path_val = os.path.join(args.data_path, 'val.txt')
            instances_path_train = os.path.join(args.data_path, 'train.txt')

        data_path = f'{data_path}/'  # args.data

        dataset_train = NUS_WIDE_detection(data_path,
                                           instances_path_train, train_transform)
        dataset_val = NUS_WIDE_detection(data_path,
                                         instances_path_val, val_transform)
    else:
        # COCO Data loading
        instances_path_val = os.path.join(args.data_path, 'annotations/instances_val2017.json')
        instances_path_train = os.path.join(args.data_path, 'annotations/instances_train2017.json')

        data_path_val = f'{args.data_path}/val2017'  # args.data
        data_path_train = f'{args.data_path}/train2017'  # args.data

        dataset_train = CocoDetection(data_path_train,
                                      instances_path_train, train_transform, count_max=args.count_max)
        dataset_val = CocoDetection(data_path_val,
                                    instances_path_val, val_transform, count_max=args.count_max)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
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

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.use_mask_idx:
        score_net = None
    else:
        score_net = get_score_net(args)

    model = models_ltrp_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        keep_nums=args.keep_nums,
        score_net=score_net,
        random_chose=args.random_chose,
        finetune_scorenet=args.finetune_scorenet,
        img_size=args.input_size
    )

    model = add_ml_decoder_head(model, num_classes=args.nb_classes, num_of_groups=-1,
                                decoder_embedding=args.decoder_embedding, zsl=0)

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if 'model' in checkpoint.keys():
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    if args.finetune_ltrp and not args.eval:
        checkpoint = torch.load(args.finetune_ltrp, map_location='cpu')
        print("pre loading ckpt ", score_net)
        if args.score_net.startswith('dpc_knn'):
            state_dict = checkpoint
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            checkpoint_model = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
            model.score_net.init_backbone()
            msg = model.score_net.backbone.load_state_dict(checkpoint_model, strict=False)
        elif args.score_net.startswith('grad_cam_self'):
            checkpoint_model = checkpoint['model']
            msg = model.score_net.vit.load_state_dict(checkpoint_model)
        elif args.score_net.startswith('gf_net'):
            msg = model.score_net.load(checkpoint)
        elif args.score_net.startswith('IA_RED'):
            msg = model.score_net.model.load_state_dict(checkpoint)
        elif args.score_net.startswith('dynamic'):
            msg = model.score_net.load_state_dict(checkpoint['model'])
        elif args.score_net.startswith('tcformer'):
            msg = model.score_net.load_state_dict(checkpoint['model'])
        elif args.score_net.startswith('AdaViT'):
            msg = model.score_net.vit.load_state_dict(checkpoint['state_dict'])
        elif args.score_net.startswith('dge'):
            msg = model.score_net.vit.load_state_dict(checkpoint['model'])
        elif args.score_net.startswith('top_k'):
            for i, j in checkpoint['model'].items():
                if 'pos_embed' in i:
                    checkpoint['model'][i] = torch.cat(
                        [checkpoint['model'][i][:, :1, :], checkpoint['model'][i][:, 2:, :]], dim=1)
                    print('top_k after ', checkpoint['model'][i].shape)
            msg = model.score_net.load_state_dict(checkpoint["model"], strict=False)
        elif args.score_net.startswith('tome'):
            msg = model.score_net.vit.load_state_dict(checkpoint['model'])
        elif args.score_net.startswith('A_ViT'):
            msg = model.score_net.load_state_dict(checkpoint['model'])
        else:
            if args.score_net.startswith('evit'):
                checkpoint_model = checkpoint['model']

            elif args.score_net.startswith('moco'):
                # checkpoint_model = OrderedDict()
                # ckpt = checkpoint['state_dict']
                # for k, v in ckpt.items():
                #     if k.startswith('module.encoder_q.'):
                #         checkpoint_model[k[len('module.encoder_q.'):]] = ckpt[k]
                state_dict = checkpoint['state_dict']
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()}
                checkpoint_model = {k.replace("base_encoder.", ""): v for k, v in state_dict.items()}
            elif args.score_net.startswith('dino_vit_small'):
                if 'teacher' in checkpoint.keys():
                    checkpoint = checkpoint['teacher']
                state_dict = checkpoint
                # remove `module.` prefix
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                # remove `backbone.` prefix induced by multicrop wrapper
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
                checkpoint_model = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
            else:
                checkpoint_model = OrderedDict()
                checkpoint_model_ltrp = checkpoint['model']
                for k, v in checkpoint_model_ltrp.items():
                    if k.startswith('score_net.'):
                        checkpoint_model[k[10:]] = checkpoint_model_ltrp[k]
            msg = model.score_net.load_state_dict(checkpoint_model, strict=False)
        print(args.score_net + " load ", msg)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    ema_model = ModelEma(model, 0.9997)

    filter_dict = None
    if args.filter_list != '':
        with open(args.filter_list) as f:
            print('begin filter', args.filter_list)
            filter_list = f.readlines()
            filter_list = [dataset_train.cat2cat[COCO_CLS_NAME_2_CLS_ID_DICT[i[:-1]]] for i in filter_list]  #
            all_labels = np.arange(0, len(COCO_CLS_NAME_2_CLS_ID_DICT))
            filter_list_averse = np.setdiff1d(all_labels, filter_list)
            filter_dict = {}
            filter_dict['seen'] = filter_list_averse
            filter_dict['un_seen'] = filter_list
            print('filter_dict', filter_dict)

    if args.eval:
        test_stats = evaluate_multi_label_coco(data_loader_val, model, ema_model, device, args, filter_dict)
        #print(f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']:.1f}%")
        print(
            f"Max mAP, mAP_seen, mAP_unseen: {test_stats['mAP']:.1f}%, {test_stats['mAP_seen']:.1f}% and {test_stats['mAP_unseen']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0



    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if args.use_mask_idx:
            train_stats = train_one_epoch_offline(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args
            )
        else:
            train_stats = train_one_epoch_multi_label_coco(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args
            )

        if args.output_dir and misc.is_main_process():
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, name='checkpoint.pth')

        if args.use_mask_idx:
            test_stats = evaluate_offline(data_loader_val, model, device)
        else:
            # test_stats = evaluate(data_loader_val, model, device)
            test_stats = evaluate_multi_label_coco(data_loader_val, model, ema_model, device, args, filter_dict)
        if filter_dict is None:
            print(
                f"mAP, mAP on seen and mAP on unseen of the network on the {len(dataset_val)} test images: {test_stats['mAP']:.1f}%")
        else:
            print(f"mAP, mAP on seen and mAP on unseen of the network on the {len(dataset_val)} test images: {test_stats['mAP']:.1f}%. {test_stats['mAP_seen']:.1f}%,{test_stats['mAP_unseen']:.1f}% ")

        if test_stats['mAP'] > max_accuracy:
            if args.output_dir and misc.is_main_process():
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, name='checkpoint-best.pth')


            max_accuracy = test_stats['mAP']
            if filter_dict is not None:
                max_accuracy_seen = test_stats['mAP_seen']
                max_accuracy_unseen = test_stats['mAP_unseen']

        if filter_dict is None:
            print(
                f'Max mAP, mAP_seen, mAP_unseen: {max_accuracy:.2f}%')
        else:
            print(f'Max mAP, mAP_seen, mAP_unseen: {max_accuracy:.2f}%,  {max_accuracy_seen:.2f}% and {max_accuracy_unseen:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/mAp', test_stats['mAP'], epoch)
            if filter_dict is not None:
                log_writer.add_scalar('perf/mAp_seen', test_stats['mAP_seen'], epoch)
                log_writer.add_scalar('perf/mAp_unseen', test_stats['mAP_unseen'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
