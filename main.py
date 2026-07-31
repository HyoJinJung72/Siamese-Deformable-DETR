# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def _safe_torch_save(obj, path):
    path = Path(path)
    tmp_path = path.with_name(path.name + '.tmp')
    try:
        torch.save(obj, tmp_path)
        tmp_path.replace(path)
    except Exception as exc:
        print(f"[warning] Failed to save {path}: {exc}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--deep_pcb_path', default='./data/DeepPCB', type=str)
    parser.add_argument('--custom_pcb_path', default='./data/CustomPCB', type=str,
                        help='Path to the CustomPCB dataset (class4_full layout)')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrained', default='', help='Path to pre-trained model for fine-tuning')
    parser.add_argument('--save_latest', action='store_true',
                        help='Also save checkpoint.pth every epoch for resuming interrupted training')
    parser.add_argument('--save_checkpoint_interval', default=0, type=int,
                        help='Save checkpointXXXX.pth every N epochs. 0 disables periodic checkpoints')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--class_balanced_sampling', default='none',
                        choices=['none', 'inverse', 'sqrt_inverse'],
                        help='Use image-level weighted sampling based on rare GT classes during training')
    parser.add_argument('--class_balanced_bg_weight', default=0.25, type=float,
                        help='Sampling weight for images without GT boxes when class-balanced sampling is enabled')
    parser.add_argument('--class_balanced_num_samples', default=0, type=int,
                        help='Number of weighted samples per epoch. 0 uses len(dataset_train)')
    parser.add_argument('--disable_random_crop', action='store_true',
                        help='Disable RandomSizeCrop in CustomPCB train transforms while keeping resize and horizontal flip')
    parser.add_argument('--use_template', action='store_true',
                        help='Use paired template images and fuse features before the encoder')
    parser.add_argument('--template_ablation_mode', default='normal',
                        choices=['normal', 'shuffled', 'zero', 'test_as_template'],
                        help='Template input ablation for checking whether paired Gerber information is useful')
    parser.add_argument('--template_shuffle_offset', default=1, type=int,
                        help='Deterministic image offset used when --template_ablation_mode shuffled is enabled')
    parser.add_argument('--use_tcdf', action='store_true',
                        help='Enable Template Context-Difference Fusion (TCDF): fuse the context and difference branches with a learnable per-level gate')
    parser.add_argument('--save_eval_vis', action='store_true',
                        help='Save predicted bounding box visualizations during evaluation')
    parser.add_argument('--eval_vis_score_thresh', default=0.2, type=float,
                        help='Score threshold for saved evaluation visualizations')
    parser.add_argument('--eval_vis_max_images', default=50, type=int,
                        help='Deprecated: evaluation visualizations are saved for the full validation set')
    parser.add_argument('--eval_vis_topk', default=10, type=int,
                        help='Maximum number of predicted boxes to draw per image')
    parser.add_argument('--eval_vis_draw_gt', action='store_true',
                        help='Also draw ground-truth boxes in evaluation visualizations')
    parser.add_argument('--save_eval_cam', action='store_true',
                        help='Save CAM-style activation overlays from encoder input features during evaluation')
    parser.add_argument('--eval_cam_level', default=0, type=int,
                        help='Feature level used for CAM overlays. 0 is the highest-resolution encoder input feature')
    parser.add_argument('--eval_cam_alpha', default=0.45, type=float,
                        help='Heatmap opacity for CAM overlays')
    parser.add_argument('--save_eval_tffn_analysis', action='store_true',
                        help='Save TFFN pre-encoder test/template/diff/delta/fused heatmaps and per-image metrics')
    parser.add_argument('--eval_tffn_level', default=0, type=int,
                        help='Feature level used for TFFN analysis. 0 is the highest-resolution encoder input feature')
    parser.add_argument('--eval_tffn_alpha', default=0.45, type=float,
                        help='Heatmap opacity for TFFN analysis overlays')
    parser.add_argument('--save_dual_gate_stats', action='store_true',
                        help='Save learned Dual TFFN basic/diff gate weights during evaluation')

    # Alignment-robustness evaluation: perturb only the test image (and its boxes)
    # at eval time, leaving the template untouched, so that the test/template pair
    # becomes misaligned by a controlled amount.
    parser.add_argument('--eval_test_perturb', default='none',
                        choices=['none', 'translate', 'rotate', 'scale', 'affine',
                                 'brightness', 'contrast', 'blur', 'noise'],
                        help='Perturbation applied to the val test image to simulate misalignment')
    parser.add_argument('--eval_test_perturb_value', default=0.0, type=float,
                        help='Generic perturbation magnitude used when the type-specific value is 0')
    parser.add_argument('--eval_test_perturb_dx', default=0.0, type=float,
                        help='Horizontal shift in pixels for --eval_test_perturb translate/affine')
    parser.add_argument('--eval_test_perturb_dy', default=0.0, type=float,
                        help='Vertical shift in pixels for --eval_test_perturb translate/affine')
    parser.add_argument('--eval_test_perturb_angle', default=0.0, type=float,
                        help='Rotation in degrees for --eval_test_perturb rotate/affine')
    parser.add_argument('--eval_test_perturb_scale', default=1.0, type=float,
                        help='Scale factor for --eval_test_perturb scale/affine')
    parser.add_argument('--eval_test_perturb_seed', default=42, type=int,
                        help='Seed for stochastic perturbations (noise)')

    # Preferred knob for alignment-robustness studies: perturb only the TEMPLATE.
    # The test image and its annotations are left untouched, so COCO evaluation
    # (which scores against the original annotation file) stays valid, and a
    # reference-free model is exactly invariant by construction.
    parser.add_argument('--eval_template_perturb', default='none',
                        choices=['none', 'translate', 'rotate', 'scale', 'affine',
                                 'brightness', 'contrast', 'blur', 'noise'],
                        help='Perturbation applied to the val template image to simulate misalignment')
    parser.add_argument('--eval_template_perturb_value', default=0.0, type=float,
                        help='Generic perturbation magnitude used when the type-specific value is 0')
    parser.add_argument('--eval_template_perturb_dx', default=0.0, type=float,
                        help='Horizontal template shift in pixels for translate/affine')
    parser.add_argument('--eval_template_perturb_dy', default=0.0, type=float,
                        help='Vertical template shift in pixels for translate/affine')
    parser.add_argument('--eval_template_perturb_angle', default=0.0, type=float,
                        help='Template rotation in degrees for rotate/affine')
    parser.add_argument('--eval_template_perturb_scale', default=1.0, type=float,
                        help='Template scale factor for scale/affine')
    parser.add_argument('--eval_template_perturb_seed', default=42, type=int,
                        help='Seed for stochastic template perturbations (noise)')

    # Training-time misalignment augmentation: perturb ONLY the template with a small
    # random geometric offset each step so the model learns to tolerate test/template
    # misregistration. The test image and target boxes are left untouched.
    parser.add_argument('--train_template_misalign_translate', default=0.0, type=float,
                        help='Max random template shift in pixels during training (0 disables)')
    parser.add_argument('--train_template_misalign_rotate', default=0.0, type=float,
                        help='Max random template rotation in degrees during training (0 disables)')
    parser.add_argument('--train_template_misalign_scale', default=0.0, type=float,
                        help='Max random template scale jitter as a +/- fraction during training (0 disables)')

    return parser


def build_class_balanced_sampler(dataset, args):
    if not hasattr(dataset, 'coco') or not hasattr(dataset, 'ids'):
        raise ValueError('--class_balanced_sampling requires a COCO-style dataset with image ids.')

    annotations = dataset.coco.dataset.get('annotations', [])
    class_counts = {}
    labels_by_image = {int(image_id): [] for image_id in dataset.ids}
    valid_image_ids = set(labels_by_image.keys())
    for annotation in annotations:
        image_id = int(annotation['image_id'])
        if image_id not in valid_image_ids:
            continue
        category_id = int(annotation['category_id'])
        class_counts[category_id] = class_counts.get(category_id, 0) + 1
        labels_by_image[image_id].append(category_id)

    if not class_counts:
        raise ValueError('Cannot build class-balanced sampler because the training annotations have no boxes.')

    power = 1.0 if args.class_balanced_sampling == 'inverse' else 0.5
    max_count = max(class_counts.values())
    class_weights = {
        category_id: (max_count / max(count, 1)) ** power
        for category_id, count in class_counts.items()
    }
    bg_weight = max(float(args.class_balanced_bg_weight), 0.0)
    image_weights = []
    for image_id in dataset.ids:
        labels = labels_by_image[int(image_id)]
        if labels:
            image_weights.append(max(class_weights[label] for label in labels))
        else:
            image_weights.append(bg_weight)

    num_samples = int(args.class_balanced_num_samples) if args.class_balanced_num_samples else len(dataset)
    print('Using class-balanced sampling:')
    print(f'  mode={args.class_balanced_sampling}, num_samples={num_samples}, bg_weight={bg_weight}')
    print(f'  class_counts={class_counts}')
    print(f'  class_weights={class_weights}')
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(image_weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        if args.class_balanced_sampling != 'none':
            sampler_train = build_class_balanced_sampler(dataset_train, args)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.distributed and args.class_balanced_sampling != 'none':
        raise ValueError('--class_balanced_sampling is currently implemented for single-GPU training only.')

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    best_performance = -float('inf')
    best_ap75 = -float('inf')

    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model']
        model_state_dict = model_without_ddp.state_dict()

        for key in list(state_dict.keys()):
            if 'class_embed' in key:
                del state_dict[key]
            elif key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
                print(
                    f"Skip pretrained key with shape mismatch: {key} "
                    f"{tuple(state_dict[key].shape)} -> {tuple(model_state_dict[key].shape)}"
                )
                del state_dict[key]

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(state_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print("Missing Keys when loading pretrained weights:")
            for key in missing_keys:
                print(f"  {key}")
        if len(unexpected_keys) > 0:
            print("Unexpected Keys when loading pretrained weights:")
            for key in unexpected_keys:
                print(f"  {key}")

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        best_performance = checkpoint.get('performance', best_performance)
        best_ap75 = checkpoint.get('performance_ap75', best_ap75)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args
            )

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args)
        lr_scheduler.step()
        if args.output_dir and (args.save_latest or args.save_checkpoint_interval > 0):
            checkpoint_paths = []
            if args.save_latest:
                checkpoint_paths.append(output_dir / 'checkpoint.pth')
            if args.save_checkpoint_interval > 0 and (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args
        )

        if args.output_dir and 'coco_eval_bbox' in test_stats:
            current_performance = test_stats['coco_eval_bbox'][0]
            current_ap50 = test_stats['coco_eval_bbox'][1]
            current_ap75 = test_stats['coco_eval_bbox'][2]
            if current_performance > best_performance:
                best_performance = current_performance
                best_checkpoint_path = output_dir / 'checkpoint_best.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'performance': best_performance,
                    'performance_ap50': current_ap50,
                    'performance_ap75': current_ap75,
                }, best_checkpoint_path)
                print(f"[*] New best performance at epoch {epoch}: {best_performance:.4f}. Saved to {best_checkpoint_path}")
            if current_ap75 > best_ap75:
                best_ap75 = current_ap75
                best_ap75_checkpoint_path = output_dir / 'checkpoint_best_ap75.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'performance': current_performance,
                    'performance_ap50': current_ap50,
                    'performance_ap75': best_ap75,
                }, best_ap75_checkpoint_path)
                print(f"[*] New best AP75 at epoch {epoch}: {best_ap75:.4f}. Saved to {best_ap75_checkpoint_path}")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        _safe_torch_save(coco_evaluator.coco_eval["bbox"].eval,
                                         output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
