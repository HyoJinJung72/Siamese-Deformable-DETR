# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco
from .deeppcb import build as build_deeppcb
from .custompcb import build as build_custompcb
from .custompcb_class4 import build as build_custompcb_class4
from .custompcb_class4_full import build as build_custompcb_class4_full
from .custompcb_gerber import build as build_custompcb_gerber
from .custompcb_full import build as build_custompcb_full


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file in ['deeppcb', 'DeepPCB']:
        return build_deeppcb(image_set, args)
    if args.dataset_file in ['custompcb', 'CustomPCB']:
        return build_custompcb(image_set, args)
    if args.dataset_file in ['custompcb_class4', 'CustomPCB_class4']:
        return build_custompcb_class4(image_set, args)
    if args.dataset_file in ['custompcb_class4_full', 'CustomPCB_class4_full']:
        return build_custompcb_class4_full(image_set, args)
    if args.dataset_file in ['custompcb_gerber', 'CustomPCB_gerber']:
        return build_custompcb_gerber(image_set, args)
    if args.dataset_file in ['custompcb_full', 'CustomPCB_full']:
        return build_custompcb_full(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
