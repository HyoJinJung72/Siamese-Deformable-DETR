import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .coco import CocoDetection, make_coco_transforms
from .robustness import (apply_eval_test_perturbation, build_eval_test_perturbation_args,
                         apply_eval_template_perturbation, build_eval_template_perturbation_args,
                         apply_train_template_misalignment, build_train_template_misalign_args)


def apply_random_affine_pair(image, template_image, target, translate=(0.1, 0.1), scale=(0.9, 1.1)):
    w, h = image.size
    max_dx = translate[0] * w
    max_dy = translate[1] * h
    tx = random.uniform(-max_dx, max_dx)
    ty = random.uniform(-max_dy, max_dy)
    s = random.uniform(scale[0], scale[1])

    image = TF.affine(image, 0, [tx, ty], s, 0.0)
    if template_image is not None:
        template_image = TF.affine(template_image, 0, [tx, ty], s, 0.0)

    if target is not None and 'boxes' in target:
        boxes = target['boxes']
        if len(boxes) > 0:
            boxes[:, 0::2] = (boxes[:, 0::2] - w / 2) * s + w / 2 + tx
            boxes[:, 1::2] = (boxes[:, 1::2] - h / 2) * s + h / 2 + ty
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            for field in ['boxes', 'labels', 'masks', 'area', 'iscrowd']:
                if field in target:
                    target[field] = target[field][keep]
            if 'area' in target and 'boxes' in target:
                kept_boxes = target['boxes']
                target['area'] = (kept_boxes[:, 2] - kept_boxes[:, 0]) * (kept_boxes[:, 3] - kept_boxes[:, 1])

    return image, template_image, target


class DeepPCBDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, use_template=False, apply_affine=False,
                 eval_test_perturbation=None, eval_template_perturbation=None,
                 train_template_misalign=None):
        super().__init__(
            img_folder,
            ann_file,
            transforms=transforms,
            return_masks=return_masks,
            use_template=use_template,
            template_root=img_folder,
            template_key='template_file_name',
        )
        self.apply_affine = apply_affine
        self.eval_test_perturbation = eval_test_perturbation
        self.eval_template_perturbation = eval_template_perturbation
        self.train_template_misalign = train_template_misalign

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]

        template_img = None
        if self.use_template:
            template_path = Path(self.root) / img_info['template_file_name']
            if not template_path.exists():
                raise FileNotFoundError(f'Template image not found: {template_path}')
            template_img = Image.open(template_path).convert('RGB')

        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        img, target = apply_eval_test_perturbation(
            img, target, self.eval_test_perturbation, image_id=image_id)

        if template_img is not None:
            template_img = apply_eval_template_perturbation(
                template_img, self.eval_template_perturbation, image_id=image_id)
            template_img = apply_train_template_misalignment(
                template_img, self.train_template_misalign)

        if self.apply_affine:
            img, template_img, target = apply_random_affine_pair(img, template_img, target)

        if self._transforms is not None:
            random_state = random.getstate()
            torch_state = torch.random.get_rng_state()
            img, target = self._transforms(img, target)
            if self.use_template:
                random.setstate(random_state)
                torch.random.set_rng_state(torch_state)
                template_img, _ = self._transforms(template_img, {})

        if self.use_template:
            return img, template_img, target
        return img, target


def build(image_set, args):
    root = Path(getattr(args, 'deep_pcb_path', '') or args.coco_path)
    if not root.exists() and not root.is_absolute():
        parent_relative_root = Path.cwd().parent / root
        if parent_relative_root.exists():
            root = parent_relative_root
    assert root.exists(), f'provided DeepPCB path {root} does not exist'

    paths = {
        'train': (root, root / 'annotations' / 'instances_train2017.json'),
        'val': (root, root / 'annotations' / 'instances_val2017.json'),
    }
    img_folder, ann_file = paths[image_set]
    dataset = DeepPCBDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(
            image_set,
            max_size=(getattr(args, 'input_max_size', 0) or 1333),
        ),
        return_masks=args.masks,
        use_template=getattr(args, 'use_template', False),
        apply_affine=(image_set == 'train' and getattr(args, 'use_affine', False)),
        eval_test_perturbation=build_eval_test_perturbation_args(args, image_set),
        eval_template_perturbation=build_eval_template_perturbation_args(args, image_set),
        train_template_misalign=build_train_template_misalign_args(args, image_set),
    )
    return dataset
