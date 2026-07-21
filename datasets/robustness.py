import math

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import ImageEnhance, ImageFilter


def build_eval_test_perturbation_args(args, image_set):
    perturb_type = getattr(args, 'eval_test_perturb', 'none')
    if image_set != 'val' or perturb_type == 'none':
        return None
    value = getattr(args, 'eval_test_perturb_value', 0.0)
    scale = getattr(args, 'eval_test_perturb_scale', 1.0)
    if perturb_type == 'scale' and value != 0.0:
        scale = value
    return {
        'type': perturb_type,
        'value': value,
        'dx': getattr(args, 'eval_test_perturb_dx', 0.0),
        'dy': getattr(args, 'eval_test_perturb_dy', 0.0),
        'angle': getattr(args, 'eval_test_perturb_angle', 0.0),
        'scale': scale,
        'seed': getattr(args, 'eval_test_perturb_seed', 42),
    }


def _filter_target_by_keep(target, keep):
    for field in ['boxes', 'labels', 'masks', 'area', 'iscrowd']:
        if field in target:
            target[field] = target[field][keep]
    if 'area' in target and 'boxes' in target:
        boxes = target['boxes']
        target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return target


def _transform_boxes_xyxy(target, image_size, dx=0.0, dy=0.0, angle=0.0, scale=1.0):
    if target is None or 'boxes' not in target or len(target['boxes']) == 0:
        return target

    w, h = image_size
    boxes = target['boxes']
    corners = torch.stack([
        torch.stack([boxes[:, 0], boxes[:, 1]], dim=1),
        torch.stack([boxes[:, 2], boxes[:, 1]], dim=1),
        torch.stack([boxes[:, 2], boxes[:, 3]], dim=1),
        torch.stack([boxes[:, 0], boxes[:, 3]], dim=1),
    ], dim=1)

    center = corners.new_tensor([w / 2.0, h / 2.0])
    theta = math.radians(angle)
    cos_t = math.cos(theta) * scale
    sin_t = math.sin(theta) * scale
    rot = corners.new_tensor([[cos_t, -sin_t], [sin_t, cos_t]])
    translation = corners.new_tensor([dx, dy])
    transformed = (corners - center) @ rot.t() + center + translation

    x0 = transformed[..., 0].min(dim=1).values.clamp(min=0, max=w)
    y0 = transformed[..., 1].min(dim=1).values.clamp(min=0, max=h)
    x1 = transformed[..., 0].max(dim=1).values.clamp(min=0, max=w)
    y1 = transformed[..., 1].max(dim=1).values.clamp(min=0, max=h)
    target['boxes'] = torch.stack([x0, y0, x1, y1], dim=1)

    keep = (target['boxes'][:, 2] > target['boxes'][:, 0]) & (target['boxes'][:, 3] > target['boxes'][:, 1])
    return _filter_target_by_keep(target, keep)


def apply_eval_test_perturbation(image, target, perturb_args, image_id=0):
    if perturb_args is None or perturb_args.get('type', 'none') == 'none':
        return image, target

    perturb_type = perturb_args['type']
    value = perturb_args.get('value', 0.0)
    dx = perturb_args.get('dx', 0.0)
    dy = perturb_args.get('dy', 0.0)
    angle = perturb_args.get('angle', 0.0)
    scale = perturb_args.get('scale', 1.0)

    if perturb_type == 'translate':
        dx = dx if dx != 0.0 else value
        image = TF.affine(image, angle=0.0, translate=[int(round(dx)), int(round(dy))],
                          scale=1.0, shear=0.0)
        target = _transform_boxes_xyxy(target, image.size, dx=dx, dy=dy)
    elif perturb_type == 'rotate':
        angle = angle if angle != 0.0 else value
        image = TF.affine(image, angle=angle, translate=[0, 0], scale=1.0, shear=0.0)
        target = _transform_boxes_xyxy(target, image.size, angle=angle)
    elif perturb_type == 'scale':
        image = TF.affine(image, angle=0.0, translate=[0, 0], scale=scale, shear=0.0)
        target = _transform_boxes_xyxy(target, image.size, scale=scale)
    elif perturb_type == 'affine':
        image = TF.affine(image, angle=angle, translate=[int(round(dx)), int(round(dy))],
                          scale=scale, shear=0.0)
        target = _transform_boxes_xyxy(target, image.size, dx=dx, dy=dy, angle=angle, scale=scale)
    elif perturb_type == 'brightness':
        image = ImageEnhance.Brightness(image).enhance(value)
    elif perturb_type == 'contrast':
        image = ImageEnhance.Contrast(image).enhance(value)
    elif perturb_type == 'blur':
        image = image.filter(ImageFilter.GaussianBlur(radius=value))
    elif perturb_type == 'noise':
        std = value * 255.0 if value <= 1.0 else value
        rng = np.random.default_rng(int(perturb_args.get('seed', 42)) + int(image_id))
        array = np.asarray(image).astype(np.float32)
        array = np.clip(array + rng.normal(0.0, std, size=array.shape), 0, 255).astype(np.uint8)
        image = TF.to_pil_image(array)
    else:
        raise ValueError(f'Unknown eval_test_perturb type: {perturb_type}')

    return image, target
