# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import csv
import copy
from typing import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
import util.misc as utils
from util import box_ops
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from PIL import Image, ImageDraw


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _save_dual_tffn_gate_stats(model, output_dir):
    model_core = _unwrap_model(model)
    if not hasattr(model_core, "dual_tffn_gate_logits"):
        return

    output_path = Path(output_dir) / "dual_tffn_gate_stats.csv"
    diff_weights = torch.sigmoid(model_core.dual_tffn_gate_logits.detach()).cpu().tolist()
    primary_branch = "diff" if getattr(model_core, "use_tffn_diff", False) else "basic"
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "basic_weight", "diff_weight", "primary_branch"])
        for level, diff_weight in enumerate(diff_weights):
            writer.writerow([level, 1.0 - diff_weight, diff_weight, primary_branch])
    print(f"Saved Dual TFFN gate stats to {output_path}")


def _reliability_mode(args):
    return getattr(args, "template_reliability_mode", "none") if args is not None else "none"


def _reliability_is_active(args, epoch):
    mode = _reliability_mode(args)
    warmup = getattr(args, "template_reliability_warmup_epochs", 0)
    return mode != "none" and epoch >= warmup


def _clamp_reliability(scores, args):
    min_weight = getattr(args, "template_reliability_min_weight", 0.3)
    return scores.clamp(0.0, 1.0) * (1.0 - min_weight) + min_weight


def _get_reliability_memory(model):
    model_core = _unwrap_model(model)
    if not hasattr(model_core, "template_reliability_memory"):
        model_core.template_reliability_memory = {}
    return model_core.template_reliability_memory


def _target_image_ids(targets):
    return [int(target["image_id"].item()) for target in targets]


def _lookup_loss_reliability(model, targets, device):
    memory = _get_reliability_memory(model)
    weights = [memory.get(image_id, 1.0) for image_id in _target_image_ids(targets)]
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _update_loss_reliability(model, targets, reliability, args):
    memory = _get_reliability_memory(model)
    ema = getattr(args, "template_reliability_ema", 0.7)
    for image_id, score in zip(_target_image_ids(targets), reliability.detach().cpu().tolist()):
        old_score = memory.get(image_id, score)
        memory[image_id] = ema * old_score + (1.0 - ema) * score


def _get_or_create_consistency_teacher(model):
    model_core = _unwrap_model(model)
    if not hasattr(model_core, "template_consistency_teacher"):
        teacher = copy.deepcopy(model_core)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        object.__setattr__(model_core, "template_consistency_teacher", teacher)
    return model_core.template_consistency_teacher


@torch.no_grad()
def _update_consistency_teacher(model, args):
    if _reliability_mode(args) != "consistency":
        return
    if getattr(args, "template_consistency_teacher", "student") != "ema":
        return

    teacher = _get_or_create_consistency_teacher(model)
    student = _unwrap_model(model)
    decay = getattr(args, "template_consistency_ema_decay", 0.999)
    decay = min(max(float(decay), 0.0), 1.0)
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    for key, teacher_value in teacher_state.items():
        student_value = student_state[key].detach()
        if torch.is_floating_point(teacher_value):
            teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value)
    teacher.eval()


@torch.no_grad()
def _estimate_per_sample_detection_loss(outputs, targets, criterion, args):
    outputs_for_match = {
        "pred_logits": outputs["pred_logits"].detach(),
        "pred_boxes": outputs["pred_boxes"].detach(),
    }
    indices = criterion.matcher(outputs_for_match, targets)
    logits = outputs["pred_logits"].detach()
    boxes = outputs["pred_boxes"].detach()
    scores = []

    for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
        logits_i = logits[batch_idx]
        boxes_i = boxes[batch_idx]
        target_onehot = torch.zeros_like(logits_i)
        if src_idx.numel() > 0:
            labels = targets[batch_idx]["labels"][tgt_idx]
            target_onehot[src_idx, labels] = 1

        prob = logits_i.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(logits_i, target_onehot, reduction="none")
        p_t = prob * target_onehot + (1 - prob) * (1 - target_onehot)
        focal_loss = ce_loss * ((1 - p_t) ** 2)
        if criterion.focal_alpha >= 0:
            alpha_t = criterion.focal_alpha * target_onehot + (1 - criterion.focal_alpha) * (1 - target_onehot)
            focal_loss = alpha_t * focal_loss
        num_targets = max(len(targets[batch_idx]["labels"]), 1)
        cls_loss = focal_loss.mean(0).sum() * logits_i.shape[0] / num_targets

        if src_idx.numel() > 0:
            target_boxes = targets[batch_idx]["boxes"][tgt_idx]
            bbox_loss = F.l1_loss(boxes_i[src_idx], target_boxes, reduction="none").sum(dim=1).mean()
            giou_loss = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(boxes_i[src_idx]),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))
            giou_loss = giou_loss.mean()
        else:
            bbox_loss = torch.zeros((), device=logits.device)
            giou_loss = torch.zeros((), device=logits.device)

        total_loss = (
            getattr(args, "cls_loss_coef", 2.0) * cls_loss +
            getattr(args, "bbox_loss_coef", 5.0) * bbox_loss +
            getattr(args, "giou_loss_coef", 2.0) * giou_loss
        )
        scores.append(total_loss)

    losses = torch.stack(scores)
    centered_losses = losses - losses.min()
    scale = losses.mean().clamp_min(1e-6) * getattr(args, "template_reliability_temperature", 1.0)
    reliability = torch.exp(-centered_losses / scale)
    return _clamp_reliability(reliability, args)


def _consistency_views(args):
    views = getattr(args, "template_consistency_views", ["brightness"])
    if isinstance(views, str):
        views = [views]
    return views or ["brightness"]


def _perturb_samples_for_consistency(samples, view, args):
    tensors = samples.tensors
    if view == "brightness":
        factor = getattr(args, "template_consistency_factor", 1.1)
        perturbed = tensors * factor
    elif view == "contrast":
        factor = getattr(args, "template_consistency_contrast_factor", 1.1)
        spatial_mean = tensors.mean(dim=(-2, -1), keepdim=True)
        perturbed = (tensors - spatial_mean) * factor + spatial_mean
    elif view == "noise":
        std = getattr(args, "template_consistency_noise_std", 0.03)
        perturbed = tensors + torch.randn_like(tensors) * std
    else:
        raise ValueError(f"Unknown consistency perturbation view: {view}")
    return utils.NestedTensor(perturbed, samples.mask)


def _query_scope_weights(query_reliability, top_indices, args):
    if getattr(args, "template_consistency_query_scope", "topk") == "all":
        return query_reliability

    weights = torch.ones_like(query_reliability)
    weights.scatter_(1, top_indices, torch.gather(query_reliability, 1, top_indices))
    return weights


def _consistency_query_and_sample_scores(ref_outputs, aug_outputs, args):
    prob = ref_outputs["pred_logits"].detach().sigmoid()
    aug_prob = aug_outputs["pred_logits"].detach().sigmoid()
    boxes = ref_outputs["pred_boxes"].detach()
    aug_boxes = aug_outputs["pred_boxes"].detach()

    objectness = prob.max(dim=-1).values
    aug_objectness = aug_prob.max(dim=-1).values
    topk = min(getattr(args, "template_consistency_topk", 20), objectness.shape[1])
    top_indices = objectness.topk(topk, dim=1).indices

    obj_delta = torch.abs(objectness - aug_objectness)
    box_delta = torch.abs(boxes - aug_boxes).sum(dim=-1)
    obj_score = torch.exp(-obj_delta / max(getattr(args, "template_consistency_obj_temp", 0.1), 1e-6))
    box_score = torch.exp(-box_delta / max(getattr(args, "template_consistency_box_temp", 0.05), 1e-6))
    query_reliability = 0.5 * (obj_score + box_score)
    sample_reliability = torch.gather(query_reliability, 1, top_indices).mean(dim=1)

    return query_reliability, sample_reliability, top_indices


def _consistency_score_from_outputs(ref_outputs, aug_outputs, args):
    query_reliability, sample_reliability, top_indices = _consistency_query_and_sample_scores(
        ref_outputs, aug_outputs, args
    )

    if getattr(args, "template_consistency_granularity", "sample") == "query":
        return _query_scope_weights(query_reliability, top_indices, args)

    return sample_reliability


@torch.no_grad()
def _prediction_consistency_reliability(model, samples, template_samples, outputs, args):
    teacher_mode = getattr(args, "template_consistency_teacher", "student")
    teacher_model = _get_or_create_consistency_teacher(model) if teacher_mode == "ema" else None
    was_training = model.training
    if was_training:
        model.eval()
    if teacher_model is not None:
        teacher_model.eval()
    try:
        ref_outputs = outputs if teacher_model is None else teacher_model(samples, template_samples)
        view_scores = []
        for view in _consistency_views(args):
            perturbed_samples = _perturb_samples_for_consistency(samples, view, args)
            aug_outputs = model(perturbed_samples, template_samples)
            view_scores.append(_consistency_score_from_outputs(ref_outputs, aug_outputs, args))
    finally:
        if was_training:
            model.train()

    if len(view_scores) == 1:
        reliability = view_scores[0]
    else:
        stacked_scores = torch.stack(view_scores, dim=0)
        if getattr(args, "template_consistency_reduce", "mean") == "min":
            reliability = stacked_scores.min(dim=0).values
        else:
            reliability = stacked_scores.mean(dim=0)
    return _clamp_reliability(reliability, args)


def _get_sample_weights(model, samples, template_samples, outputs, targets, criterion, args, epoch):
    if not _reliability_is_active(args, epoch):
        return None

    mode = _reliability_mode(args)
    if mode == "diff":
        reliability = outputs.get("template_reliability", None)
        if reliability is None:
            return None
        return reliability.detach().to(outputs["pred_logits"].device)
    if mode == "loss":
        return _lookup_loss_reliability(model, targets, outputs["pred_logits"].device)
    if mode == "consistency":
        return _prediction_consistency_reliability(model, samples, template_samples, outputs, args)
    return None


def _get_category_id_to_name(coco_gt):
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    return {cat["id"]: cat["name"] for cat in cats}


def _map_model_labels_to_category_ids(coco_gt, labels):
    label_map = getattr(coco_gt, "model_label_to_category_id", None)
    if label_map is None:
        return labels
    mapped = [label_map.get(int(label), int(label)) for label in labels.tolist()]
    return torch.as_tensor(mapped, dtype=labels.dtype)


def _mean_valid_precision(precision_slice):
    valid = precision_slice[precision_slice > -1]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def _save_per_class_ap(coco_eval, output_dir):
    precision = coco_eval.eval["precision"]
    cat_ids = coco_eval.params.catIds
    id_to_name = _get_category_id_to_name(coco_eval.cocoGt)
    output_path = Path(output_dir) / "eval_per_class.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category_id", "category_name", "ap", "ap50", "ap75"])
        for class_index, cat_id in enumerate(cat_ids):
            ap = _mean_valid_precision(precision[:, :, class_index, 0, -1])
            ap50 = _mean_valid_precision(precision[0, :, class_index, 0, -1])
            ap75 = _mean_valid_precision(precision[5, :, class_index, 0, -1])
            writer.writerow([
                cat_id,
                id_to_name.get(cat_id, str(cat_id)),
                ap,
                ap50,
                ap75,
            ])

    print(f"Saved per-class AP to {output_path}")


def _resolve_dataset_image_path(dataset, image_id):
    image_info = dataset.coco.loadImgs(image_id)[0]
    return Path(dataset.root) / image_info["file_name"]


def _resolve_dataset_template_path(dataset, image_id):
    if not getattr(dataset, "use_template", False):
        return None

    if hasattr(dataset, "_get_template_path"):
        image_info = dataset.coco.loadImgs(image_id)[0]
        template_path = dataset._get_template_path(image_info)
        if template_path is not None:
            return Path(template_path)

    if hasattr(dataset, "_resolve_template_path"):
        template_path = dataset._resolve_template_path(image_id)
        if template_path is not None:
            return Path(template_path)

    return None


def _get_visualization_filename(dataset, image_id):
    image_info = dataset.coco.loadImgs(image_id)[0]
    filename = Path(image_info.get("file_name", f"{image_id}.jpg")).name
    if not Path(filename).suffix:
        filename = f"{filename}.jpg"
    return filename


def _get_template_analysis_filename(dataset, image_id, test_filename):
    template_path = _resolve_dataset_template_path(dataset, image_id)
    if template_path is None:
        return test_filename

    test_path = Path(test_filename)
    template_name = template_path.name
    if not Path(template_name).suffix:
        template_name = f"{template_name}{test_path.suffix or '.jpg'}"
    return f"{test_path.stem}__template__{template_name}"


def _label_color(label, num_classes, label_to_color_index):
    if num_classes <= 1:
        return (255, 0, 0)

    palette = [
        (255, 0, 0), (0, 128, 255), (0, 200, 80),
        (255, 170, 0), (180, 0, 255), (255, 0, 180),
        (0, 220, 220), (160, 220, 0), (255, 90, 0),
        (80, 80, 255),
    ]
    color_index = label_to_color_index.get(int(label), int(label))
    return palette[color_index % len(palette)]


def _readable_text_color(color):
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return (0, 0, 0) if luminance > 160 else (255, 255, 255)


def _draw_label(draw, xy, caption, color):
    x0, y0 = xy
    padding = 3
    text_bbox = draw.textbbox((x0, y0), caption)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    bg = [
        x0,
        y0,
        x0 + text_w + padding * 2,
        y0 + text_h + padding * 2,
    ]
    draw.rectangle(bg, fill=color)
    draw.text((x0 + padding, y0 + padding), caption, fill=_readable_text_color(color))


def _draw_boxes(draw, boxes, labels, scores, id_to_name, width, num_classes, label_to_color_index):
    for box, score, label in zip(boxes, scores, labels):
        color = _label_color(label.item(), num_classes, label_to_color_index)
        x0, y0, x1, y1 = box.tolist()
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([int(x0), int(y0), int(x1), int(y1)], outline=color, width=width)
        class_name = id_to_name.get(label.item(), str(label.item()))
        caption = f"{class_name}:{score:.2f}"
        label_y = y0 - 18 if y0 >= 18 else y0 + 2
        _draw_label(draw, (int(x0), int(label_y)), caption, color)


def _save_prediction_visualizations(dataset, targets, results, output_dir, score_thresh, max_images, saved_count,
                                    topk, draw_gt):
    vis_dir = Path(output_dir) / "eval_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(output_dir) / "ground_truth"
    if draw_gt:
        gt_dir.mkdir(parents=True, exist_ok=True)
    id_to_name = _get_category_id_to_name(dataset.coco)
    valid_cat_ids = set(dataset.coco.getCatIds())
    cat_ids = sorted(valid_cat_ids)
    label_to_color_index = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    num_classes = len(cat_ids)

    for target, result in zip(targets, results):
        if saved_count >= max_images:
            break
        image_id = target["image_id"].item()
        image_path = _resolve_dataset_image_path(dataset, image_id)
        save_name = _get_visualization_filename(dataset, image_id)
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        boxes = result["boxes"].detach().cpu()
        scores = result["scores"].detach().cpu()
        labels = _map_model_labels_to_category_ids(dataset.coco, result["labels"].detach().cpu())

        valid_label_mask = torch.tensor([label.item() in valid_cat_ids for label in labels], dtype=torch.bool)
        boxes = boxes[valid_label_mask]
        scores = scores[valid_label_mask]
        labels = labels[valid_label_mask]

        if boxes.numel() > 0:
            h, w = image.height, image.width
            boxes[:, 0::2] = boxes[:, 0::2].clamp_(min=0, max=w - 1)
            boxes[:, 1::2] = boxes[:, 1::2].clamp_(min=0, max=h - 1)

        keep = scores >= score_thresh
        pred_boxes = boxes[keep]
        pred_scores = scores[keep]
        pred_labels = labels[keep]

        if pred_boxes.shape[0] > topk:
            top_indices = torch.topk(pred_scores, k=topk).indices
            pred_boxes = pred_boxes[top_indices]
            pred_scores = pred_scores[top_indices]
            pred_labels = pred_labels[top_indices]

        if draw_gt and "boxes" in target and target["boxes"].numel() > 0:
            gt_image = Image.open(image_path).convert("RGB")
            gt_draw = ImageDraw.Draw(gt_image)
            gt_boxes = target["boxes"].detach().cpu().clone()
            gt_labels = _map_model_labels_to_category_ids(dataset.coco, target["labels"].detach().cpu().clone())
            gt_scores = torch.ones(gt_labels.shape[0])
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            orig_h, orig_w = target["orig_size"].detach().cpu().tolist()
            scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=gt_boxes.dtype)
            gt_boxes = gt_boxes * scale
            gt_boxes[:, 0::2] = gt_boxes[:, 0::2].clamp_(min=0, max=gt_image.width - 1)
            gt_boxes[:, 1::2] = gt_boxes[:, 1::2].clamp_(min=0, max=gt_image.height - 1)
            _draw_boxes(gt_draw, gt_boxes, gt_labels, gt_scores,
                        {k: f"gt:{v}" for k, v in id_to_name.items()},
                        width=4, num_classes=num_classes,
                        label_to_color_index=label_to_color_index)
            gt_image.save(gt_dir / save_name)

        _draw_boxes(draw, pred_boxes, pred_labels, pred_scores, id_to_name,
                    width=4, num_classes=num_classes,
                    label_to_color_index=label_to_color_index)

        image.save(vis_dir / save_name)
        saved_count += 1

    return saved_count


def _normalize_cam(cam):
    cam = cam.float()
    finite_mask = torch.isfinite(cam)
    if not finite_mask.any():
        return torch.zeros_like(cam)
    cam = torch.where(finite_mask, cam, torch.zeros_like(cam))
    low = torch.quantile(cam.flatten(), 0.02)
    high = torch.quantile(cam.flatten(), 0.98)
    if (high - low).abs() < 1e-6:
        low = cam.min()
        high = cam.max()
    cam = (cam - low) / (high - low + 1e-6)
    return cam.clamp(0.0, 1.0)


def _jet_colormap(cam):
    cam = cam.clamp(0.0, 1.0)
    red = (1.5 - torch.abs(4.0 * cam - 3.0)).clamp(0.0, 1.0)
    green = (1.5 - torch.abs(4.0 * cam - 2.0)).clamp(0.0, 1.0)
    blue = (1.5 - torch.abs(4.0 * cam - 1.0)).clamp(0.0, 1.0)
    return torch.stack([red, green, blue], dim=-1)


# TFFN analysis 시각화용: map_type별 colormap (논문 figure 색 대비)
_TFFN_CMAP = {
    'template': 'viridis',       # Reference F^R : 파랑-노랑
    'test': 'Oranges',           # Test F^T      : 주황
    'diff': 'magma',             # |F^R-F^T|     : 보라-밝음
    'delta_context': 'viridis',  # Δ_context     : 맥락(파랑-초록)
    'delta_diff': 'magma',       # Δ_diff        : 결함 강조(보라-밝음)
    'delta': 'magma',
    'fused': 'viridis',          # F^fused       : 융합
}
# 순수 히트맵(이미지 미혼합)으로 렌더 -> 어두운 배경에 결함만 밝게(대비 강조)
_TFFN_PURE = {'diff', 'delta_context', 'delta_diff', 'delta'}
# map별 alpha 오버라이드(오버레이 맵 전용): 값 클수록 colormap 활성이 더 진하게 보임
_TFFN_ALPHA = {'fused': 0.85}


def _apply_named_colormap(cam, cmap_name):
    """cam: [H,W] in [0,1] -> [H,W,3] RGB in [0,1] (matplotlib colormap)."""
    import matplotlib
    arr = cam.clamp(0.0, 1.0).cpu().numpy()
    rgba = matplotlib.colormaps[cmap_name](arr)  # [H,W,4]
    return torch.from_numpy(rgba[..., :3]).float()


def _make_cam_overlay(image, feature, alpha):
    # Classifier-style CAM is not directly defined for DETR, so this summarizes
    # the encoder-input feature activation that the detector receives.
    cam = feature.detach().float().abs().mean(dim=0)
    cam = F.interpolate(
        cam[None, None],
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu()
    cam = _normalize_cam(cam)
    heatmap = (_jet_colormap(cam) * 255).byte().numpy()
    heatmap_image = Image.fromarray(heatmap, mode="RGB")
    return Image.blend(image, heatmap_image, alpha=max(0.0, min(float(alpha), 1.0)))


def _activation_map(feature, image):
    activation = feature.detach().float().abs().mean(dim=0)
    activation = F.interpolate(
        activation[None, None],
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu()
    return activation


def _make_activation_overlay(image, feature, alpha, cmap=None, pure=False):
    activation = _normalize_cam(_activation_map(feature, image))
    if cmap is None:
        heatmap = (_jet_colormap(activation) * 255).byte().numpy()
    else:
        heatmap = (_apply_named_colormap(activation, cmap) * 255).byte().numpy()
    heatmap_image = Image.fromarray(heatmap, mode="RGB")
    if pure:  # 이미지와 혼합하지 않고 순수 히트맵 반환
        return heatmap_image
    return Image.blend(image, heatmap_image, alpha=max(0.0, min(float(alpha), 1.0)))


def _gt_mask_from_target(target, height, width):
    mask = torch.zeros((height, width), dtype=torch.bool)
    if "boxes" not in target or target["boxes"].numel() == 0:
        return mask
    boxes = target["boxes"].detach().cpu()
    boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    orig_h, orig_w = target["orig_size"].detach().cpu().tolist()
    scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=boxes.dtype)
    boxes = boxes * scale
    boxes[:, 0::2] = boxes[:, 0::2].clamp_(min=0, max=width - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp_(min=0, max=height - 1)
    for box in boxes:
        x0, y0, x1, y1 = box.round().to(torch.int64).tolist()
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        mask[y0:y1 + 1, x0:x1 + 1] = True
    return mask


def _activation_metrics(activation, gt_mask):
    activation = activation.float()
    flattened = activation.flatten()
    global_mean = flattened.mean().item()
    max_value = flattened.max().item()
    k = max(1, int(flattened.numel() * 0.05))
    top5_mean = flattened.topk(k).values.mean().item()
    concentration = top5_mean / (global_mean + 1e-6)

    if gt_mask.any():
        inside = activation[gt_mask].mean().item()
        outside_mask = ~gt_mask
        outside = activation[outside_mask].mean().item() if outside_mask.any() else float("nan")
        ratio = inside / (outside + 1e-6) if outside == outside else float("nan")
        coverage = gt_mask.float().mean().item()
    else:
        inside = float("nan")
        outside = float("nan")
        ratio = float("nan")
        coverage = 0.0

    return {
        "global_mean": global_mean,
        "max": max_value,
        "top5_mean": top5_mean,
        "top5_global_ratio": concentration,
        "gt_inside_mean": inside,
        "gt_outside_mean": outside,
        "gt_inside_outside_ratio": ratio,
        "gt_coverage": coverage,
    }


def _save_cam_visualizations(dataset, targets, outputs, output_dir, max_images, saved_count, level, alpha):
    cam_features = outputs.get("cam_features")
    if not cam_features:
        return saved_count

    level = max(0, min(int(level), len(cam_features) - 1))
    features = cam_features[level].detach().cpu()
    cam_dir = Path(output_dir) / "eval_cam"
    cam_dir.mkdir(parents=True, exist_ok=True)

    for batch_index, target in enumerate(targets):
        if saved_count >= max_images:
            break
        image_id = target["image_id"].item()
        image_path = _resolve_dataset_image_path(dataset, image_id)
        save_name = _get_visualization_filename(dataset, image_id)
        image = Image.open(image_path).convert("RGB")
        cam_image = _make_cam_overlay(image, features[batch_index], alpha)
        cam_image.save(cam_dir / save_name)
        saved_count += 1

    return saved_count


def _save_tffn_analysis(dataset, targets, outputs, output_dir, max_images, saved_count, level, alpha):
    analysis_levels = outputs.get("tffn_analysis")
    if not analysis_levels:
        return saved_count

    level = max(0, min(int(level), len(analysis_levels) - 1))
    analysis = {
        name: tensor.detach().cpu()
        for name, tensor in analysis_levels[level].items()
    }
    map_names = [name for name in
                 ["test", "template", "diff", "delta_context", "delta_diff", "delta", "fused"]
                 if name in analysis]
    analysis_dir = Path(output_dir) / "eval_tffn_analysis"
    for map_name in map_names:
        (analysis_dir / map_name).mkdir(parents=True, exist_ok=True)

    csv_path = Path(output_dir) / "eval_tffn_analysis.csv"
    fieldnames = [
        "image_id", "file_name", "template_file_name", "level", "map_type",
        "has_gt", "num_gt_boxes",
        "global_mean", "max", "top5_mean", "top5_global_ratio",
        "gt_inside_mean", "gt_outside_mean", "gt_inside_outside_ratio",
        "gt_coverage",
    ]
    write_header = True
    if csv_path.exists():
        with csv_path.open("r", newline="") as existing_file:
            existing_reader = csv.reader(existing_file)
            existing_header = next(existing_reader, None)
        write_header = existing_header != fieldnames
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            if csv_path.stat().st_size > 0:
                f.write("\n")
            writer.writeheader()

        for batch_index, target in enumerate(targets):
            if saved_count >= max_images:
                break
            image_id = target["image_id"].item()
            image_path = _resolve_dataset_image_path(dataset, image_id)
            template_path = _resolve_dataset_template_path(dataset, image_id)
            save_name = _get_visualization_filename(dataset, image_id)
            template_save_name = _get_template_analysis_filename(dataset, image_id, save_name)
            image = Image.open(image_path).convert("RGB")
            template_image = (
                Image.open(template_path).convert("RGB")
                if template_path is not None and template_path.exists()
                else image
            )
            gt_mask = _gt_mask_from_target(target, image.height, image.width)
            num_gt_boxes = int(target["boxes"].shape[0]) if "boxes" in target else 0
            has_gt = gt_mask.any().item()

            for map_name in map_names:
                feature = analysis[map_name][batch_index]
                overlay_image = template_image if map_name == "template" else image
                overlay_save_name = template_save_name if map_name == "template" else save_name
                activation = _activation_map(feature, image)
                overlay = _make_activation_overlay(overlay_image, feature,
                                                   _TFFN_ALPHA.get(map_name, alpha),
                                                   cmap=_TFFN_CMAP.get(map_name),
                                                   pure=(map_name in _TFFN_PURE))
                overlay.save(analysis_dir / map_name / overlay_save_name)

                row = {
                    "image_id": int(image_id),
                    "file_name": save_name,
                    "template_file_name": template_path.name if template_path is not None else "",
                    "level": level,
                    "map_type": map_name,
                    "has_gt": int(has_gt),
                    "num_gt_boxes": num_gt_boxes,
                }
                row.update(_activation_metrics(activation, gt_mask))
                writer.writerow(row)

            saved_count += 1

    return saved_count


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    if _reliability_is_active(args, epoch):
        metric_logger.add_meter('template_reliability', utils.SmoothedValue(window_size=20, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets, template_samples = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples, template_samples)
        sample_weights = _get_sample_weights(model, samples, template_samples, outputs, targets, criterion, args, epoch)
        loss_dict = criterion(outputs, targets, sample_weights=sample_weights)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        _update_consistency_teacher(model, args)

        if _reliability_mode(args) == "loss":
            reliability = _estimate_per_sample_detection_loss(outputs, targets, criterion, args)
            _update_loss_reliability(model, targets, reliability, args)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        if sample_weights is not None:
            metric_logger.update(template_reliability=sample_weights.mean().item())

        samples, targets, template_samples = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    save_eval_vis = bool(args is not None and getattr(args, "save_eval_vis", False) and output_dir)
    eval_vis_thresh = getattr(args, "eval_vis_score_thresh", 0.2) if args is not None else 0.2
    eval_vis_max_images = len(data_loader.dataset)
    eval_vis_topk = getattr(args, "eval_vis_topk", 10) if args is not None else 10
    eval_vis_draw_gt = bool(args is not None and getattr(args, "eval_vis_draw_gt", False))
    saved_vis_count = 0
    save_eval_cam = bool(args is not None and getattr(args, "save_eval_cam", False) and output_dir)
    eval_cam_level = getattr(args, "eval_cam_level", 0) if args is not None else 0
    eval_cam_alpha = getattr(args, "eval_cam_alpha", 0.45) if args is not None else 0.45
    eval_cam_max_images = len(data_loader.dataset)
    saved_cam_count = 0
    save_eval_tffn_analysis = bool(
        args is not None and getattr(args, "save_eval_tffn_analysis", False) and output_dir)
    eval_tffn_level = getattr(args, "eval_tffn_level", 0) if args is not None else 0
    eval_tffn_alpha = getattr(args, "eval_tffn_alpha", 0.45) if args is not None else 0.45
    eval_tffn_max_images = len(data_loader.dataset)
    saved_tffn_count = 0
    if (args is not None and getattr(args, "save_dual_gate_stats", False)
            and output_dir and utils.is_main_process()):
        _save_dual_tffn_gate_stats(model, output_dir)

    for batch in metric_logger.log_every(data_loader, 10, header):
        if len(batch) == 3:
            samples, template_samples, targets = batch
            template_samples = template_samples.to(device)
        else:
            samples, targets = batch
            template_samples = None
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, template_samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
        if save_eval_vis and utils.is_main_process() and saved_vis_count < eval_vis_max_images:
            saved_vis_count = _save_prediction_visualizations(
                data_loader.dataset, targets, results, output_dir,
                eval_vis_thresh, eval_vis_max_images, saved_vis_count,
                eval_vis_topk, eval_vis_draw_gt)
        if save_eval_cam and utils.is_main_process() and saved_cam_count < eval_cam_max_images:
            saved_cam_count = _save_cam_visualizations(
                data_loader.dataset, targets, outputs, output_dir,
                eval_cam_max_images, saved_cam_count,
                eval_cam_level, eval_cam_alpha)
        if save_eval_tffn_analysis and utils.is_main_process() and saved_tffn_count < eval_tffn_max_images:
            saved_tffn_count = _save_tffn_analysis(
                data_loader.dataset, targets, outputs, output_dir,
                eval_tffn_max_images, saved_tffn_count,
                eval_tffn_level, eval_tffn_alpha)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        if utils.is_main_process() and output_dir and 'bbox' in postprocessors.keys():
            _save_per_class_ap(coco_evaluator.coco_eval['bbox'], output_dir)
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
