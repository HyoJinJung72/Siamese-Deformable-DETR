"""CutPaste-based synthetic defect generation for CustomPCB.

Standalone extraction of the CutPaste augmentation used in the paper. A real
defect region is cut from a defective board and pasted onto a normal board.
Unlike the original CutPaste (random placement):

  * Registered placement -- the defect is aligned to the corresponding circuit
    location on the target board via phase-correlation registration.
  * Feather blending      -- the paste boundary is Gaussian-softened (sigma) so
    the model cannot learn the paste seam instead of the defect.

This file was extracted from TF-IDG/run_custom_pcb.py, keeping only the
functions on the cutpaste code path (diffusion / structural-mask generation
removed). It reproduces the CustomPCB_cutpaste datasets bit-for-bit given the
same arguments and seed.

Example (num3 held out as the test board):
    python cutpaste_custom_pcb.py \
        --class-mode class4 \
        --data-root /path/to/CustomPCB_class4 \
        --mask-xml  /path/to/CustomPCB_masking_annotation/annotations_class4.xml \
        --output-dir /path/to/out/CustomPCB_cutpaste_num3 \
        --splits train val \
        --defect-boards num1,num2,num3,num4,num5,num6,num7,num8 \
        --normal-boards num1,num2,num4,num5,num6,num7,num8,num9 \
        --exclude-boards num3 \
        --normal-source all_normal --placement aligned --require-aligned-target \
        --ref-policy same_class --ref-match-scope same_sample \
        --combination-sampling unique_cycle --no-combination-recycle \
        --composite-mode cutpaste --cutpaste-placement registered \
        --source-blend feather --defect-blend-sigma 1.0 \
        --target-mask-policy source_mask \
        --mask-registration phase_translation \
        --samples-per-class 100 --seed 20260703
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is expected in the tfidg env.
    cKDTree = None

# ---- module constants ----


PCB_RE = re.compile(
    r"^(?P<board>num\d+)_(?P<status>normal|defect)_(?P<side>front|back)_(?P<frame>\d+)\.(?P<ext>jpg|jpeg|png)$",
    re.IGNORECASE,
)


GERBER_RE = re.compile(
    r"^gerber_normal_(?P<side>front|back)_(?P<frame>\d+)\.(?P<ext>jpg|jpeg|png)$",
    re.IGNORECASE,
)


CLASS4_LABELS = ("open", "short", "silk", "pad_open")


CROP_BACK_INTERP_CHOICES = ("bilinear", "area", "nearest", "cubic", "lanczos")



# ---- functions extracted from TF-IDG/run_custom_pcb.py (cutpaste dependency closure) ----


@dataclass(frozen=True)
class DefectRecord:
    image_name: str
    image_path: Path
    label: str
    mask: np.ndarray
    board: str
    side: str
    frame: str
    split: str
    instance_index: int = 0


@dataclass(frozen=True)
class NormalRecord:
    image_path: Path
    board: str
    side: str
    frame: str
    split: str


@dataclass(frozen=True)
class GenerationCombination:
    shape_record: DefectRecord
    ref_record: DefectRecord
    normal_record: NormalRecord
    placement_note: str
    ref_match_note: str


def normalize_label(label):
    return label.strip().lower()


def parse_board_list(value):
    return {
        part.strip().lower()
        for part in value.split(",")
        if part.strip()
    }


def parse_class_override_map(value, cast=str):
    """Parse "class=value,class=value" into {class: cast(value)}.

    An empty string yields an empty map, which means "no per-class override"
    so the caller falls back to the global argument and behavior is unchanged.
    """
    mapping = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid class override entry '{part}'. Expected 'class=value'."
            )
        key, raw = part.split("=", 1)
        key = normalize_label(key)
        raw = raw.strip()
        if not key or not raw:
            raise ValueError(
                f"Invalid class override entry '{part}'. Expected 'class=value'."
            )
        mapping[key] = cast(raw)
    return mapping


def resolve_ref_match_scope(args, label):
    return getattr(args, "class_ref_match_scope_map", {}).get(
        label, args.ref_match_scope
    )


def resolve_mask_jitter(args, label):
    return getattr(args, "class_mask_jitter_map", {}).get(
        label, args.mask_jitter_pixels
    )


def any_scope_uses_context(args, labels):
    if args.ref_match_scope == "class_context":
        return True
    overrides = getattr(args, "class_ref_match_scope_map", {})
    return any(
        overrides.get(label) == "class_context" for label in labels
    )


def resolve_blend_sigma(args, label):
    return getattr(args, "class_defect_blend_sigma_map", {}).get(
        label, args.defect_blend_sigma
    )


def resolve_contrast_scale(args, label):
    return getattr(args, "class_defect_contrast_scale_map", {}).get(
        label, args.defect_contrast_scale
    )


def resolve_mask_dilate(args, label):
    return getattr(args, "class_defect_mask_dilate_map", {}).get(
        label, tuple(args.defect_mask_dilate_range)
    )


def resolve_samples_per_class(args, label):
    return getattr(args, "class_samples_per_class_map", {}).get(
        label, args.samples_per_class
    )


def resolve_composite_mode(args, label):
    return getattr(args, "class_composite_mode_map", {}).get(
        label, args.composite_mode
    )


def resolve_cutpaste_placement(args, label):
    return getattr(args, "class_cutpaste_placement_map", {}).get(
        label, args.cutpaste_placement
    )


def dilate_defect_mask(mask, iterations):
    """Grow a defect mask by `iterations` of a 3x3 elliptical kernel."""
    if iterations <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grown = cv2.dilate(mask.astype(np.uint8), kernel, iterations=int(iterations))
    return grown if int(grown.sum()) > 0 else mask


def cutpaste_source_defect(
    target_image, ref_image, ref_mask, target_mask, blend, sigma,
    placement="registered",
):
    """Cut the real defect pixels out of the reference image and paste them
    onto the target, keeping the defect's *real* texture instead of a
    diffusion-hallucinated one.

    placement='registered' : resize the reference defect to the target-mask
        bbox (original behavior; distortion-free only when ref == shape).
    placement='native_transfer' : keep the reference defect at its native
        size and drop it centered on the target-mask centroid. This lets a
        real defect's appearance be placed at *another* real defect's location
        (empirical placement) without resize distortion.

    `blend='feather'` uses a Gaussian-feathered alpha; `blend='poisson'` uses
    gradient-domain seamlessClone to harmonize lighting while keeping texture.

    Returns (composited_image, placed_mask) where placed_mask matches the
    pasted pixels exactly, or (None, None) if the geometry is degenerate.
    """
    ref_mask = resize_mask_to_image(ref_mask, ref_image)
    ref_bin = (ref_mask > 0).astype(np.uint8)
    tar_bin = (resize_mask_to_image(target_mask, target_image) > 0).astype(np.uint8)
    rys, rxs = np.where(ref_bin)
    tys, txs = np.where(tar_bin)
    if len(rxs) == 0 or len(txs) == 0:
        return None, None

    ry1, ry2, rx1, rx2 = int(rys.min()), int(rys.max()) + 1, int(rxs.min()), int(rxs.max()) + 1
    crop = ref_image[ry1:ry2, rx1:rx2]
    crop_mask = ref_bin[ry1:ry2, rx1:rx2]
    H, W = target_image.shape[:2]

    if placement == "native_transfer":
        ch, cw = crop.shape[:2]
        if ch > H or cw > W:  # native defect bigger than the board -> can't place
            return None, None
        cy, cx = int(round(tys.mean())), int(round(txs.mean()))
        ty1 = max(0, min(cy - ch // 2, H - ch))
        tx1 = max(0, min(cx - cw // 2, W - cw))
        place_crop, place_mask = crop, crop_mask
        th, tw = ch, cw
    else:  # registered
        ty1, ty2, tx1, tx2 = int(tys.min()), int(tys.max()) + 1, int(txs.min()), int(txs.max()) + 1
        tw, th = tx2 - tx1, ty2 - ty1
        if tw < 1 or th < 1:
            return None, None
        place_crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        place_mask = cv2.resize(crop_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

    if int(place_mask.sum()) == 0:
        return None, None

    out = target_image.copy()
    out_mask = np.zeros((H, W), dtype=np.uint8)
    out_mask[ty1:ty1 + th, tx1:tx1 + tw] = place_mask
    region = out[ty1:ty1 + th, tx1:tx1 + tw]

    if blend == "poisson":
        try:
            cloned = cv2.seamlessClone(
                place_crop.astype(np.uint8),
                region.astype(np.uint8),
                (place_mask * 255).astype(np.uint8),
                (tw // 2, th // 2),
                cv2.NORMAL_CLONE,
            )
            out[ty1:ty1 + th, tx1:tx1 + tw] = cloned
            return out, out_mask
        except cv2.error:
            pass  # fall back to feather below

    alpha = place_mask.astype(np.float32)
    if sigma > 0:
        radius = max(1, int(round(sigma * 3)))
        ksize = radius * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), float(sigma))
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    blended = region.astype(np.float32) * (1.0 - alpha) + place_crop.astype(np.float32) * alpha
    out[ty1:ty1 + th, tx1:tx1 + tw] = np.clip(blended, 0, 255).astype(np.uint8)
    return out, out_mask


def parse_pcb_name(path):
    match = PCB_RE.match(path.name)
    if not match:
        return None
    return match.groupdict()


def parse_gerber_name(path):
    match = GERBER_RE.match(path.name)
    if not match:
        return None
    groups = match.groupdict()
    groups["board"] = "gerber"
    groups["status"] = "normal"
    return groups


def decode_cvat_rle(rle, width, height):
    counts = [int(part.strip()) for part in rle.split(",") if part.strip()]
    flat = np.zeros(width * height, dtype=np.uint8)
    cursor = 0
    value = 0
    for count in counts:
        if count > 0 and value == 1:
            flat[cursor : cursor + count] = 1
        cursor += count
        value = 1 - value
        if cursor >= flat.size:
            break
    return flat.reshape((height, width))


def decode_mask_element(mask_el, image_width, image_height):
    left = int(float(mask_el.attrib["left"]))
    top = int(float(mask_el.attrib["top"]))
    width = int(float(mask_el.attrib["width"]))
    height = int(float(mask_el.attrib["height"]))

    crop = decode_cvat_rle(mask_el.attrib["rle"], width, height)
    full = np.zeros((image_height, image_width), dtype=np.uint8)

    y2 = min(top + height, image_height)
    x2 = min(left + width, image_width)
    crop = crop[: y2 - top, : x2 - left]
    full[top:y2, left:x2] = np.maximum(full[top:y2, left:x2], crop)
    return full


def image_files_by_split(data_root, splits):
    files = {}
    for split in splits:
        image_dir = data_root / "images" / split
        if not image_dir.exists():
            continue
        for path in image_dir.iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            files[path.name] = (path, split)
    return files


def split_mask_components(mask, min_component_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    components = []
    for component_index in range(1, count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        components.append(
            (component_index, (labels == component_index).astype(np.uint8))
        )
    return components


def load_defect_records(
    data_root,
    mask_xml,
    class_mode,
    splits,
    include_classes,
    split_components=False,
    min_component_area=1,
):
    image_map = image_files_by_split(data_root, splits)
    if not image_map:
        raise FileNotFoundError(f"No images found under {data_root / 'images'}")

    try:
        root = ET.parse(mask_xml).getroot()
    except ET.ParseError:
        fallback_xml = mask_xml.with_name("annotations_class4.xml")
        if class_mode == "binary" and fallback_xml.exists() and fallback_xml != mask_xml:
            print(
                f"Could not parse {mask_xml}; falling back to {fallback_xml} "
                "and merging all labels into 'defect'."
            )
            root = ET.parse(fallback_xml).getroot()
        else:
            raise
    records = []
    skipped_without_image = 0
    skipped_without_name_match = 0

    for image_el in root.findall("image"):
        image_name = image_el.attrib["name"]
        if image_name not in image_map:
            skipped_without_image += 1
            continue

        image_path, split = image_map[image_name]
        parsed = parse_pcb_name(image_path)
        if parsed is None or parsed["status"].lower() != "defect":
            skipped_without_name_match += 1
            continue

        image_width = int(float(image_el.attrib["width"]))
        image_height = int(float(image_el.attrib["height"]))
        grouped_masks = defaultdict(list)

        for mask_el in image_el.findall("mask"):
            xml_label = normalize_label(mask_el.attrib["label"])
            label = "defect" if class_mode == "binary" else xml_label
            if include_classes and label not in include_classes:
                continue
            grouped_masks[label].append(
                decode_mask_element(mask_el, image_width, image_height)
            )

        for label, masks in grouped_masks.items():
            mask = np.maximum.reduce(masks).astype(np.uint8)
            if int(mask.sum()) == 0:
                continue
            instances = (
                split_mask_components(mask, min_component_area)
                if split_components
                else [(0, mask)]
            )
            for instance_index, instance_mask in instances:
                records.append(
                    DefectRecord(
                        image_name=image_name,
                        image_path=image_path,
                        label=label,
                        mask=instance_mask,
                        board=parsed["board"].lower(),
                        side=parsed["side"].lower(),
                        frame=parsed["frame"],
                        split=split,
                        instance_index=instance_index,
                    )
                )

    if not records:
        raise RuntimeError("No usable defect masks were loaded from the XML annotation.")

    return records, {
        "skipped_without_image": skipped_without_image,
        "skipped_without_name_match": skipped_without_name_match,
    }


def load_normal_records(data_root, splits, normal_source, normal_board_prefix):
    records = []
    # Some boards (e.g. num9 normals) are duplicated across splits. Keep one
    # record per file name so the combination pool is not silently inflated
    # with pixel-identical targets when several splits are loaded.
    seen_names = set()
    for split in splits:
        image_dir = data_root / "images" / split
        if not image_dir.exists():
            continue
        for path in image_dir.iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if path.name in seen_names:
                continue
            seen_names.add(path.name)

            parsed = parse_pcb_name(path)
            if parsed and parsed["status"].lower() == "normal":
                board = parsed["board"].lower()
                if normal_source == "num9" and board != normal_board_prefix:
                    continue
                if normal_source in {"num9", "all_normal"}:
                    records.append(
                        NormalRecord(
                            image_path=path,
                            board=board,
                            side=parsed["side"].lower(),
                            frame=parsed["frame"],
                            split=split,
                        )
                    )
                continue

            parsed = parse_gerber_name(path)
            if parsed and normal_source == "gerber":
                records.append(
                    NormalRecord(
                        image_path=path,
                        board="gerber",
                        side=parsed["side"].lower(),
                        frame=parsed["frame"],
                        split=split,
                    )
                )

    if not records:
        raise RuntimeError(
            f"No normal target images found for normal_source={normal_source}."
        )
    return records


def build_normal_index(normal_records):
    by_key = defaultdict(list)
    by_side = defaultdict(list)
    for record in normal_records:
        by_key[(record.side, record.frame)].append(record)
        by_side[record.side].append(record)
    return by_key, by_side


def choose_target_normal(shape_record, by_key, by_side, all_normals, placement, rng):
    if placement == "aligned":
        candidates = by_key.get((shape_record.side, shape_record.frame), [])
        if candidates:
            return rng.choice(candidates), "aligned"

    candidates = by_side.get(shape_record.side, [])
    if candidates:
        return rng.choice(candidates), "same_side_fallback"
    return rng.choice(all_normals), "any_fallback"


def target_candidates_for_shape(
    shape_record,
    by_key,
    by_side,
    all_normals,
    placement,
    require_aligned_target,
):
    if placement == "aligned":
        candidates = by_key.get((shape_record.side, shape_record.frame), [])
        if candidates:
            return candidates, "aligned"
        if require_aligned_target:
            return [], "missing_aligned_target"

    candidates = by_side.get(shape_record.side, [])
    if candidates:
        return candidates, "same_side_fallback"
    if require_aligned_target:
        return [], "missing_aligned_target"
    return all_normals, "any_fallback"


def defect_record_key(record):
    return (record.image_name, record.label, record.instance_index)


def normalized_histogram(values, bins, value_range):
    histogram, _ = np.histogram(values, bins=bins, range=value_range)
    histogram = histogram.astype(np.float32)
    total = float(histogram.sum())
    return histogram / total if total > 0 else histogram


def local_context_feature(image, mask):
    mask = resize_mask_to_image(mask, image).astype(bool)
    ys, xs = np.where(mask)
    if not len(ys):
        return np.zeros(132, dtype=np.float32)

    y1, y2 = int(ys.min()), int(ys.max() + 1)
    x1, x2 = int(xs.min()), int(xs.max() + 1)
    padding = max(8, int(round(max(y2 - y1, x2 - x1) * 1.5)))
    y1, y2 = max(0, y1 - padding), min(image.shape[0], y2 + padding)
    x1, x2 = max(0, x1 - padding), min(image.shape[1], x2 + padding)

    crop = image[y1:y2, x1:x2].astype(np.uint8)
    crop_mask = mask[y1:y2, x1:x2]
    context_pixels = ~crop_mask
    if not context_pixels.any():
        context_pixels = np.ones_like(crop_mask, dtype=bool)

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.clip(cv2.magnitude(grad_x, grad_y), 0, 512)

    fill_value = int(np.median(gray[context_pixels]))
    filled_gray = gray.copy()
    filled_gray[crop_mask] = fill_value
    spatial = cv2.resize(
        filled_gray,
        (8, 8),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32).reshape(-1) / 255.0

    feature = np.concatenate(
        [
            normalized_histogram(hsv[:, :, 0][context_pixels], 12, (0, 180)),
            normalized_histogram(hsv[:, :, 1][context_pixels], 12, (0, 256)),
            normalized_histogram(hsv[:, :, 2][context_pixels], 12, (0, 256)),
            normalized_histogram(gray[context_pixels], 16, (0, 256)),
            normalized_histogram(magnitude[context_pixels], 16, (0, 512)),
            spatial,
        ]
    ).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm > 0 else feature


def build_context_feature_cache(records):
    cache = {}
    image_cache = {}
    for record in records:
        if record.image_path not in image_cache:
            image_cache[record.image_path] = read_rgb(record.image_path)
        cache[defect_record_key(record)] = local_context_feature(
            image_cache[record.image_path],
            record.mask,
        )
    return cache


def context_match_distance(shape_record, ref_record, context_features, area_weight):
    shape_feature = context_features[defect_record_key(shape_record)]
    ref_feature = context_features[defect_record_key(ref_record)]
    cosine_distance = float(1.0 - np.clip(np.dot(shape_feature, ref_feature), -1.0, 1.0))
    shape_area = max(1, int(shape_record.mask.sum()))
    ref_area = max(1, int(ref_record.mask.sum()))
    area_distance = min(2.0, abs(float(np.log(ref_area / shape_area))))
    return cosine_distance + area_weight * area_distance


def reference_candidates_for_shape(
    shape_record,
    labels,
    by_label,
    by_label_side,
    by_label_side_frame,
    ref_match_scope,
    ref_policy,
    rng,
    context_features=None,
    context_top_k=5,
    context_area_weight=0.25,
):
    if ref_match_scope == "same_sample":
        return [shape_record], (
            f"same_sample:{shape_record.label}:"
            f"{shape_record.image_name}:instance{shape_record.instance_index}"
        )

    if ref_match_scope == "class_context":
        class_candidates = [
            record
            for record in by_label[shape_record.label]
            if record.image_name != shape_record.image_name
        ]
        if not class_candidates:
            return [shape_record], f"class_context_self_fallback:{shape_record.label}"
        if context_features is None:
            raise ValueError("Context features are required for class_context matching.")
        ranked = sorted(
            class_candidates,
            key=lambda record: (
                context_match_distance(
                    shape_record,
                    record,
                    context_features,
                    context_area_weight,
                ),
                record.image_name,
                record.instance_index,
            ),
        )
        selected = ranked[: min(context_top_k, len(ranked))]
        return selected, f"class_context:{shape_record.label}:top{len(selected)}"

    if ref_match_scope == "class_only_distinct":
        class_candidates = [
            record
            for record in by_label[shape_record.label]
            if record.image_name != shape_record.image_name
        ]
        if class_candidates:
            return (
                class_candidates,
                f"class_only_distinct:{shape_record.label}",
            )
        return [shape_record], f"class_only_self_fallback:{shape_record.label}"

    if ref_match_scope == "class_side_slice_preferred":
        strict_candidates = [
            record
            for record in by_label_side_frame[
                (shape_record.label, shape_record.side, shape_record.frame)
            ]
            if record.image_name != shape_record.image_name
        ]
        if strict_candidates:
            return (
                strict_candidates,
                f"preferred_slice:{shape_record.label}:{shape_record.side}:"
                f"{shape_record.frame}",
            )

        side_candidates = [
            record
            for record in by_label_side[(shape_record.label, shape_record.side)]
            if record.image_name != shape_record.image_name
        ]
        if side_candidates:
            return (
                side_candidates,
                f"preferred_side_fallback:{shape_record.label}:{shape_record.side}",
            )

        return (
            [shape_record],
            f"preferred_self_fallback:{shape_record.label}:{shape_record.side}:"
            f"{shape_record.frame}",
        )

    if ref_match_scope == "class_side_slice":
        return (
            by_label_side_frame[
                (shape_record.label, shape_record.side, shape_record.frame)
            ],
            f"class_side_slice:{shape_record.label}:{shape_record.side}:"
            f"{shape_record.frame}",
        )
    if ref_match_scope == "class_side":
        return (
            by_label_side[(shape_record.label, shape_record.side)],
            f"class_side:{shape_record.label}:{shape_record.side}",
        )

    ref_label = shape_record.label if ref_policy == "same_class" else rng.choice(labels)
    return by_label[ref_label], f"class:{ref_label}"


def area_distance(left_area, right_area):
    left_area = max(1, int(left_area))
    right_area = max(1, int(right_area))
    return abs(float(np.log(left_area / right_area)))


def build_combination_pool(
    label,
    labels,
    by_label,
    by_label_side,
    by_label_side_frame,
    by_key,
    by_side,
    normal_records,
    args,
    rng,
    context_features=None,
):
    combinations = []
    for shape_record in by_label[label]:
        ref_candidates, ref_match_note = reference_candidates_for_shape(
            shape_record,
            labels,
            by_label,
            by_label_side,
            by_label_side_frame,
            resolve_ref_match_scope(args, label),
            args.ref_policy,
            rng,
            context_features,
            args.context_top_k,
            args.context_area_weight,
        )
        normal_candidates, placement_note = target_candidates_for_shape(
            shape_record,
            by_key,
            by_side,
            normal_records,
            args.placement,
            args.require_aligned_target,
        )
        for ref_record in ref_candidates:
            for normal_record in normal_candidates:
                combinations.append(
                    GenerationCombination(
                        shape_record=shape_record,
                        ref_record=ref_record,
                        normal_record=normal_record,
                        placement_note=placement_note,
                        ref_match_note=ref_match_note,
                    )
                )
    rng.shuffle(combinations)
    return combinations


def shift_mask(mask, max_pixels, rng):
    if max_pixels <= 0:
        return mask
    dx = rng.randint(-max_pixels, max_pixels)
    dy = rng.randint(-max_pixels, max_pixels)
    if dx == 0 and dy == 0:
        return mask

    height, width = mask.shape
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return shifted if int(shifted.sum()) > 0 else mask


def register_mask_to_target(
    shape_image,
    target_image,
    mask,
    mode,
    min_response,
    max_shift,
):
    target_mask = resize_mask_to_image(mask, target_image)
    registration = {
        "mode": mode,
        "applied": False,
        "dx": 0.0,
        "dy": 0.0,
        "magnitude": 0.0,
        "response": None,
    }
    if mode == "none":
        return target_mask, registration

    target_height, target_width = target_image.shape[:2]
    if shape_image.shape[:2] != (target_height, target_width):
        shape_image = cv2.resize(
            shape_image,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )

    shape_gray = cv2.cvtColor(shape_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    shape_gray = shape_gray.astype(np.float32)
    target_gray = target_gray.astype(np.float32)
    window = cv2.createHanningWindow(
        (target_width, target_height),
        cv2.CV_32F,
    )
    (dx, dy), response = cv2.phaseCorrelate(shape_gray, target_gray, window)
    magnitude = float(np.hypot(dx, dy))
    registration.update(
        {
            "dx": float(dx),
            "dy": float(dy),
            "magnitude": magnitude,
            "response": float(response),
        }
    )

    if not np.isfinite([dx, dy, response]).all():
        raise ValueError("Mask registration returned non-finite values.")
    if response < min_response:
        raise ValueError(
            f"Mask registration response {response:.3f} is below "
            f"{min_response:.3f}."
        )
    if max_shift > 0 and magnitude > max_shift:
        raise ValueError(
            f"Mask registration shift {magnitude:.2f}px exceeds "
            f"{max_shift:.2f}px."
        )

    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    registered = cv2.warpAffine(
        target_mask.astype(np.uint8),
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if int(registered.sum()) == 0:
        raise ValueError("Mask registration moved the entire mask outside the image.")
    registration["applied"] = True
    return registered, registration


def resize_mask_to_image(mask, image):
    height, width = image.shape[:2]
    if mask.shape == (height, width):
        return mask.astype(np.uint8)
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def find_gerber_template(gerber_root, normal_record):
    if gerber_root is None:
        return None
    gerber_root = Path(gerber_root)
    if not gerber_root.exists():
        return None
    stem = f"gerber_normal_{normal_record.side}_{normal_record.frame}"
    candidates = []
    for split in ("train", "val", "test"):
        image_dir = gerber_root / "images" / split
        for suffix in (".png", ".jpg", ".jpeg"):
            candidates.append(image_dir / f"{stem}{suffix}")
    for suffix in (".png", ".jpg", ".jpeg"):
        candidates.append(gerber_root / f"{stem}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def clean_binary_mask(mask, min_area=8, close_size=3):
    mask = (mask > 0).astype(np.uint8)
    if close_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == component_id] = 1
    return cleaned


def write_rgb(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))


def next_output_index(directory):
    if not directory.exists():
        return 0
    max_index = -1
    for path in directory.glob("*.png"):
        match = re.match(r"(\d+)_", path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def compose_debug_image(ref_image, ref_mask, target_image, target_mask, generated, ref_processed):
    height, width = target_image.shape[:2]

    def prep_image(image):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = image * 255
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    tiles = [
        prep_image(ref_image),
        prep_image(ref_mask * 255),
        prep_image(ref_processed),
        prep_image(target_image),
        prep_image(target_mask * 255),
        prep_image(generated),
    ]
    grid = np.zeros((height * 2, width * 3, 3), dtype=np.uint8)
    for tile, (row, col) in zip(tiles, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]):
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = tile
    return grid


def parse_classes(value, class_mode):
    if not value:
        return set()
    classes = {normalize_label(part) for part in value.split(",") if part.strip()}
    if class_mode == "binary" and classes != {"defect"}:
        raise ValueError("Binary mode only supports class name 'defect'.")
    return classes


def target_mask_policy_for_label(args, label):
    override = getattr(args, "class_target_mask_policy_map", {}).get(label)
    if override is not None:
        return override
    if args.target_mask_policy != "structure":
        return "source_mask"
    structure_classes = set(getattr(args, "structure_class_set", []) or [])
    if structure_classes and label not in structure_classes:
        return "source_mask"
    return "structure"


# ---- cutpaste-only orchestration (slim replacement for the original main) ----
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="CutPaste-based synthetic defect generation for CustomPCB."
    )
    p.add_argument("--class-mode", choices=["class4", "binary"], default="class4")
    p.add_argument("--data-root", type=Path, default=None, required=True)
    p.add_argument("--mask-xml", type=Path, default=None, required=True)
    p.add_argument("--output-dir", type=Path, default=None, required=True)
    p.add_argument("--gerber-root", type=Path, default=None)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--defect-boards", default="")
    p.add_argument("--normal-boards", default="")
    p.add_argument("--exclude-boards", default="")
    p.add_argument("--classes", default="")
    p.add_argument("--output-suffix", default="")
    p.add_argument("--normal-source", choices=["num9", "all_normal", "gerber"], default="num9")
    p.add_argument("--normal-board-prefix", default="num9")
    p.add_argument("--placement", choices=["aligned", "same_side_random"], default="aligned")
    p.add_argument("--require-aligned-target", action="store_true")
    p.add_argument("--ref-policy", choices=["same_class", "any_class"], default="same_class")
    p.add_argument(
        "--ref-match-scope",
        choices=[
            "class", "same_sample", "class_only_distinct", "class_context",
            "class_side", "class_side_slice", "class_side_slice_preferred",
        ],
        default="class",
    )
    p.add_argument("--context-top-k", type=int, default=5)
    p.add_argument("--context-area-weight", type=float, default=0.25)
    p.add_argument("--combination-sampling", choices=["random", "unique_cycle"], default="random")
    p.add_argument("--no-combination-recycle", action="store_true")
    p.add_argument("--samples-per-class", type=int, default=20)
    p.add_argument("--mask-jitter-pixels", type=int, default=0)
    p.add_argument("--target-crop-ratio", nargs=2, type=float, default=[1.5, 3.0], metavar=("Y", "X"))
    p.add_argument("--crop-back-interp", choices=CROP_BACK_INTERP_CHOICES, default="bilinear")
    p.add_argument("--composite-mode", choices=["cutpaste"], default="cutpaste")
    p.add_argument("--cutpaste-placement", choices=["registered", "native_transfer"], default="registered")
    p.add_argument("--source-blend", choices=["feather", "poisson"], default="feather")
    p.add_argument("--source-paste-weight", type=float, default=0.5)
    p.add_argument("--defect-blend-sigma", type=float, default=0.0,
                   help="Gaussian feather sigma (px) for the paste boundary. 0 = hard paste.")
    p.add_argument("--defect-contrast-scale", type=float, default=1.0)
    p.add_argument("--defect-mask-dilate-range", nargs=2, type=int, default=[0, 0], metavar=("LO", "HI"))
    p.add_argument("--target-mask-policy", choices=["source_mask"], default="source_mask")
    p.add_argument("--mask-registration", choices=["none", "phase_translation"], default="none")
    p.add_argument("--registration-min-response", type=float, default=0.5)
    p.add_argument("--registration-max-shift", type=float, default=20.0)
    p.add_argument("--split-mask-components", action="store_true")
    p.add_argument("--min-component-area", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    # kept for command-line compatibility with the original pipeline (no effect here)
    p.add_argument("--disable-reference-augmentation", action="store_true")
    return p


def run_generation(args, records, normal_records):
    by_label = defaultdict(list)
    by_label_side = defaultdict(list)
    by_label_side_frame = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)
        by_label_side[(record.label, record.side)].append(record)
        by_label_side_frame[(record.label, record.side, record.frame)].append(record)
    labels = sorted(by_label)

    by_key, by_side = build_normal_index(normal_records)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    context_features = (
        build_context_feature_cache(records)
        if any_scope_uses_context(args, labels)
        else None
    )

    combination_pools = {}
    if args.combination_sampling == "unique_cycle":
        for label in labels:
            combination_pools[label] = build_combination_pool(
                label, labels, by_label, by_label_side, by_label_side_frame,
                by_key, by_side, normal_records, args, rng, context_features,
            )
            if not combination_pools[label]:
                raise RuntimeError(f"No valid generation combinations for class '{label}'.")
            print(f"[{label}] unique generation combinations: {len(combination_pools[label])}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    generation_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (output_dir / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2), encoding="ascii"
    )

    generated = 0
    with metadata_path.open("a", encoding="utf-8") as metadata_file:
        for label in labels:
            actual_mask_policy = target_mask_policy_for_label(args, label)
            target_count = resolve_samples_per_class(args, label)
            test_dir = output_dir / "test" / label
            mask_dir = output_dir / "ground_truth" / label
            source_dir = output_dir / "source" / label
            test_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            source_dir.mkdir(parents=True, exist_ok=True)

            failures = 0
            made_for_label = 0
            combination_index = 0
            combination_cycle = 0
            while made_for_label < target_count:
                if failures >= args.max_retries:
                    print(f"Stopped {label}: reached max retries.")
                    break

                if args.combination_sampling == "unique_cycle":
                    pool = combination_pools[label]
                    if combination_index >= len(pool):
                        if args.no_combination_recycle:
                            print(
                                f"Stopped {label}: exhausted {len(pool)} unique "
                                f"combinations ({made_for_label} generated)."
                            )
                            break
                        combination_cycle += 1
                        combination_index = 0
                        rng.shuffle(pool)
                    combination = pool[combination_index]
                    combination_index += 1
                    shape_record = combination.shape_record
                    ref_record = combination.ref_record
                    normal_record = combination.normal_record
                    placement_note = combination.placement_note
                    ref_match_note = combination.ref_match_note
                else:
                    combination_cycle = 0
                    shape_record = rng.choice(by_label[label])
                    ref_candidates, ref_match_note = reference_candidates_for_shape(
                        shape_record, labels, by_label, by_label_side, by_label_side_frame,
                        resolve_ref_match_scope(args, label), args.ref_policy, rng,
                        context_features, args.context_top_k, args.context_area_weight,
                    )
                    if not ref_candidates:
                        raise RuntimeError(f"No reference candidates for {ref_match_note}.")
                    ref_record = rng.choice(ref_candidates)
                    normal_candidates, placement_note = target_candidates_for_shape(
                        shape_record, by_key, by_side, normal_records,
                        args.placement, args.require_aligned_target,
                    )
                    if not normal_candidates:
                        failures += 1
                        print(f"[{label}] skipped {shape_record.image_name}: no aligned normal target")
                        continue
                    normal_record = rng.choice(normal_candidates)

                try:
                    target_image = read_rgb(normal_record.image_path)
                    ref_image = read_rgb(ref_record.image_path)
                    shape_image = (
                        ref_image
                        if shape_record.image_path == ref_record.image_path
                        else read_rgb(shape_record.image_path)
                    )
                    target_mask, registration = register_mask_to_target(
                        shape_image, target_image, shape_record.mask,
                        args.mask_registration, args.registration_min_response,
                        args.registration_max_shift,
                    )
                    target_mask = shift_mask(target_mask, resolve_mask_jitter(args, label), rng)

                    dilate_lo, dilate_hi = resolve_mask_dilate(args, label)
                    dilate_iters = 0
                    if dilate_hi > 0:
                        dilate_iters = rng.randint(dilate_lo, dilate_hi)
                        target_mask = dilate_defect_mask(target_mask, dilate_iters)

                    ref_image = read_rgb(ref_record.image_path)
                    ref_mask = resize_mask_to_image(ref_record.mask, ref_image)
                    result, cutpaste_mask = cutpaste_source_defect(
                        target_image, ref_image, ref_mask, target_mask,
                        args.source_blend, resolve_blend_sigma(args, label),
                        placement=resolve_cutpaste_placement(args, label),
                    )
                    if result is None:
                        raise ValueError("cutpaste geometry degenerate")
                    target_mask = cutpaste_mask
                except Exception as exc:
                    failures += 1
                    print(f"[{label}] skipped sample after error: {exc}")
                    continue

                blend_sigma = resolve_blend_sigma(args, label)
                contrast_scale = resolve_contrast_scale(args, label)

                index = next_output_index(test_dir)
                stem = (
                    f"{index:04d}_{normal_record.image_path.stem}"
                    f"_ref-{ref_record.image_path.stem}_shape-{shape_record.image_path.stem}"
                )
                image_path = test_dir / f"{stem}.png"
                mask_path = mask_dir / f"{stem}_mask.png"
                source_path = source_dir / f"source_{stem}.png"

                write_rgb(image_path, result)
                cv2.imwrite(str(mask_path), (target_mask * 255).astype(np.uint8))
                source = compose_debug_image(
                    ref_image, ref_mask, target_image, target_mask, result, ref_image
                )
                write_rgb(source_path, source)

                metadata_file.write(
                    json.dumps(
                        {
                            "label": label,
                            "image": str(image_path),
                            "mask": str(mask_path),
                            "source": str(source_path),
                            "reference": str(ref_record.image_path),
                            "shape_source": str(shape_record.image_path),
                            "normal_target": str(normal_record.image_path),
                            "placement": placement_note,
                            "ref_match_scope": args.ref_match_scope,
                            "resolved_ref_match_scope": resolve_ref_match_scope(args, label),
                            "mask_jitter_pixels": resolve_mask_jitter(args, label),
                            "defect_blend_sigma": blend_sigma,
                            "defect_contrast_scale": contrast_scale,
                            "defect_mask_dilate_iters": dilate_iters,
                            "composite_mode": "cutpaste",
                            "source_blend": args.source_blend,
                            "cutpaste_placement": resolve_cutpaste_placement(args, label),
                            "ref_match": ref_match_note,
                            "combination_sampling": args.combination_sampling,
                            "combination_cycle": combination_cycle,
                            "reference_board": ref_record.board,
                            "reference_instance": ref_record.instance_index,
                            "shape_board": shape_record.board,
                            "shape_instance": shape_record.instance_index,
                            "target_board": normal_record.board,
                            "side": shape_record.side,
                            "slice": shape_record.frame,
                            "target_mask_policy": actual_mask_policy,
                            "mask_registration": registration,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                metadata_file.flush()
                generated += 1
                made_for_label += 1
                print(f"[{label}/{actual_mask_policy}] generated {image_path}")

    print(f"Done. Generated {generated} synthetic defect images in {output_dir}.")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.registration_min_response < 0:
        parser.error("--registration-min-response must be non-negative.")
    if args.registration_max_shift < 0:
        parser.error("--registration-max-shift must be non-negative.")
    if args.min_component_area < 1:
        parser.error("--min-component-area must be at least 1.")
    if not 0.0 <= args.source_paste_weight <= 1.0:
        parser.error("--source-paste-weight must be in [0, 1].")
    if args.defect_blend_sigma < 0:
        parser.error("--defect-blend-sigma must be non-negative.")
    if args.defect_contrast_scale < 0:
        parser.error("--defect-contrast-scale must be non-negative.")
    if args.defect_mask_dilate_range[0] < 0 or args.defect_mask_dilate_range[1] < args.defect_mask_dilate_range[0]:
        parser.error("--defect-mask-dilate-range must satisfy 0 <= LO <= HI.")
    if args.target_crop_ratio[0] <= 0 or args.target_crop_ratio[1] <= 0:
        parser.error("--target-crop-ratio values must be positive.")

    args.data_root = args.data_root.resolve()
    args.mask_xml = args.mask_xml.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.gerber_root is not None:
        args.gerber_root = args.gerber_root.resolve()
    args.normal_board_prefix = args.normal_board_prefix.lower()

    exclude_boards = parse_board_list(args.exclude_boards)
    defect_boards = parse_board_list(args.defect_boards) - exclude_boards
    normal_boards = parse_board_list(args.normal_boards) - exclude_boards

    include_classes = parse_classes(args.classes, args.class_mode)
    if args.class_mode == "class4" and not include_classes:
        include_classes = set(CLASS4_LABELS)

    records, load_stats = load_defect_records(
        args.data_root, args.mask_xml, args.class_mode, args.splits, include_classes,
        split_components=args.split_mask_components, min_component_area=args.min_component_area,
    )
    records = [
        r for r in records
        if r.board not in exclude_boards and (not defect_boards or r.board in defect_boards)
    ]
    if not records:
        raise RuntimeError("No defect records remain after board filtering.")

    normal_records = load_normal_records(
        args.data_root, args.splits, args.normal_source, args.normal_board_prefix
    )
    normal_records = [
        r for r in normal_records
        if r.board not in exclude_boards and (not normal_boards or r.board in normal_boards)
    ]
    if not normal_records:
        raise RuntimeError("No normal records remain after board filtering.")

    print(f"Loaded {len(records)} defect records and {len(normal_records)} normal records.")
    run_generation(args, records, normal_records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
