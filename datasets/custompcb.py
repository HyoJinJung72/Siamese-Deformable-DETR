import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageEnhance, ImageFilter

from .coco import CocoDetection
from .robustness import (apply_eval_test_perturbation, build_eval_test_perturbation_args,
                         apply_eval_template_perturbation, build_eval_template_perturbation_args,
                         apply_train_template_misalignment, build_train_template_misalign_args)
import datasets.transforms as T


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

    if target is not None and 'suppression_masks' in target:
        masks = target['suppression_masks'].float()
        masks = TF.affine(
            masks, 0, [tx, ty], s, 0.0,
            interpolation=InterpolationMode.NEAREST)
        target['suppression_masks'] = masks > 0.5

    return image, template_image, target


def apply_gerber_template_augmentation(
        template_image, prob=1.0, line_width_prob=0.5, blur_radius=0.8,
        brightness=0.2, contrast=0.2, color=0.15, noise_std=0.02):
    """Make a clean Gerber rendering look closer to a camera-captured PCB template."""
    if random.random() > prob:
        return template_image

    image = template_image
    if line_width_prob > 0 and random.random() < line_width_prob:
        # MinFilter expands dark traces; MaxFilter thins them. This mimics rendering/capture width drift.
        image = image.filter(ImageFilter.MinFilter(3) if random.random() < 0.5 else ImageFilter.MaxFilter(3))

    if blur_radius > 0:
        radius = random.uniform(0.0, blur_radius)
        if radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

    if brightness > 0:
        factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if contrast > 0:
        factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
        image = ImageEnhance.Contrast(image).enhance(factor)
    if color > 0:
        factor = random.uniform(max(0.0, 1.0 - color), 1.0 + color)
        image = ImageEnhance.Color(image).enhance(factor)

    if noise_std > 0:
        tensor = TF.to_tensor(image)
        std = random.uniform(0.0, noise_std)
        tensor = (tensor + torch.randn_like(tensor) * std).clamp(0.0, 1.0)
        image = TF.to_pil_image(tensor)

    return image


class CustomPCBDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, use_template=False, apply_affine=False,
                 eval_test_perturbation=None, eval_template_perturbation=None, image_set='train', open_set_defect_mode='none',
                 open_set_holdout_class='', open_set_defect_name='defect',
                 gerber_template_aug=False, gerber_template_aug_prob=1.0,
                 gerber_aug_line_width_prob=0.5, gerber_aug_blur=0.8,
                 gerber_aug_brightness=0.2, gerber_aug_contrast=0.2,
                 gerber_aug_color=0.15, gerber_aug_noise_std=0.02,
                 template_ablation_mode='normal', template_shuffle_offset=1,
                 mask_annotation_path='', train_template_misalign=None):
        super().__init__(img_folder, ann_file, transforms=transforms, return_masks=return_masks)
        self.use_template = use_template
        self.apply_affine = apply_affine
        self.eval_test_perturbation = eval_test_perturbation
        self.eval_template_perturbation = eval_template_perturbation
        self.train_template_misalign = train_template_misalign
        self.image_set = image_set
        self.mask_annotations = self._load_mask_annotations(mask_annotation_path)
        self.gerber_template_aug = gerber_template_aug
        self.gerber_template_aug_params = {
            'prob': gerber_template_aug_prob,
            'line_width_prob': gerber_aug_line_width_prob,
            'blur_radius': gerber_aug_blur,
            'brightness': gerber_aug_brightness,
            'contrast': gerber_aug_contrast,
            'color': gerber_aug_color,
            'noise_std': gerber_aug_noise_std,
        }
        self.template_ablation_mode = template_ablation_mode
        self.template_shuffle_offset = max(int(template_shuffle_offset), 1)
        self.open_set_defect_mode = open_set_defect_mode
        if open_set_defect_mode != 'none':
            self._configure_open_set_defect(
                image_set=image_set,
                holdout_class=open_set_holdout_class,
                defect_name=open_set_defect_name,
            )
        category_ids = sorted(self.coco.getCatIds())
        self.category_id_to_model_label = {category_id: idx for idx, category_id in enumerate(category_ids)}
        self.model_label_to_category_id = {idx: category_id for category_id, idx in self.category_id_to_model_label.items()}
        self.coco.model_label_to_category_id = self.model_label_to_category_id

        for img_id in self.ids:
            img_info = self.coco.imgs[img_id]
            if 'file_name' not in img_info:
                group_image_list = img_info.get('group_image_list', [])
                if group_image_list:
                    img_info['file_name'] = group_image_list[0]
                else:
                    img_info['file_name'] = img_info.get('group_name', '') + '.jpg'

    @staticmethod
    def _resolve_optional_path(path):
        if not path:
            return None
        path = Path(path).expanduser()
        candidates = [path]
        if not path.is_absolute():
            candidates.extend([
                Path.cwd() / path,
                Path.cwd().parent / path,
                Path.home() / path,
            ])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f'Could not find mask annotation file: {path}')

    @staticmethod
    def _decode_cvat_rle(rle, width, height):
        counts = [int(value.strip()) for value in rle.split(',') if value.strip()]
        total = int(width) * int(height)
        flat = []
        value = 0
        for count in counts:
            if count > 0:
                flat.extend([value] * count)
            value = 1 - value
        if len(flat) < total:
            flat.extend([0] * (total - len(flat)))
        elif len(flat) > total:
            flat = flat[:total]
        return torch.as_tensor(flat, dtype=torch.uint8).view(int(height), int(width))

    def _load_mask_annotations(self, mask_annotation_path):
        resolved_path = self._resolve_optional_path(mask_annotation_path)
        if resolved_path is None:
            return {}

        root = ET.parse(resolved_path).getroot()
        annotations = {}
        num_masks = 0
        for image_node in root.findall('image'):
            image_name = Path(image_node.get('name', '')).name
            image_width = int(float(image_node.get('width', 0)))
            image_height = int(float(image_node.get('height', 0)))
            masks = []
            for mask_node in image_node.findall('mask'):
                rle = mask_node.get('rle', '')
                if not rle:
                    continue
                masks.append({
                    'rle': rle,
                    'left': int(float(mask_node.get('left', 0))),
                    'top': int(float(mask_node.get('top', 0))),
                    'width': int(float(mask_node.get('width', 0))),
                    'height': int(float(mask_node.get('height', 0))),
                    'label': mask_node.get('label', ''),
                })
            if masks:
                annotations[image_name] = {
                    'width': image_width,
                    'height': image_height,
                    'masks': masks,
                }
                num_masks += len(masks)

        print(f'Loaded {num_masks} CVAT segmentation masks from {resolved_path}')
        return annotations

    def _get_segmentation_masks(self, img_info, image_size):
        if not self.mask_annotations:
            return None

        image_name = Path(img_info.get('file_name', '')).name
        annotation = self.mask_annotations.get(image_name)
        if annotation is None:
            return None

        xml_width = annotation['width']
        xml_height = annotation['height']
        full_masks = []
        for mask_info in annotation['masks']:
            mask = torch.zeros((xml_height, xml_width), dtype=torch.uint8)
            crop = self._decode_cvat_rle(
                mask_info['rle'],
                mask_info['width'],
                mask_info['height'],
            )
            left = max(mask_info['left'], 0)
            top = max(mask_info['top'], 0)
            right = min(left + crop.shape[1], xml_width)
            bottom = min(top + crop.shape[0], xml_height)
            if right > left and bottom > top:
                mask[top:bottom, left:right] |= crop[:bottom - top, :right - left]
            full_masks.append(mask)

        if not full_masks:
            return None

        masks = torch.stack(full_masks, dim=0).bool()
        image_width, image_height = image_size
        if (xml_width, xml_height) != (image_width, image_height):
            masks = torch.nn.functional.interpolate(
                masks[:, None].float(),
                size=(image_height, image_width),
                mode='nearest',
            )[:, 0] > 0.5
        return masks

    def _resolve_open_set_holdout_id(self, holdout_class):
        if not holdout_class:
            raise ValueError('--open_set_holdout_class must be set when --open_set_defect_mode is enabled')
        categories = self.coco.dataset.get('categories', [])
        by_name = {str(cat['name']).lower(): int(cat['id']) for cat in categories}
        by_id = {str(int(cat['id'])): int(cat['id']) for cat in categories}
        key = str(holdout_class).strip()
        if key in by_id:
            return by_id[key]
        lowered = key.lower()
        if lowered in by_name:
            return by_name[lowered]
        valid = ', '.join(f"{cat['id']}:{cat['name']}" for cat in categories)
        raise ValueError(f'Unknown hold-out class {holdout_class!r}. Valid categories are: {valid}')

    def _configure_open_set_defect(self, image_set, holdout_class, defect_name):
        if self.open_set_defect_mode != 'holdout_defect':
            raise ValueError(f'Unsupported open-set defect mode: {self.open_set_defect_mode}')

        holdout_id = self._resolve_open_set_holdout_id(holdout_class)
        original_categories = list(self.coco.dataset.get('categories', []))
        known_categories = [dict(cat) for cat in original_categories if int(cat['id']) != holdout_id]
        defect_id = max(int(cat['id']) for cat in original_categories) + 1
        defect_category = {'id': defect_id, 'name': defect_name, 'supercategory': 'defect'}

        images = [dict(image) for image in self.coco.dataset.get('images', [])]
        annotations = [dict(ann) for ann in self.coco.dataset.get('annotations', [])]

        if image_set == 'train':
            holdout_image_ids = {
                int(ann['image_id']) for ann in annotations
                if int(ann['category_id']) == holdout_id
            }
            images = [image for image in images if int(image['id']) not in holdout_image_ids]
            valid_image_ids = {int(image['id']) for image in images}
            annotations = [
                ann for ann in annotations
                if int(ann['image_id']) in valid_image_ids and int(ann['category_id']) != holdout_id
            ]
        else:
            for ann in annotations:
                if int(ann['category_id']) == holdout_id:
                    ann['category_id'] = defect_id

        self.coco.dataset['images'] = images
        self.coco.dataset['annotations'] = annotations
        self.coco.dataset['categories'] = known_categories + [defect_category]
        self.coco.createIndex()
        self.ids = sorted(self.coco.imgs.keys())
        self.coco.open_set_holdout_category_id = holdout_id
        self.coco.open_set_defect_category_id = defect_id
        self.coco.open_set_defect_name = defect_name

    def _get_template_path(self, img_info):
        group_image_list = img_info.get('group_image_list', [])
        if len(group_image_list) > 1:
            template_path = Path(self.root) / group_image_list[1]
            if template_path.exists():
                return template_path

        image_name = img_info['file_name']
        template_name = image_name
        for i in range(1, 9):
            template_name = template_name.replace(f'num{i}', 'num9')
        template_name = template_name.replace('_defect_', '_normal_')

        candidates = [
            Path(self.root) / template_name,
            Path(self.root).parent / 'train' / template_name,
            Path(self.root).parent / 'val' / template_name,
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _get_shuffled_template_path(self, idx):
        if len(self.ids) <= 1:
            return None
        shuffled_idx = (idx + self.template_shuffle_offset) % len(self.ids)
        if shuffled_idx == idx:
            shuffled_idx = (idx + 1) % len(self.ids)
        shuffled_info = self.coco.loadImgs(self.ids[shuffled_idx])[0]
        return self._get_template_path(shuffled_info)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]

        template_img = None
        template_path = None
        if self.use_template:
            if self.template_ablation_mode == 'test_as_template':
                template_img = img.copy()
            elif self.template_ablation_mode == 'zero':
                template_img = Image.new('RGB', img.size, (124, 116, 104))
            else:
                if self.template_ablation_mode == 'shuffled':
                    template_path = self._get_shuffled_template_path(idx)
                else:
                    template_path = self._get_template_path(img_info)
                template_img = Image.open(template_path).convert('RGB') if template_path is not None else img.copy()
            if self.image_set == 'train' and self.gerber_template_aug and template_path is not None:
                template_img = apply_gerber_template_augmentation(
                    template_img, **self.gerber_template_aug_params)

        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        segmentation_masks = self._get_segmentation_masks(img_info, img.size)
        if segmentation_masks is not None:
            target['suppression_masks'] = segmentation_masks
        if 'labels' in target:
            labels = [self.category_id_to_model_label[int(label)] for label in target['labels'].tolist()]
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

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


def make_custompcb_transforms(image_set, disable_random_crop=False, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 기본 max_size=640 기준의 짧은변 스케일을 max_size 비율로 함께 상향한다.
    # (예: --input_max_size 1280 -> 짧은변 [512,640,768], val 768)
    f = max_size / 640.0
    scales = [int(round(s * f)) for s in (256, 320, 384)]
    val_scale = int(round(384 * f))
    crop_lo, crop_hi = int(round(128 * f)), int(round(256 * f))

    if image_set == 'train':
        if disable_random_crop:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize(scales, max_size=max_size),
                normalize,
            ])
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales),
                    T.RandomSizeCrop(crop_lo, crop_hi),
                    T.RandomResize([scales[-1]], max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([val_scale], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_custompcb_variant(image_set, args, path_attr, train_ann_name, val_ann_name):
    root = Path(getattr(args, path_attr, '') or args.coco_path)
    if not root.exists() and not root.is_absolute():
        parent_relative_root = Path.cwd().parent / root
        if parent_relative_root.exists():
            root = parent_relative_root
    assert root.exists(), f'provided CustomPCB path {root} does not exist'

    paths = {
        'train': (root / 'images' / 'train', root / 'annotations' / train_ann_name),
        'val': (root / 'images' / 'val', root / 'annotations' / val_ann_name),
    }
    img_folder, ann_file = paths[image_set]
    dataset = CustomPCBDetection(
        img_folder,
        ann_file,
        transforms=make_custompcb_transforms(
            image_set,
            disable_random_crop=getattr(args, 'disable_random_crop', False),
            max_size=(getattr(args, 'input_max_size', 0) or 640),
        ),
        return_masks=args.masks,
        use_template=getattr(args, 'use_template', False),
        apply_affine=(image_set == 'train' and getattr(args, 'use_affine', False)),
        eval_test_perturbation=build_eval_test_perturbation_args(args, image_set),
        eval_template_perturbation=build_eval_template_perturbation_args(args, image_set),
        train_template_misalign=build_train_template_misalign_args(args, image_set),
        image_set=image_set,
        open_set_defect_mode=getattr(args, 'open_set_defect_mode', 'none'),
        open_set_holdout_class=getattr(args, 'open_set_holdout_class', ''),
        open_set_defect_name=getattr(args, 'open_set_defect_name', 'defect'),
        gerber_template_aug=(image_set == 'train' and getattr(args, 'gerber_template_aug', False)),
        gerber_template_aug_prob=getattr(args, 'gerber_template_aug_prob', 1.0),
        gerber_aug_line_width_prob=getattr(args, 'gerber_aug_line_width_prob', 0.5),
        gerber_aug_blur=getattr(args, 'gerber_aug_blur', 0.8),
        gerber_aug_brightness=getattr(args, 'gerber_aug_brightness', 0.2),
        gerber_aug_contrast=getattr(args, 'gerber_aug_contrast', 0.2),
        gerber_aug_color=getattr(args, 'gerber_aug_color', 0.15),
        gerber_aug_noise_std=getattr(args, 'gerber_aug_noise_std', 0.02),
        template_ablation_mode=getattr(args, 'template_ablation_mode', 'normal'),
        template_shuffle_offset=getattr(args, 'template_shuffle_offset', 1),
        mask_annotation_path=getattr(args, 'custom_mask_annotation_path', ''),
    )
    return dataset


def build(image_set, args):
    return build_custompcb_variant(
        image_set,
        args,
        path_attr='custom_pcb_path',
        train_ann_name='custom_train.json',
        val_ann_name='custom_val.json',
    )
