# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TFRM(nn.Module):
    """Template Feature Rectification Module for projected feature maps."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 1)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.test_channel_attn = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )
        self.template_channel_attn = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()
        self.test_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.template_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        nn.init.constant_(self.test_proj.weight, 0)
        nn.init.constant_(self.test_proj.bias, 0)
        nn.init.constant_(self.template_proj.weight, 0)
        nn.init.constant_(self.template_proj.bias, 0)

    def forward(self, test_feat, template_feat):
        interaction = test_feat * template_feat
        avg_out = torch.mean(interaction, dim=1, keepdim=True)
        max_out, _ = torch.max(interaction, dim=1, keepdim=True)
        spatial_map = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))

        test_rect = test_feat * spatial_map
        template_rect = template_feat * spatial_map
        test_attn = self.channel_sigmoid(
            self.test_channel_attn(F.adaptive_avg_pool2d(test_rect, 1)) +
            self.test_channel_attn(F.adaptive_max_pool2d(test_rect, 1)))
        template_attn = self.channel_sigmoid(
            self.template_channel_attn(F.adaptive_avg_pool2d(template_rect, 1)) +
            self.template_channel_attn(F.adaptive_max_pool2d(template_rect, 1)))

        return self.test_proj(test_rect * test_attn), self.template_proj(template_rect * template_attn)


class TFFN(nn.Module):
    """Template Feature Fusion Network returning a residual delta."""
    def __init__(self, in_channels, use_diff=False):
        super().__init__()
        self.use_diff = use_diff
        input_channels = in_channels * (3 if use_diff else 2)
        self.fusion = nn.Conv2d(input_channels, in_channels, kernel_size=1)
        nn.init.constant_(self.fusion.weight, 0)
        nn.init.constant_(self.fusion.bias, 0)

    def forward(self, test_feat, template_feat):
        features = [test_feat, template_feat]
        if self.use_diff:
            features.append(torch.abs(test_feat - template_feat))
        return self.fusion(torch.cat(features, dim=1))


class ImageTransformModule(nn.Module):
    """Learnable Gerber-to-PCB image transform module with an identity start."""
    def __init__(self, in_channels=3, hidden_dim=32, dilation_rates=(1, 2, 4)):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=rate, dilation=rate),
                nn.ReLU(inplace=True),
            )
            for rate in dilation_rates
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * len(dilation_rates), hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
        )
        nn.init.constant_(self.fuse[-1].weight, 0)
        nn.init.constant_(self.fuse[-1].bias, 0)

    def forward(self, image):
        feat = self.entry(image)
        feat = torch.cat([branch(feat) for branch in self.branches], dim=1)
        return image + self.fuse(feat)


class FeatureSuppressionModule(nn.Module):
    """Spatially suppress non-defect template-test feature differences."""
    def __init__(self, in_channels):
        super().__init__()
        self.mask = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        nn.init.constant_(self.mask.weight, 0)
        nn.init.constant_(self.mask.bias, 4.0)

    def forward(self, test_feat, template_feat):
        diff = torch.abs(test_feat - template_feat)
        avg_out = diff.mean(dim=1, keepdim=True)
        max_out, _ = diff.max(dim=1, keepdim=True)
        return torch.sigmoid(self.mask(torch.cat([avg_out, max_out], dim=1)))


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                 use_template=False, use_tfrm=False, use_tffn_diff=False, use_residual_gate=False,
                 residual_gate_init=1.0, use_dual_tffn_fusion=False, dual_tffn_gate_init=-1.0,
                 template_reliability_mode='none',
                 template_reliability_min_weight=0.3, template_diff_topk_ratio=0.1,
                 template_diff_max_ratio=4.0, return_cam_features=False,
                 return_tffn_analysis=False, return_query_features=False,
                 template_feature_adapter='none', template_adapter_strength=1.0,
                 template_adapter_eps=1e-6, use_image_transform=False,
                 return_image_transform_pair=False, use_feature_suppression=False,
                 feature_suppression_floor=0.3):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_template = use_template
        self.use_tfrm = use_tfrm
        self.use_tffn_diff = use_tffn_diff
        self.use_residual_gate = use_residual_gate
        self.use_dual_tffn_fusion = use_dual_tffn_fusion
        self.template_reliability_mode = template_reliability_mode
        self.template_reliability_min_weight = template_reliability_min_weight
        self.template_diff_topk_ratio = template_diff_topk_ratio
        self.template_diff_max_ratio = template_diff_max_ratio
        self.return_cam_features = return_cam_features
        self.return_tffn_analysis = return_tffn_analysis
        self.return_query_features = return_query_features
        self.template_feature_adapter = template_feature_adapter
        self.template_adapter_strength = template_adapter_strength
        self.template_adapter_eps = template_adapter_eps
        self.use_image_transform = use_image_transform
        self.return_image_transform_pair = return_image_transform_pair
        self.use_feature_suppression = use_feature_suppression
        self.feature_suppression_floor = feature_suppression_floor
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        if self.use_image_transform:
            self.image_transform = ImageTransformModule()
        if self.use_template:
            if self.use_tfrm:
                self.tfrm_modules = nn.ModuleList([TFRM(hidden_dim) for _ in range(num_feature_levels)])
            self.tffn_modules = nn.ModuleList([TFFN(hidden_dim, use_diff=use_tffn_diff) for _ in range(num_feature_levels)])
            if self.use_feature_suppression:
                self.feature_suppression_modules = nn.ModuleList(
                    [FeatureSuppressionModule(hidden_dim) for _ in range(num_feature_levels)])
            if self.use_dual_tffn_fusion:
                if self.use_tffn_diff:
                    self.dual_tffn_basic_modules = nn.ModuleList(
                        [TFFN(hidden_dim, use_diff=False) for _ in range(num_feature_levels)])
                else:
                    self.dual_tffn_diff_modules = nn.ModuleList(
                        [TFFN(hidden_dim, use_diff=True) for _ in range(num_feature_levels)])
                if dual_tffn_gate_init < 0:
                    dual_tffn_gate_init = 0.95 if self.use_tffn_diff else 0.05
                dual_tffn_gate_init = min(max(float(dual_tffn_gate_init), 1e-4), 1.0 - 1e-4)
                dual_tffn_gate_logit = math.log(dual_tffn_gate_init / (1.0 - dual_tffn_gate_init))
                self.dual_tffn_gate_logits = nn.Parameter(
                    torch.full((num_feature_levels,), dual_tffn_gate_logit, dtype=torch.float))
            if self.use_residual_gate:
                if self.use_tfrm:
                    self.tfrm_gates = nn.Parameter(
                        torch.full((num_feature_levels,), residual_gate_init, dtype=torch.float))
                self.tffn_gates = nn.Parameter(
                    torch.full((num_feature_levels,), residual_gate_init, dtype=torch.float))

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def _compute_diff_reliability(self, test_feat, template_feat):
        diff_map = torch.abs(test_feat - template_feat).mean(dim=1).flatten(1)
        num_positions = diff_map.shape[1]
        if num_positions == 0:
            return torch.ones(test_feat.shape[0], device=test_feat.device, dtype=test_feat.dtype)

        k = max(1, int(num_positions * self.template_diff_topk_ratio))
        topk_mean = diff_map.topk(k, dim=1).values.mean(dim=1)
        global_mean = diff_map.mean(dim=1)
        concentration = topk_mean / (global_mean + 1e-6)
        reliability = (concentration - 1.0) / max(self.template_diff_max_ratio - 1.0, 1e-6)
        reliability = reliability.clamp(0.0, 1.0)
        min_weight = self.template_reliability_min_weight
        reliability = min_weight + (1.0 - min_weight) * reliability
        return reliability.detach()

    def _adapt_template_feature(self, test_feat, template_feat):
        if self.template_feature_adapter == 'none':
            return template_feat
        if self.template_feature_adapter != 'adain':
            raise ValueError(f'Unsupported template feature adapter: {self.template_feature_adapter}')

        eps = self.template_adapter_eps
        test_mean = test_feat.mean(dim=(-2, -1), keepdim=True)
        test_std = test_feat.var(dim=(-2, -1), keepdim=True, unbiased=False).add(eps).sqrt()
        template_mean = template_feat.mean(dim=(-2, -1), keepdim=True)
        template_std = template_feat.var(dim=(-2, -1), keepdim=True, unbiased=False).add(eps).sqrt()
        aligned_template = (template_feat - template_mean) / template_std * test_std + test_mean
        strength = min(max(float(self.template_adapter_strength), 0.0), 1.0)
        return template_feat * (1.0 - strength) + aligned_template * strength

    def _fuse_template_feature(self, level, test_feat, template_feat):
        if template_feat.shape[-2:] != test_feat.shape[-2:]:
            template_feat = F.interpolate(
                template_feat,
                size=test_feat.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        template_feat = self._adapt_template_feature(test_feat, template_feat)
        suppression_mask = None
        if self.use_feature_suppression:
            suppression_mask = self.feature_suppression_modules[level](test_feat, template_feat)
            gate = self.feature_suppression_floor + (1.0 - self.feature_suppression_floor) * suppression_mask
            template_feat = test_feat + (template_feat - test_feat) * gate
        if self.use_tfrm:
            test_delta, template_delta = self.tfrm_modules[level](test_feat, template_feat)
            if self.use_residual_gate:
                tfrm_gate = self.tfrm_gates[level].view(1, 1, 1, 1)
                test_delta = tfrm_gate * test_delta
                template_delta = tfrm_gate * template_delta
            test_feat = test_feat + test_delta
            template_feat = template_feat + template_delta

        reliability = None
        if self.template_reliability_mode == 'diff':
            reliability = self._compute_diff_reliability(test_feat, template_feat)

        diff_feat = torch.abs(test_feat - template_feat)
        if self.use_dual_tffn_fusion:
            diff_weight = torch.sigmoid(self.dual_tffn_gate_logits[level]).view(1, 1, 1, 1)
            if self.use_tffn_diff:
                diff_delta = self.tffn_modules[level](test_feat, template_feat)
                basic_delta = self.dual_tffn_basic_modules[level](test_feat, template_feat)
            else:
                basic_delta = self.tffn_modules[level](test_feat, template_feat)
                diff_delta = self.dual_tffn_diff_modules[level](test_feat, template_feat)
            fusion_delta = (1.0 - diff_weight) * basic_delta + diff_weight * diff_delta
        else:
            fusion_delta = self.tffn_modules[level](test_feat, template_feat)
        if self.use_residual_gate:
            fusion_delta = self.tffn_gates[level].view(1, 1, 1, 1) * fusion_delta
        fused_feat = test_feat + fusion_delta

        analysis = None
        if self.return_tffn_analysis:
            analysis = {
                'test': test_feat.detach(),
                'template': template_feat.detach(),
                'diff': diff_feat.detach(),
                'delta': fusion_delta.detach(),
                'fused': fused_feat.detach(),
            }
            if self.use_dual_tffn_fusion:
                analysis['delta_context'] = basic_delta.detach()   # Δ_context (A분기)
                analysis['delta_diff'] = diff_delta.detach()        # Δ_diff   (B분기)
                analysis['dual_diff_weight'] = diff_weight.detach()
            if suppression_mask is not None:
                analysis['suppression_mask'] = suppression_mask.detach()

        return fused_feat, reliability, analysis, suppression_mask

    def forward(self, samples: NestedTensor, template_samples: NestedTensor = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        use_template = self.use_template and template_samples is not None
        if use_template and not isinstance(template_samples, NestedTensor):
            template_samples = nested_tensor_from_tensor_list(template_samples)
        image_transform_pair = None
        if use_template and self.use_image_transform:
            transformed_template = self.image_transform(template_samples.tensors)
            if self.return_image_transform_pair:
                image_transform_pair = {
                    'test': samples.tensors.detach(),
                    'transformed_template': transformed_template,
                    'padding_mask': samples.mask,
                }
            template_samples = NestedTensor(transformed_template, template_samples.mask)
        if use_template:
            features, pos, template_features = self.backbone(samples, template_samples)
        else:
            features, pos = self.backbone(samples)
            template_features = None

        srcs = []
        masks = []
        template_reliability = []
        cam_features = []
        tffn_analysis = []
        feature_suppression_masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src = self.input_proj[l](src)
            if use_template:
                template_src, _ = template_features[l].decompose()
                template_src = self.input_proj[l](template_src)
                src, reliability, analysis, suppression_mask = self._fuse_template_feature(l, src, template_src)
                if reliability is not None:
                    template_reliability.append(reliability)
                if analysis is not None:
                    tffn_analysis.append(analysis)
                if suppression_mask is not None:
                    feature_suppression_masks.append(suppression_mask)
            srcs.append(src)
            if self.return_cam_features:
                cam_features.append(src.detach())
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            template_src = None
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                    if use_template:
                        template_src = self.input_proj[l](template_features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                    if use_template:
                        template_src = self.input_proj[l](template_src)
                if use_template:
                    src, reliability, analysis, suppression_mask = self._fuse_template_feature(l, src, template_src)
                    if reliability is not None:
                        template_reliability.append(reliability)
                    if analysis is not None:
                        tffn_analysis.append(analysis)
                    if suppression_mask is not None:
                        feature_suppression_masks.append(suppression_mask)
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                if self.return_cam_features:
                    cam_features.append(src.detach())
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.return_query_features:
            out['query_features'] = hs[-1]
        if self.return_cam_features:
            out['cam_features'] = cam_features
        if tffn_analysis:
            out['tffn_analysis'] = tffn_analysis
        if template_reliability:
            out['template_reliability'] = torch.stack(template_reliability, dim=0).mean(dim=0)
        if image_transform_pair is not None:
            out['image_transform_pair'] = image_transform_pair
        if feature_suppression_masks:
            out['feature_suppression_masks'] = feature_suppression_masks
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25,
                 generic_defect_label=None, boundary_loss_focus=0.0,
                 query_contrast_temperature=0.1,
                 box_query_contrast_temperature=0.1,
                 box_query_contrast_pos_iou=0.6,
                 box_query_contrast_neg_iou=0.2,
                 box_query_contrast_neg_topk=50,
                 box_query_contrast_positive_mode='class'):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.generic_defect_label = generic_defect_label
        self.boundary_loss_focus = boundary_loss_focus
        self.query_contrast_temperature = query_contrast_temperature
        self.box_query_contrast_temperature = box_query_contrast_temperature
        self.box_query_contrast_pos_iou = box_query_contrast_pos_iou
        self.box_query_contrast_neg_iou = box_query_contrast_neg_iou
        self.box_query_contrast_neg_topk = box_query_contrast_neg_topk
        self.box_query_contrast_positive_mode = box_query_contrast_positive_mode

    def _zero_query_loss(self, outputs):
        if 'query_features' in outputs:
            return outputs['query_features'].sum() * 0.0
        return outputs['pred_logits'].sum() * 0.0

    def _matched_weights(self, idx, sample_weights):
        if sample_weights is None:
            return None
        batch_idx, src_idx = idx
        if sample_weights.dim() == 2:
            return sample_weights[batch_idx, src_idx]
        return sample_weights[batch_idx]

    def _supervised_contrastive_loss(self, features, labels, temperature, anchor_mask=None, anchor_weights=None):
        if features.shape[0] < 2:
            return features.sum() * 0.0

        labels = labels.to(features.device)
        if anchor_mask is None:
            anchor_mask = labels >= 0
        else:
            anchor_mask = anchor_mask.to(features.device) & (labels >= 0)

        features = F.normalize(features, dim=-1)
        logits = torch.matmul(features, features.t()) / max(float(temperature), 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = ~torch.eye(labels.shape[0], dtype=torch.bool, device=features.device)
        valid_label = labels >= 0
        positive_mask = (
            labels[:, None].eq(labels[None, :]) &
            valid_label[:, None] &
            valid_label[None, :] &
            self_mask
        )
        exp_logits = torch.exp(logits) * self_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-6))
        positive_count = positive_mask.sum(dim=1)
        valid_anchor = anchor_mask & (positive_count > 0)
        if not valid_anchor.any():
            return features.sum() * 0.0

        anchor_loss = -(log_prob * positive_mask.float()).sum(dim=1) / positive_count.clamp_min(1)
        if anchor_weights is not None:
            anchor_weights = anchor_weights.to(features.device).float()
            weights = anchor_weights[valid_anchor].clamp_min(0.0)
            return (anchor_loss[valid_anchor] * weights).sum() / weights.sum().clamp_min(1e-6)
        return anchor_loss[valid_anchor].mean()

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, sample_weights=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        if self.generic_defect_label is not None and idx[0].numel() > 0:
            defect_label = int(self.generic_defect_label)
            if 0 <= defect_label < src_logits.shape[2]:
                target_classes_onehot[idx[0], idx[1], defect_label] = 1

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        if sample_weights is None:
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        elif sample_weights.dim() == 2:
            prob = src_logits.sigmoid()
            ce_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
            p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot)
            loss_ce = ce_loss * ((1 - p_t) ** 2)
            if self.focal_alpha >= 0:
                alpha_t = self.focal_alpha * target_classes_onehot + (1 - self.focal_alpha) * (1 - target_classes_onehot)
                loss_ce = alpha_t * loss_ce
            per_query_loss = loss_ce.sum(-1)
            loss_ce = (per_query_loss * sample_weights).sum() / num_boxes
        else:
            prob = src_logits.sigmoid()
            ce_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
            p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot)
            loss_ce = ce_loss * ((1 - p_t) ** 2)
            if self.focal_alpha >= 0:
                alpha_t = self.focal_alpha * target_classes_onehot + (1 - self.focal_alpha) * (1 - target_classes_onehot)
                loss_ce = alpha_t * loss_ce
            per_sample_loss = loss_ce.mean(1).sum(-1) * src_logits.shape[1]
            loss_ce = (per_sample_loss * sample_weights).sum() / num_boxes
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, sample_weights=None):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, sample_weights=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        if sample_weights is None:
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        else:
            batch_idx, _ = idx
            if sample_weights.dim() == 2:
                _, src_idx = idx
                matched_weights = sample_weights[batch_idx, src_idx]
            else:
                matched_weights = sample_weights[batch_idx]
            losses['loss_bbox'] = (loss_bbox.sum(dim=1) * matched_weights).sum() / num_boxes

        src_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            src_boxes_xyxy,
            target_boxes_xyxy))
        if sample_weights is None:
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            losses['loss_giou'] = (loss_giou * matched_weights).sum() / num_boxes
        if self.weight_dict.get('loss_boundary', 0) > 0:
            loss_boundary = F.l1_loss(src_boxes_xyxy, target_boxes_xyxy, reduction='none').sum(dim=1)
            if self.boundary_loss_focus > 0 and loss_boundary.numel() > 0:
                with torch.no_grad():
                    matched_iou = torch.diag(box_ops.box_iou(src_boxes_xyxy, target_boxes_xyxy)[0]).clamp(0.0, 1.0)
                    boundary_focus = 1.0 + self.boundary_loss_focus * (1.0 - matched_iou)
                loss_boundary = loss_boundary * boundary_focus
            if sample_weights is None:
                losses['loss_boundary'] = loss_boundary.sum() / num_boxes
            else:
                losses['loss_boundary'] = (loss_boundary * matched_weights).sum() / num_boxes
        return losses

    def loss_query_contrast(self, outputs, targets, indices, sample_weights=None):
        if 'query_features' not in outputs:
            return {'loss_query_contrast': self._zero_query_loss(outputs)}

        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() < 2:
            return {'loss_query_contrast': self._zero_query_loss(outputs)}

        query_features = outputs['query_features'][idx]
        labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        anchor_weights = self._matched_weights(idx, sample_weights)
        loss = self._supervised_contrastive_loss(
            query_features,
            labels,
            self.query_contrast_temperature,
            anchor_weights=anchor_weights,
        )
        return {'loss_query_contrast': loss}

    def loss_box_query_contrast(self, outputs, targets, sample_weights=None):
        if 'query_features' not in outputs:
            return {'loss_box_query_contrast': self._zero_query_loss(outputs)}

        query_features = outputs['query_features']
        pred_logits = outputs['pred_logits'].detach()
        pred_boxes = outputs['pred_boxes'].detach()
        all_features = []
        all_labels = []
        all_anchor_masks = []
        all_anchor_weights = []
        gt_offset = 0

        for batch_idx, target in enumerate(targets):
            if len(target['boxes']) == 0:
                continue

            pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[batch_idx])
            target_xyxy = box_ops.box_cxcywh_to_xyxy(target['boxes'])
            iou, _ = box_ops.box_iou(pred_xyxy, target_xyxy)
            max_iou, gt_idx = iou.max(dim=1)
            pos_mask = max_iou >= self.box_query_contrast_pos_iou
            neg_mask = max_iou <= self.box_query_contrast_neg_iou

            objectness = pred_logits[batch_idx].sigmoid().max(dim=-1).values
            neg_indices = neg_mask.nonzero(as_tuple=False).flatten()
            if self.box_query_contrast_neg_topk > 0 and neg_indices.numel() > self.box_query_contrast_neg_topk:
                neg_scores = objectness[neg_indices]
                neg_indices = neg_indices[neg_scores.topk(self.box_query_contrast_neg_topk).indices]
            pos_indices = pos_mask.nonzero(as_tuple=False).flatten()
            selected = torch.cat([pos_indices, neg_indices], dim=0)
            if selected.numel() == 0:
                continue

            labels = torch.full(
                (selected.numel(),), -1, dtype=torch.long, device=query_features.device)
            selected_pos = pos_mask[selected]
            if selected_pos.any():
                if self.box_query_contrast_positive_mode == 'gt':
                    labels[selected_pos] = gt_idx[selected[selected_pos]] + gt_offset
                else:
                    labels[selected_pos] = target['labels'][gt_idx[selected[selected_pos]]]
            anchor_mask = selected_pos
            if sample_weights is None:
                anchor_weights = max_iou[selected].to(query_features.device)
            elif sample_weights.dim() == 2:
                anchor_weights = sample_weights[batch_idx, selected].to(query_features.device)
            else:
                anchor_weights = sample_weights[batch_idx].expand(selected.numel()).to(query_features.device)
            anchor_weights = anchor_weights * max_iou[selected].to(query_features.device).clamp_min(0.0)

            all_features.append(query_features[batch_idx, selected])
            all_labels.append(labels)
            all_anchor_masks.append(anchor_mask.to(query_features.device))
            all_anchor_weights.append(anchor_weights)
            gt_offset += len(target['boxes'])

        if not all_features:
            return {'loss_box_query_contrast': self._zero_query_loss(outputs)}

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        anchor_mask = torch.cat(all_anchor_masks, dim=0)
        anchor_weights = torch.cat(all_anchor_weights, dim=0)
        loss = self._supervised_contrastive_loss(
            features,
            labels,
            self.box_query_contrast_temperature,
            anchor_mask=anchor_mask,
            anchor_weights=anchor_weights,
        )
        return {'loss_box_query_contrast': loss}

    def loss_masks(self, outputs, targets, indices, num_boxes, sample_weights=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, sample_weights=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v for k, v in outputs.items()
            if k not in ['aux_outputs', 'enc_outputs', 'cam_features',
                         'tffn_analysis', 'template_reliability',
                         'query_features', 'image_transform_pair',
                         'feature_suppression_masks']
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        if sample_weights is not None:
            sample_weights = sample_weights.detach().to(outputs['pred_logits'].device)
            if sample_weights.dim() == 2:
                matched_weight_sums = []
                for i, (src_idx, _) in enumerate(indices):
                    if src_idx.numel() > 0:
                        matched_weight_sums.append(sample_weights[i, src_idx].sum())
                    else:
                        matched_weight_sums.append(sample_weights.sum() * 0.0)
                num_boxes = torch.stack(matched_weight_sums).sum()
            else:
                num_boxes = torch.stack([
                    sample_weights[i] * len(t["labels"]) for i, t in enumerate(targets)
                ]).sum()
            num_boxes = num_boxes[None]
        else:
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        def compatible_sample_weights(output_dict):
            if sample_weights is None:
                return None
            if sample_weights.dim() == 2 and sample_weights.shape[1] != output_dict['pred_logits'].shape[1]:
                return None
            return sample_weights

        for loss in self.losses:
            kwargs = {'sample_weights': compatible_sample_weights(outputs)}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        final_sample_weights = compatible_sample_weights(outputs)
        if self.weight_dict.get('loss_query_contrast', 0) > 0:
            losses.update(self.loss_query_contrast(
                outputs, targets, indices, sample_weights=final_sample_weights))
        if self.weight_dict.get('loss_box_query_contrast', 0) > 0:
            losses.update(self.loss_box_query_contrast(
                outputs, targets, sample_weights=final_sample_weights))
        if self.weight_dict.get('loss_image_transform', 0) > 0:
            losses.update(self.loss_image_transform(outputs, targets))
        if self.weight_dict.get('loss_feature_suppression', 0) > 0:
            losses.update(self.loss_feature_suppression(outputs, targets))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {'sample_weights': compatible_sample_weights(aux_outputs)}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {'sample_weights': compatible_sample_weights(enc_outputs)}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def _defect_region_masks(self, targets, size, device, dtype):
        h, w = size
        masks = torch.zeros((len(targets), 1, h, w), device=device, dtype=dtype)
        for batch_idx, target in enumerate(targets):
            target_segmentation = target.get('suppression_mask', None)
            if target_segmentation is None:
                target_segmentation = target.get('suppression_masks', None)
            if target_segmentation is not None:
                target_segmentation = target_segmentation.to(device)
                if target_segmentation.dim() == 2:
                    target_segmentation = target_segmentation.unsqueeze(0)
                if target_segmentation.numel() > 0:
                    target_segmentation = target_segmentation.to(dtype).amax(dim=0, keepdim=True)
                    target_segmentation = F.interpolate(
                        target_segmentation.unsqueeze(0),
                        size=(h, w),
                        mode='nearest',
                    )[0]
                    masks[batch_idx] = (target_segmentation > 0.5).to(dtype)
                    continue

            if len(target['boxes']) == 0:
                continue
            boxes = box_ops.box_cxcywh_to_xyxy(target['boxes'].to(device)).clamp(0.0, 1.0)
            x0 = torch.floor(boxes[:, 0] * w).long().clamp(0, w)
            y0 = torch.floor(boxes[:, 1] * h).long().clamp(0, h)
            x1 = torch.ceil(boxes[:, 2] * w).long().clamp(0, w)
            y1 = torch.ceil(boxes[:, 3] * h).long().clamp(0, h)
            for left, top, right, bottom in zip(x0, y0, x1, y1):
                if right > left and bottom > top:
                    masks[batch_idx, :, top:bottom, left:right] = 1.0
        return masks

    def loss_image_transform(self, outputs, targets):
        pair = outputs.get('image_transform_pair', None)
        if pair is None:
            return {'loss_image_transform': self._zero_query_loss(outputs)}

        transformed = pair['transformed_template']
        test = pair['test'].to(transformed.device)
        h, w = test.shape[-2:]
        if transformed.shape[-2:] != (h, w):
            transformed = F.interpolate(
                transformed,
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            )
        defect_mask = self._defect_region_masks(targets, (h, w), transformed.device, transformed.dtype)
        valid_mask = torch.ones_like(defect_mask)
        padding_mask = pair.get('padding_mask', None)
        if padding_mask is not None:
            valid_mask = (~padding_mask).to(transformed.device).to(transformed.dtype).unsqueeze(1)
            if valid_mask.shape[-2:] != (h, w):
                valid_mask = F.interpolate(valid_mask, size=(h, w), mode='nearest')
        normal_mask = (1.0 - defect_mask) * valid_mask
        pixel_loss = (transformed - test.detach()).pow(2).mean(dim=1, keepdim=True)
        loss = (pixel_loss * normal_mask).sum() / normal_mask.sum().clamp_min(1.0)
        return {'loss_image_transform': loss}

    def loss_feature_suppression(self, outputs, targets):
        masks = outputs.get('feature_suppression_masks', None)
        if not masks:
            return {'loss_feature_suppression': self._zero_query_loss(outputs)}

        losses = []
        for mask in masks:
            target_mask = self._defect_region_masks(targets, mask.shape[-2:], mask.device, mask.dtype)
            bce = F.binary_cross_entropy(mask.clamp(1e-4, 1.0 - 1e-4), target_mask, reduction='none')
            pos = target_mask > 0.5
            neg = ~pos
            if pos.any():
                pos_loss = bce[pos].mean()
                neg_loss = bce[neg].mean() if neg.any() else bce.sum() * 0.0
                losses.append(0.5 * (pos_loss + neg_loss))
            elif neg.any():
                losses.append(bce[neg].mean())
            else:
                losses.append(bce.sum() * 0.0)
        return {'loss_feature_suppression': torch.stack(losses).mean()}


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, generic_defect_label=None, known_score_thresh=0.5, defect_score_thresh=0.3,
                 known_margin_thresh=0.15, defect_score_margin=0.2):
        super().__init__()
        self.generic_defect_label = generic_defect_label
        self.known_score_thresh = known_score_thresh
        self.defect_score_thresh = defect_score_thresh
        self.known_margin_thresh = known_margin_thresh
        self.defect_score_margin = defect_score_margin

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images in the batch
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if self.generic_defect_label is not None:
            defect_label = int(self.generic_defect_label)
            if defect_label <= 0 or defect_label >= prob.shape[-1]:
                raise ValueError(
                    f'generic_defect_label={defect_label} is incompatible with '
                    f'{prob.shape[-1]} output classes'
                )
            known_prob = prob[..., :defect_label]
            known_scores, known_labels = known_prob.max(dim=-1)
            defect_scores = prob[..., defect_label]
            if known_prob.shape[-1] > 1:
                top2_known_scores = known_prob.topk(2, dim=-1).values
                known_margins = top2_known_scores[..., 0] - top2_known_scores[..., 1]
            else:
                known_margins = torch.ones_like(known_scores)

            low_known_confidence = known_scores < self.known_score_thresh
            ambiguous_known_class = known_margins < self.known_margin_thresh
            defect_dominant = defect_scores >= (known_scores + self.defect_score_margin)
            unknown_mask = (
                (defect_scores >= self.defect_score_thresh) &
                (low_known_confidence | ambiguous_known_class | defect_dominant)
            )
            query_scores = torch.where(unknown_mask, defect_scores, known_scores)
            query_labels = torch.where(
                unknown_mask,
                torch.full_like(known_labels, defect_label),
                known_labels,
            )
            topk = min(100, query_scores.shape[1])
            scores, topk_boxes = torch.topk(query_scores, topk, dim=1)
            labels = torch.gather(query_labels, 1, topk_boxes)
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        else:
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    use_template = getattr(args, 'use_template', False)
    use_tfrm = getattr(args, 'use_tfrm', False)
    use_tffn_diff = getattr(args, 'use_tffn_diff', False)
    use_residual_gate = getattr(args, 'use_residual_gate', False)
    use_dual_tffn_fusion = getattr(args, 'use_tcdf', False)
    residual_gate_init = getattr(args, 'residual_gate_init', 1.0)
    template_reliability_mode = getattr(args, 'template_reliability_mode', 'none')
    use_image_transform = getattr(args, 'use_image_transform', False)
    use_feature_suppression = getattr(args, 'use_feature_suppression', False)
    use_query_contrast = (
        getattr(args, 'query_contrast_loss_coef', 0.0) > 0 or
        getattr(args, 'box_query_contrast_loss_coef', 0.0) > 0
    )
    if args.masks and use_template:
        raise ValueError("Template fusion is implemented for detection mode; disable --masks to use TFRM/TFFN.")
    if use_residual_gate and not use_template:
        raise ValueError("--use_residual_gate requires --use_template.")
    if use_dual_tffn_fusion and not use_template:
        raise ValueError("--use_tcdf requires --use_template.")
    if template_reliability_mode != 'none' and not use_template:
        raise ValueError("--template_reliability_mode requires --use_template.")
    if use_image_transform and not use_template:
        raise ValueError("--use_image_transform requires --use_template.")
    if use_feature_suppression and not use_template:
        raise ValueError("--use_feature_suppression requires --use_template.")
    if args.dataset_file == 'coco':
        num_classes = 91
    elif args.dataset_file in ['deeppcb', 'DeepPCB']:
        num_classes = 7
    elif args.dataset_file in ['custompcb', 'CustomPCB']:
        num_classes = 1
    elif args.dataset_file in ['custompcb_class4', 'CustomPCB_class4',
                               'custompcb_class4_full', 'CustomPCB_class4_full']:
        num_classes = 4
    elif args.dataset_file in ['custompcb_gerber', 'CustomPCB_gerber', 'custompcb_full', 'CustomPCB_full']:
        num_classes = 1
    else:
        num_classes = 20
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        use_template=use_template,
        use_tfrm=use_tfrm,
        use_tffn_diff=use_tffn_diff,
        use_residual_gate=use_residual_gate,
        residual_gate_init=residual_gate_init,
        use_dual_tffn_fusion=use_dual_tffn_fusion,
        dual_tffn_gate_init=getattr(args, 'dual_tffn_gate_init', -1.0),
        template_reliability_mode=template_reliability_mode,
        template_reliability_min_weight=getattr(args, 'template_reliability_min_weight', 0.3),
        template_diff_topk_ratio=getattr(args, 'template_diff_topk_ratio', 0.1),
        template_diff_max_ratio=getattr(args, 'template_diff_max_ratio', 4.0),
        return_cam_features=getattr(args, 'save_eval_cam', False),
        return_tffn_analysis=getattr(args, 'save_eval_tffn_analysis', False),
        return_query_features=use_query_contrast,
        template_feature_adapter=getattr(args, 'template_feature_adapter', 'none'),
        template_adapter_strength=getattr(args, 'template_adapter_strength', 1.0),
        use_image_transform=use_image_transform,
        return_image_transform_pair=getattr(args, 'image_transform_loss_coef', 0.0) > 0,
        use_feature_suppression=use_feature_suppression,
        feature_suppression_floor=getattr(args, 'feature_suppression_floor', 0.3),
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if getattr(args, 'boundary_loss_coef', 0.0) > 0:
        weight_dict['loss_boundary'] = args.boundary_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    if getattr(args, 'query_contrast_loss_coef', 0.0) > 0:
        weight_dict['loss_query_contrast'] = args.query_contrast_loss_coef
    if getattr(args, 'box_query_contrast_loss_coef', 0.0) > 0:
        weight_dict['loss_box_query_contrast'] = args.box_query_contrast_loss_coef
    if getattr(args, 'image_transform_loss_coef', 0.0) > 0:
        weight_dict['loss_image_transform'] = args.image_transform_loss_coef
    if getattr(args, 'feature_suppression_loss_coef', 0.0) > 0:
        weight_dict['loss_feature_suppression'] = args.feature_suppression_loss_coef
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    generic_defect_label = None
    if getattr(args, 'open_set_defect_mode', 'none') != 'none':
        generic_defect_label = num_classes - 1
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(
        num_classes, matcher, weight_dict, losses,
        focal_alpha=args.focal_alpha,
        generic_defect_label=generic_defect_label,
        boundary_loss_focus=getattr(args, 'boundary_loss_focus', 0.0),
        query_contrast_temperature=getattr(args, 'query_contrast_temperature', 0.1),
        box_query_contrast_temperature=getattr(args, 'box_query_contrast_temperature', 0.1),
        box_query_contrast_pos_iou=getattr(args, 'box_query_contrast_pos_iou', 0.6),
        box_query_contrast_neg_iou=getattr(args, 'box_query_contrast_neg_iou', 0.2),
        box_query_contrast_neg_topk=getattr(args, 'box_query_contrast_neg_topk', 50),
        box_query_contrast_positive_mode=getattr(args, 'box_query_contrast_positive_mode', 'class'),
    )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(
        generic_defect_label=generic_defect_label,
        known_score_thresh=getattr(args, 'open_set_known_score_thresh', 0.5),
        defect_score_thresh=getattr(args, 'open_set_defect_score_thresh', 0.3),
        known_margin_thresh=getattr(args, 'open_set_known_margin_thresh', 0.15),
        defect_score_margin=getattr(args, 'open_set_defect_score_margin', 0.2),
    )}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
