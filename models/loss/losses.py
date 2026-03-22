"""
Loss functions for DSITT.
Implements:
- Focal Loss for classification
- L1 Loss + GIoU Loss / NWD Loss for bounding box regression
- Collective Average Loss (CAL) for multi-frame normalization

Reference: DETR, Deformable DETR, MOTR
NWD Reference: A Normalized Gaussian Wasserstein Distance for Tiny Object Detection (AAAI 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .nwd_loss import nwd_loss


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0):
    """
    Sigmoid focal loss for classification.

    Args:
        inputs: [N, C] predicted logits
        targets: [N, C] one-hot targets
        num_boxes: normalization factor
        alpha: weighting factor
        gamma: focusing parameter
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def generalized_box_iou_loss(boxes1, boxes2):
    """
    Compute GIoU loss between paired boxes.

    Args:
        boxes1: [N, 4] (cx, cy, w, h) predicted
        boxes2: [N, 4] (cx, cy, w, h) targets

    Returns:
        giou_loss: scalar
    """
    # Convert to x1y1x2y2
    b1 = torch.stack([
        boxes1[:, 0] - boxes1[:, 2] / 2,
        boxes1[:, 1] - boxes1[:, 3] / 2,
        boxes1[:, 0] + boxes1[:, 2] / 2,
        boxes1[:, 1] + boxes1[:, 3] / 2,
    ], dim=-1)
    b2 = torch.stack([
        boxes2[:, 0] - boxes2[:, 2] / 2,
        boxes2[:, 1] - boxes2[:, 3] / 2,
        boxes2[:, 0] + boxes2[:, 2] / 2,
        boxes2[:, 1] + boxes2[:, 3] / 2,
    ], dim=-1)

    # Intersection
    inter_x1 = torch.max(b1[:, 0], b2[:, 0])
    inter_y1 = torch.max(b1[:, 1], b2[:, 1])
    inter_x2 = torch.min(b1[:, 2], b2[:, 2])
    inter_y2 = torch.min(b1[:, 3], b2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter

    iou = inter / (union + 1e-6)

    # Enclosing box
    enclose_x1 = torch.min(b1[:, 0], b2[:, 0])
    enclose_y1 = torch.min(b1[:, 1], b2[:, 1])
    enclose_x2 = torch.max(b1[:, 2], b2[:, 2])
    enclose_y2 = torch.max(b1[:, 3], b2[:, 3])
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

    giou = iou - (enclose_area - union) / (enclose_area + 1e-6)

    return (1 - giou).mean()


class DSITTLoss(nn.Module):
    """
    Collective Average Loss (CAL) for DSITT.
    Collects losses across all frames in a video clip and normalizes
    by total number of targets.

    Supports two box regression modes:
    - 'giou': Standard GIoU loss (default, from Deformable DETR)
    - 'nwd': Normalized Wasserstein Distance loss (better for small targets)
    """

    def __init__(self, num_classes: int = 7,
                 cls_weight: float = 2.0,
                 l1_weight: float = 5.0,
                 giou_weight: float = 2.0,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 box_loss_type: str = 'giou',
                 nwd_constant: float = 4.0):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight  # also used as nwd_weight when box_loss_type='nwd'
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.box_loss_type = box_loss_type
        self.nwd_constant = nwd_constant

    def forward(
        self,
        frame_outputs: List[Dict],
        frame_targets: List[Dict],
        frame_assignments: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute collective average loss across all frames.

        Args:
            frame_outputs: list of per-frame outputs, each with:
                'pred_logits': [B, N_q, num_classes]
                'pred_boxes': [B, N_q, 4]
            frame_targets: list of per-frame targets, each with:
                'labels': [M] class labels
                'boxes': [M, 4] ground truth boxes
            frame_assignments: list of per-frame assignments from TALA

        Returns:
            loss_dict: dict of loss components
        """
        total_cls_loss = 0.0
        total_l1_loss = 0.0
        total_giou_loss = 0.0
        total_targets = 0

        for output, target, assignment in zip(
            frame_outputs, frame_targets, frame_assignments
        ):
            matched_q = assignment['matched_query_indices']
            matched_g = assignment['matched_gt_indices']

            if len(matched_q) == 0:
                continue

            num_matched = len(matched_q)
            total_targets += num_matched

            pred_logits = output['pred_logits'][0]  # [N_q, C]
            pred_boxes = output['pred_boxes'][0]    # [N_q, 4]

            gt_labels = target['labels']  # [M]
            gt_boxes = target['boxes']    # [M, 4]

            # Classification loss (all queries contribute)
            # Create target one-hot
            target_classes_onehot = torch.zeros(
                pred_logits.shape[0], self.num_classes,
                dtype=pred_logits.dtype, device=pred_logits.device
            )
            target_classes_onehot[matched_q, gt_labels[matched_g]] = 1.0

            cls_loss = sigmoid_focal_loss(
                pred_logits, target_classes_onehot,
                num_boxes=max(num_matched, 1),
                alpha=self.focal_alpha, gamma=self.focal_gamma
            )

            # Box losses (only matched queries)
            pred_matched_boxes = pred_boxes[matched_q]
            gt_matched_boxes = gt_boxes[matched_g]

            l1_loss = F.l1_loss(pred_matched_boxes, gt_matched_boxes, reduction='mean')

            if self.box_loss_type == 'nwd':
                box_reg_loss = nwd_loss(
                    pred_matched_boxes, gt_matched_boxes,
                    constant=self.nwd_constant
                )
            else:
                box_reg_loss = generalized_box_iou_loss(
                    pred_matched_boxes, gt_matched_boxes
                )

            total_cls_loss += cls_loss * num_matched
            total_l1_loss += l1_loss * num_matched
            total_giou_loss += box_reg_loss * num_matched

        # Normalize by total targets (Collective Average Loss)
        total_targets = max(total_targets, 1)
        loss_cls = total_cls_loss / total_targets
        loss_l1 = total_l1_loss / total_targets
        loss_giou = total_giou_loss / total_targets

        # Weighted sum
        total_loss = (
            self.cls_weight * loss_cls +
            self.l1_weight * loss_l1 +
            self.giou_weight * loss_giou
        )

        # Use descriptive key based on loss type
        box_loss_key = 'loss_nwd' if self.box_loss_type == 'nwd' else 'loss_giou'

        # Auxiliary decoding losses (from intermediate decoder layers)
        device = loss_cls.device if isinstance(loss_cls, torch.Tensor) else (
            frame_outputs[0]['pred_logits'].device
        )
        aux_loss = torch.tensor(0.0, device=device)
        num_aux_layers = 0
        for output, target, assignment in zip(
            frame_outputs, frame_targets, frame_assignments
        ):
            aux_cls_list = output.get('aux_outputs_class', [])
            aux_coord_list = output.get('aux_outputs_coord', [])
            matched_q = assignment['matched_query_indices']
            matched_g = assignment['matched_gt_indices']

            if len(matched_q) == 0 or len(aux_cls_list) == 0:
                continue

            gt_labels = target['labels']
            gt_boxes = target['boxes']

            # Apply loss to all intermediate layers except the last (already counted above)
            for layer_idx in range(len(aux_cls_list) - 1):
                aux_cls = aux_cls_list[layer_idx][0]  # [N_q, C]
                aux_coord = aux_coord_list[layer_idx][0]  # [N_q, 4]

                # Classification loss
                target_onehot = torch.zeros(
                    aux_cls.shape[0], self.num_classes,
                    dtype=aux_cls.dtype, device=aux_cls.device
                )
                target_onehot[matched_q, gt_labels[matched_g]] = 1.0
                aux_cls_loss = sigmoid_focal_loss(
                    aux_cls, target_onehot,
                    num_boxes=max(len(matched_q), 1),
                    alpha=self.focal_alpha, gamma=self.focal_gamma
                )

                # Box loss
                aux_l1 = F.l1_loss(aux_coord[matched_q], gt_boxes[matched_g], reduction='mean')
                if self.box_loss_type == 'nwd':
                    aux_box = nwd_loss(aux_coord[matched_q], gt_boxes[matched_g], constant=self.nwd_constant)
                else:
                    aux_box = generalized_box_iou_loss(aux_coord[matched_q], gt_boxes[matched_g])

                aux_loss = aux_loss + (
                    self.cls_weight * aux_cls_loss +
                    self.l1_weight * aux_l1 +
                    self.giou_weight * aux_box
                )
                num_aux_layers += 1

        if num_aux_layers > 0:
            aux_loss = aux_loss / num_aux_layers
            total_loss = total_loss + aux_loss

        return {
            'loss': total_loss,
            'loss_cls': loss_cls,
            'loss_l1': loss_l1,
            box_loss_key: loss_giou,
            'loss_aux': aux_loss,
        }
