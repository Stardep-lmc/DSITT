"""
Dual-modality data augmentation for RGBT-Tiny.

Geometric transforms (flip) are applied synchronously to both RGB and IR.
Color jitter is applied only to RGB (IR is thermal, color augmentation is meaningless).
All transforms operate on normalized tensors and (cx, cy, w, h) boxes in [0,1].
"""

import random
import torch


class DualModalityTransform:
    """Synchronized augmentation for RGB-IR pairs with bbox adjustment."""

    def __init__(self, train: bool = True):
        self.train = train

    def __call__(
        self,
        img_rgb: torch.Tensor,
        img_ir: torch.Tensor,
        boxes: torch.Tensor,
    ):
        """
        Args:
            img_rgb: [3, H, W] normalized tensor
            img_ir:  [3, H, W] normalized tensor
            boxes:   [N, 4] (cx, cy, w, h) in [0, 1]

        Returns:
            img_rgb, img_ir, boxes (possibly augmented)
        """
        if not self.train or len(boxes) == 0:
            return img_rgb, img_ir, boxes

        # 1. Random horizontal flip (sync RGB + IR + boxes)
        if random.random() < 0.5:
            img_rgb = img_rgb.flip(-1)
            img_ir = img_ir.flip(-1)
            boxes = boxes.clone()
            boxes[:, 0] = 1.0 - boxes[:, 0]  # flip cx

        # 2. Random brightness/contrast on RGB only
        #    Works on normalized tensors by shifting/scaling pixel values
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            img_rgb = img_rgb * brightness_factor

        if random.random() < 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = img_rgb.mean()
            img_rgb = (img_rgb - mean) * contrast_factor + mean

        return img_rgb, img_ir, boxes