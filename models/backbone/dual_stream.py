"""
Dual-Stream Backbone for multi-modal (RGB + IR) feature extraction.

Design choices:
1. Two independent Backbone instances (no weight sharing)
2. Each produces multi-scale features + position embeddings
3. Supports single-modality fallback (when one stream receives zeros)

Improvement over naive dual-stream:
- Modality dropout during training: randomly zero out one modality
  to build robustness against single-modality scenarios
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from .resnet import Backbone, build_backbone


class DualStreamBackbone(nn.Module):
    """
    Dual-stream backbone that processes RGB and IR images independently.
    
    During training, applies modality dropout to build robustness.
    """

    def __init__(self, d_model: int = 256, pretrained: bool = True,
                 modality_dropout: float = 0.0):
        """
        Args:
            d_model: output feature dimension
            pretrained: use ImageNet pretrained weights
            modality_dropout: probability of dropping a modality during training
                             (0.0 = no dropout, 0.1 = 10% chance of dropping one modality)
        """
        super().__init__()

        self.backbone_rgb = build_backbone(d_model=d_model, pretrained=pretrained)
        self.backbone_ir = build_backbone(d_model=d_model, pretrained=pretrained)

        self.num_channels = d_model
        self.num_feature_levels = 4
        self.modality_dropout = modality_dropout

    def forward(
        self,
        img_rgb: torch.Tensor,
        img_ir: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
               List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            img_rgb: [B, 3, H, W] RGB image
            img_ir: [B, 3, H, W] IR image

        Returns:
            srcs_rgb: list of [B, d_model, Hi, Wi] RGB multi-scale features
            pos_rgb: list of [B, d_model, Hi, Wi] RGB position embeddings
            srcs_ir: list of [B, d_model, Hi, Wi] IR multi-scale features
            pos_ir: list of [B, d_model, Hi, Wi] IR position embeddings
        """
        # Modality dropout during training
        if self.training and self.modality_dropout > 0:
            rand_val = torch.rand(1).item()
            if rand_val < self.modality_dropout / 2:
                # Drop RGB (zero it out)
                img_rgb = torch.zeros_like(img_rgb)
            elif rand_val < self.modality_dropout:
                # Drop IR (zero it out)
                img_ir = torch.zeros_like(img_ir)

        # Process each modality independently
        srcs_rgb, pos_rgb = self.backbone_rgb(img_rgb)
        srcs_ir, pos_ir = self.backbone_ir(img_ir)

        return srcs_rgb, pos_rgb, srcs_ir, pos_ir


def build_dual_stream_backbone(d_model: int = 256, pretrained: bool = True,
                                modality_dropout: float = 0.0) -> DualStreamBackbone:
    """Build dual-stream backbone."""
    return DualStreamBackbone(
        d_model=d_model, pretrained=pretrained,
        modality_dropout=modality_dropout
    )