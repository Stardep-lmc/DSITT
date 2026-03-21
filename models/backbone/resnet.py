"""
Backbone: ResNet-50 with FPN for multi-scale feature extraction.
Based on torchvision's ResNet and Deformable DETR's backbone design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, List


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d where the batch statistics and affine parameters are fixed."""

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """ResNet backbone that returns multi-scale features."""

    def __init__(self, backbone: nn.Module, num_channels: List[int],
                 return_layers: List[str]):
        super().__init__()
        self.body = backbone
        self.num_channels = num_channels
        self.return_layers = return_layers

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] input image tensor
        Returns:
            dict of multi-scale features: {'0': C2, '1': C3, '2': C4, '3': C5}
        """
        features = {}
        x = self.body.conv1(x)
        x = self.body.bn1(x)
        x = self.body.relu(x)
        x = self.body.maxpool(x)

        x = self.body.layer1(x)
        if 'layer1' in self.return_layers:
            features['0'] = x

        x = self.body.layer2(x)
        if 'layer2' in self.return_layers:
            features['1'] = x

        x = self.body.layer3(x)
        if 'layer3' in self.return_layers:
            features['2'] = x

        x = self.body.layer4(x)
        if 'layer4' in self.return_layers:
            features['3'] = x

        return features


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion."""

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            output = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(lateral)
            self.output_convs.append(output)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: dict from backbone, {'0': C2, '1': C3, '2': C4, '3': C5}
        Returns:
            list of FPN features [P2, P3, P4, P5], each with out_channels dims
        """
        keys = sorted(features.keys())
        laterals = [self.lateral_convs[i](features[k]) for i, k in enumerate(keys)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=(h, w), mode='nearest'
            )

        # Output convolutions
        outputs = [self.output_convs[i](laterals[i]) for i in range(len(laterals))]
        return outputs


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embedding for 2D feature maps."""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * 3.14159265358979

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map
        Returns:
            pos: [B, num_pos_feats*2, H, W] position embedding
        """
        b, c, h, w = x.shape
        mask = torch.zeros(b, h, w, dtype=torch.bool, device=x.device)
        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # [B, H, W, D]
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)

        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)
        return pos


class Backbone(nn.Module):
    """Complete backbone: ResNet-50 + FPN + Position Embedding."""

    def __init__(self, d_model: int = 256, pretrained: bool = True,
                 freeze_bn: bool = True):
        super().__init__()

        # Build ResNet-50
        if pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = resnet50(weights=None)

        # Optionally freeze batch norm
        if freeze_bn:
            for module in backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.weight.requires_grad_(False)
                    module.bias.requires_grad_(False)
                    module.eval()

        # ResNet channel dimensions: layer1=256, layer2=512, layer3=1024, layer4=2048
        self.backbone = BackboneBase(
            backbone,
            num_channels=[256, 512, 1024, 2048],
            return_layers=['layer1', 'layer2', 'layer3', 'layer4']
        )

        # FPN
        self.fpn = FPN(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=d_model
        )

        # Position embedding
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=d_model // 2
        )

        self.num_channels = d_model
        self.num_feature_levels = 4

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W] input image
        Returns:
            srcs: list of [B, d_model, Hi, Wi] multi-scale features
            pos_embeds: list of [B, d_model, Hi, Wi] position embeddings
        """
        # Extract multi-scale features
        backbone_features = self.backbone(x)
        fpn_features = self.fpn(backbone_features)

        # Generate position embeddings for each level
        pos_embeds = [self.position_embedding(feat) for feat in fpn_features]

        return fpn_features, pos_embeds


def build_backbone(d_model: int = 256, pretrained: bool = True) -> Backbone:
    """Build backbone module."""
    return Backbone(d_model=d_model, pretrained=pretrained)