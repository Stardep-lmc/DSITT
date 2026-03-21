"""
Deformable Transformer Encoder for multi-scale feature processing.
Reference: Deformable DETR (Zhu et al., ICLR 2021)
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from ..ops.ms_deform_attn import MSDeformAttn


class DeformableTransformerEncoderLayer(nn.Module):
    """Single encoder layer with multi-scale deformable self-attention."""

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # Self-attention (deformable)
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes,
                level_start_index, padding_mask=None):
        """
        Args:
            src: [B, sum(Hi*Wi), d_model]
            pos: [B, sum(Hi*Wi), d_model]
            reference_points: [B, sum(Hi*Wi), n_levels, 2]
            spatial_shapes: [n_levels, 2]
            level_start_index: [n_levels]
            padding_mask: [B, sum(Hi*Wi)] (optional)
        """
        # Self-attention
        src2 = self.self_attn(
            query=src + pos,
            reference_points=reference_points,
            input_flatten=src,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    """Stack of Deformable Transformer Encoder Layers."""

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 n_levels=4, n_heads=8, n_points=4, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableTransformerEncoderLayer(
                d_model, d_ffn, dropout, n_levels, n_heads, n_points
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_levels = n_levels

        # Level embedding (learnable, one per feature level)
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
        normal_(self.level_embed)

    def get_reference_points(self, spatial_shapes, device):
        """
        Generate reference points for each spatial location across all levels.

        Args:
            spatial_shapes: [n_levels, 2] (H, W)
            device: torch device

        Returns:
            reference_points: [1, sum(Hi*Wi), n_levels, 2] normalized (0~1)
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            H, W = int(H), int(W)
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1) / H  # normalize to [0, 1]
            ref_x = ref_x.reshape(-1) / W
            ref = torch.stack([ref_x, ref_y], -1)  # [H*W, 2]
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, dim=0)  # [sum(Hi*Wi), 2]
        # Expand to all levels
        reference_points = reference_points[:, None, :].repeat(
            1, len(spatial_shapes), 1
        )  # [sum(Hi*Wi), n_levels, 2]
        return reference_points.unsqueeze(0)  # [1, sum(Hi*Wi), n_levels, 2]

    def forward(self, srcs, pos_embeds):
        """
        Args:
            srcs: list of [B, d_model, Hi, Wi] multi-scale features (from FPN)
            pos_embeds: list of [B, d_model, Hi, Wi] position embeddings

        Returns:
            memory: [B, sum(Hi*Wi), d_model] encoded features
            spatial_shapes: [n_levels, 2]
            level_start_index: [n_levels]
        """
        # Flatten multi-scale features
        src_flatten = []
        pos_flatten = []
        spatial_shapes = []

        for lvl, (src, pos) in enumerate(zip(srcs, pos_embeds)):
            B, C, H, W = src.shape
            spatial_shapes.append((H, W))

            src = src.flatten(2).transpose(1, 2)  # [B, H*W, C]
            pos = pos.flatten(2).transpose(1, 2)  # [B, H*W, C]

            # Add level embedding
            src = src + self.level_embed[lvl].view(1, 1, -1)

            src_flatten.append(src)
            pos_flatten.append(pos)

        src_flatten = torch.cat(src_flatten, dim=1)  # [B, sum(Hi*Wi), C]
        pos_flatten = torch.cat(pos_flatten, dim=1)

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat([
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ])

        # Reference points
        reference_points = self.get_reference_points(
            spatial_shapes, device=src_flatten.device
        )
        reference_points = reference_points.expand(B, -1, -1, -1)

        # Encoder layers
        output = src_flatten
        for layer in self.layers:
            output = layer(
                output, pos_flatten, reference_points,
                spatial_shapes, level_start_index
            )

        return output, spatial_shapes, level_start_index