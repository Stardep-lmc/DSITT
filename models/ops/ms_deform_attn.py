"""
Multi-Scale Deformable Attention Module (Pure PyTorch implementation).
Reference: Deformable DETR (Zhu et al., ICLR 2021)

This is a pure PyTorch implementation without custom CUDA operators.
Slower than the CUDA version but portable and easy to debug.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import math


def ms_deform_attn_core_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor
) -> torch.Tensor:
    """
    Core multi-scale deformable attention function (pure PyTorch).

    Args:
        value: [B, sum(Hi*Wi), M, D]  (M=num_heads, D=d_model//M)
        value_spatial_shapes: [num_levels, 2] (H, W for each level)
        sampling_locations: [B, sum(Hi*Wi), M, num_levels, num_points, 2]
            normalized to [0, 1]
        attention_weights: [B, sum(Hi*Wi), M, num_levels * num_points]

    Returns:
        output: [B, sum(Hi*Wi), M*D]
    """
    B, _, M, D = value.shape
    _, Len_q, _, num_levels, num_points, _ = sampling_locations.shape

    # Split value into per-level tensors
    value_list = value.split(
        [int(h * w) for h, w in value_spatial_shapes], dim=1
    )

    # Sampling grid: convert from [0,1] to [-1,1] for grid_sample
    sampling_grids = 2 * sampling_locations - 1  # [B, Len_q, M, L, P, 2]

    sampling_value_list = []
    for lid, (h, w) in enumerate(value_spatial_shapes):
        h, w = int(h), int(w)
        # value_l: [B, h*w, M, D] -> [B*M, D, h, w]
        value_l = value_list[lid].permute(0, 2, 3, 1).reshape(B * M, D, h, w)

        # sampling_grid_l: [B, Len_q, M, P, 2] -> [B*M, Len_q, P, 2]
        sampling_grid_l = sampling_grids[:, :, :, lid, :, :] \
            .permute(0, 2, 1, 3, 4).reshape(B * M, Len_q, num_points, 2)

        # grid_sample: [B*M, D, Len_q, P]
        sampling_value_l = F.grid_sample(
            value_l, sampling_grid_l,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )
        sampling_value_list.append(sampling_value_l)

    # attention_weights: [B, Len_q, M, L*P] -> [B*M, 1, Len_q, L*P]
    attention_weights = attention_weights.permute(0, 2, 1, 3) \
        .reshape(B * M, 1, Len_q, num_levels * num_points)

    # Stack sampling values: [B*M, D, Len_q, L*P]
    sampling_values = torch.stack(sampling_value_list, dim=-1) \
        .reshape(B * M, D, Len_q, num_levels * num_points)

    # Weighted sum: [B*M, D, Len_q]
    output = (sampling_values * attention_weights).sum(-1)

    # Reshape: [B, M*D, Len_q] -> [B, Len_q, M*D]
    output = output.reshape(B, M, D, Len_q).permute(0, 3, 1, 2).reshape(B, Len_q, M * D)

    return output


class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention Module.

    Args:
        d_model: hidden dimension
        n_levels: number of feature levels
        n_heads: number of attention heads
        n_points: number of sampling points per head per level
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # Initialize sampling offsets as grid pattern
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]) \
            .view(self.n_heads, 1, 1, 2) \
            .repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query: [B, Len_q, d_model]
            reference_points: [B, Len_q, n_levels, 2] normalized (0~1)
            input_flatten: [B, sum(Hi*Wi), d_model] flattened multi-scale features
            input_spatial_shapes: [n_levels, 2] (H, W)
            input_level_start_index: [n_levels] cumulative start indices
            input_padding_mask: [B, sum(Hi*Wi)] (optional)

        Returns:
            output: [B, Len_q, d_model]
        """
        B, Len_q, _ = query.shape
        B, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(B, Len_in, self.n_heads, self.d_model // self.n_heads)

        # Sampling offsets: [B, Len_q, n_heads, n_levels, n_points, 2]
        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        # Attention weights: [B, Len_q, n_heads, n_levels * n_points]
        attention_weights = self.attention_weights(query).view(
            B, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1)

        # Reference points + offsets -> sampling locations
        # reference_points: [B, Len_q, n_levels, 2] -> [B, Len_q, 1, n_levels, 1, 2]
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )  # [n_levels, 2] (W, H)
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )

        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)
        return output