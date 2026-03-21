"""
Scale-Adaptive Deformable Attention (SAS) for tiny target detection.

Core insight: Standard deformable attention samples points with unconstrained
offsets. For tiny targets (<8px), these offsets can easily land outside the
target boundary. SAS constrains the sampling range based on a learned
per-query scale parameter.

Each query learns its own scale_param ∈ (0, 1):
  - Small scale_param → small sampling range → suitable for tiny targets
  - Large scale_param → large sampling range → suitable for normal targets

The offset is constrained:
  offset = tanh(raw_offset) * scale_param * max_offset

This is more elegant than hardcoded multi-granularity query groups because:
1. Fully end-to-end learned, no manual thresholds
2. Each query adapts independently
3. scale_param also used for loss weighting (small targets get more gradient)

Improvement over V2 roadmap: Added scale_param regularization loss
to encourage diversity of scales across queries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import math


class ScaleAdaptiveDeformableAttn(nn.Module):
    """
    Multi-Scale Deformable Attention with per-query scale adaptation.

    Wraps the standard MSDeformAttn with:
    1. A scale predictor that outputs per-query scale parameters
    2. Constrained sampling offsets based on predicted scale
    3. Returns scale_param for downstream loss weighting
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4,
                 max_offset=0.5):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.max_offset = max_offset

        # Scale predictor: query → scale ∈ (0, 1)
        self.scale_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Standard deformable attention components
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
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

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        """
        Args:
            query: [B, N_q, d_model]
            reference_points: [B, N_q, n_levels, 2]
            input_flatten: [B, sum(Hi*Wi), d_model]
            input_spatial_shapes: [n_levels, 2]
            input_level_start_index: [n_levels]

        Returns:
            output: [B, N_q, d_model]
            scale_param: [B, N_q, 1] per-query scale parameter
        """
        B, N_q, _ = query.shape
        B, N_in, _ = input_flatten.shape

        # Predict per-query scale
        scale_param = self.scale_predictor(query)  # [B, N_q, 1]

        # Value projection
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.)
        value = value.view(B, N_in, self.n_heads, self.d_model // self.n_heads)

        # Raw sampling offsets
        raw_offsets = self.sampling_offsets(query).view(
            B, N_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        # Scale-constrained offsets: tanh constrains to [-1,1], then scale
        constrained_offsets = (
            torch.tanh(raw_offsets)
            * scale_param[:, :, :, None, None, None]  # broadcast scale
            * self.max_offset
        )

        # Attention weights
        attention_weights = self.attention_weights(query).view(
            B, N_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Compute sampling locations
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + constrained_offsets / offset_normalizer[None, None, None, :, None, :]
        )

        # Core deformable attention computation
        from ..ops.ms_deform_attn import ms_deform_attn_core_pytorch
        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)
        return output, scale_param


def scale_diversity_loss(scale_params: torch.Tensor, target_std: float = 0.15):
    """
    Regularization loss to encourage diversity of scale parameters.

    Without this, all queries might converge to the same scale.
    We want some queries to specialize in tiny targets (small scale)
    and others in larger targets (large scale).

    Args:
        scale_params: [B, N_q, 1] predicted scale parameters
        target_std: desired standard deviation of scales

    Returns:
        loss: scalar, penalizes lack of diversity
    """
    std = scale_params.squeeze(-1).std(dim=-1).mean()  # avg std across batch
    return F.relu(target_std - std)  # penalize if std < target