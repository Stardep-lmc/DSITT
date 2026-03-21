"""
Deformable Transformer Decoder.
Reference: Deformable DETR (Zhu et al., ICLR 2021) + MOTR (Zeng et al., ECCV 2022)

The decoder processes object queries (detect + track) through:
1. Self-attention among queries
2. Cross-attention with encoder memory (deformable)
3. FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import math

from ..ops.ms_deform_attn import MSDeformAttn


class DeformableTransformerDecoderLayer(nn.Module):
    """Single decoder layer."""

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # Self-attention (standard, among queries)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (deformable, queries attend to encoder memory)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor + pos if pos is not None else tensor

    def forward(self, tgt, query_pos, reference_points,
                memory, memory_spatial_shapes, memory_level_start_index,
                memory_padding_mask=None):
        """
        Args:
            tgt: [B, N_q, d_model] query features
            query_pos: [B, N_q, d_model] query position embeddings
            reference_points: [B, N_q, n_levels, 2] normalized reference points
            memory: [B, sum(Hi*Wi), d_model] encoder memory
            memory_spatial_shapes: [n_levels, 2]
            memory_level_start_index: [n_levels]
            memory_padding_mask: optional

        Returns:
            tgt: [B, N_q, d_model] updated query features
        """
        # 1. Self-attention among queries
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Cross-attention with encoder memory (deformable)
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, query_pos),
            reference_points=reference_points,
            input_flatten=memory,
            input_spatial_shapes=memory_spatial_shapes,
            input_level_start_index=memory_level_start_index,
            input_padding_mask=memory_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 3. FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    """Stack of Deformable Transformer Decoder Layers with iterative box refinement."""

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 n_levels=4, n_heads=8, n_points=4, num_layers=6,
                 num_classes=7):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableTransformerDecoderLayer(
                d_model, d_ffn, dropout, n_levels, n_heads, n_points
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_levels = n_levels

        # Prediction heads (shared across layers for simplicity)
        self.class_head = nn.Linear(d_model, num_classes)
        self.bbox_head = MLP(d_model, d_model, 4, num_layers=3)

        # Reference point projection (from query to initial reference point)
        self.reference_point_head = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, tgt, query_pos, memory, memory_spatial_shapes,
                memory_level_start_index, memory_padding_mask=None):
        """
        Args:
            tgt: [B, N_q, d_model] query features (detect + track concatenated)
            query_pos: [B, N_q, d_model] query position embeddings
            memory: [B, sum(Hi*Wi), d_model] encoder memory
            memory_spatial_shapes: [n_levels, 2]
            memory_level_start_index: [n_levels]

        Returns:
            hs: [B, N_q, d_model] final hidden states
            outputs_class: [B, N_q, num_classes] classification logits
            outputs_coord: [B, N_q, 4] predicted boxes (cx, cy, w, h) normalized
            reference_points: [B, N_q, 2] reference points used
        """
        output = tgt

        # Generate reference points from query embeddings
        reference_points = self.reference_point_head(query_pos).sigmoid()
        # Expand to all levels: [B, N_q, 2] -> [B, N_q, n_levels, 2]
        reference_points_input = reference_points[:, :, None, :].repeat(
            1, 1, self.n_levels, 1
        )

        # Decoder layers
        for layer in self.layers:
            output = layer(
                output, query_pos, reference_points_input,
                memory, memory_spatial_shapes, memory_level_start_index,
                memory_padding_mask
            )

        # Final predictions
        outputs_class = self.class_head(output)
        # Predict box offsets relative to reference points
        bbox_offset = self.bbox_head(output)
        # Convert to absolute coordinates
        outputs_coord = torch.cat([
            (reference_points + bbox_offset[..., :2]).sigmoid(),
            bbox_offset[..., 2:].sigmoid()
        ], dim=-1)

        return output, outputs_class, outputs_coord, reference_points


class MLP(nn.Module):
    """Simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x