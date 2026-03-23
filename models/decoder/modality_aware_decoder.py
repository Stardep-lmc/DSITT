"""
Modality-Aware Decoder (MAD) for DSITT.

Core innovation: performs cross-modal fusion at the QUERY level rather than
feature level. Each query maintains structured multi-view representations:
  - q_rgb:    RGB modality view
  - q_ir:     IR modality view
  - q_motion: temporal motion view
  - q_fused:  unified fused query (used for final prediction)

Each MAD layer performs:
  1. Self-attention among fused queries (for inter-query interaction)
  2. Modality-specific cross-attention (q_rgb→F_rgb, q_ir→F_ir)
  3. Cross-modal query interaction (q_rgb↔q_ir bidirectional)
  4. Adaptive three-view fusion → q_fused

This is fundamentally different from feature-level fusion (e.g., ViPT, TBSI)
because the fusion happens inside the tracking query, tightly coupling
multi-modal reasoning with object-level tracking.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..ops.ms_deform_attn import MSDeformAttn
from .scale_adaptive_attn import ScaleAdaptiveDeformableAttn, scale_diversity_loss


class ModalityAwareDecoderLayer(nn.Module):
    """
    Single layer of the Modality-Aware Decoder.

    Processes four query views through a structured pipeline
    that enables cross-modal reasoning at the query level.
    """

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 n_levels=4, n_heads=8, n_points=4, use_sas=True):
        super().__init__()
        self.d_model = d_model
        self.use_sas = use_sas

        # Step 1: Self-attention among fused queries
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_sa = nn.LayerNorm(d_model)
        self.dropout_sa = nn.Dropout(dropout)

        # Step 2: Modality-specific cross-attention
        # Use Scale-Adaptive attention for tiny target detection
        if use_sas:
            self.cross_attn_rgb = ScaleAdaptiveDeformableAttn(
                d_model, n_levels, n_heads, n_points
            )
            self.cross_attn_ir = ScaleAdaptiveDeformableAttn(
                d_model, n_levels, n_heads, n_points
            )
        else:
            self.cross_attn_rgb = MSDeformAttn(d_model, n_levels, n_heads, n_points)
            self.cross_attn_ir = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm_ca_rgb = nn.LayerNorm(d_model)
        self.norm_ca_ir = nn.LayerNorm(d_model)
        self.dropout_ca_rgb = nn.Dropout(dropout)
        self.dropout_ca_ir = nn.Dropout(dropout)

        # Step 3: Cross-modal query interaction (bidirectional)
        self.cross_modal_rgb2ir = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_modal_ir2rgb = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_cm_rgb = nn.LayerNorm(d_model)
        self.norm_cm_ir = nn.LayerNorm(d_model)
        self.dropout_cm_rgb = nn.Dropout(dropout)
        self.dropout_cm_ir = nn.Dropout(dropout)

        # Step 4: Adaptive three-view fusion
        # Each gate predicts importance of [q_rgb, q_ir, q_motion] → d_model
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 3),  # 3 gates for 3 views
        )
        self.fusion_proj = nn.Linear(d_model, d_model)
        self.norm_fusion = nn.LayerNorm(d_model)
        self.dropout_fusion = nn.Dropout(dropout)

        # FFN for fused query
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout_ffn = nn.Dropout(dropout)

    def with_pos(self, tensor, pos):
        return tensor + pos if pos is not None else tensor

    def forward(self, queries, query_pos, reference_points,
                memory_rgb, spatial_shapes_rgb, level_start_rgb,
                memory_ir, spatial_shapes_ir, level_start_ir):
        """
        Args:
            queries: dict with keys 'q_rgb', 'q_ir', 'q_motion', 'q_fused'
                     each [B, N_q, d_model]
            query_pos: [B, N_q, d_model] shared position embedding
            reference_points: [B, N_q, n_levels, 2]
            memory_rgb: [B, sum(Hi*Wi), d_model] RGB encoder memory
            spatial_shapes_rgb: [n_levels, 2]
            level_start_rgb: [n_levels]
            memory_ir: [B, sum(Hi*Wi), d_model] IR encoder memory
            spatial_shapes_ir: [n_levels, 2]
            level_start_ir: [n_levels]

        Returns:
            updated queries dict, gate_weights [B, N_q, 3] for analysis
        """
        q_rgb = queries['q_rgb']
        q_ir = queries['q_ir']
        q_motion = queries['q_motion']
        q_fused = queries['q_fused']

        # ======== Step 1: Self-attention among fused queries ========
        q_sa = self.with_pos(q_fused, query_pos)
        q_fused2, _ = self.self_attn(q_sa, q_sa, q_fused)
        q_fused = q_fused + self.dropout_sa(q_fused2)
        q_fused = self.norm_sa(q_fused)

        # ======== Step 2: Modality-specific cross-attention ========
        scale_params_rgb = None
        scale_params_ir = None

        if self.use_sas:
            # SAS returns (output, scale_param)
            q_rgb2, scale_params_rgb = self.cross_attn_rgb(
                query=self.with_pos(q_rgb, query_pos),
                reference_points=reference_points,
                input_flatten=memory_rgb,
                input_spatial_shapes=spatial_shapes_rgb,
                input_level_start_index=level_start_rgb,
            )
            q_ir2, scale_params_ir = self.cross_attn_ir(
                query=self.with_pos(q_ir, query_pos),
                reference_points=reference_points,
                input_flatten=memory_ir,
                input_spatial_shapes=spatial_shapes_ir,
                input_level_start_index=level_start_ir,
            )
        else:
            q_rgb2 = self.cross_attn_rgb(
                query=self.with_pos(q_rgb, query_pos),
                reference_points=reference_points,
                input_flatten=memory_rgb,
                input_spatial_shapes=spatial_shapes_rgb,
                input_level_start_index=level_start_rgb,
            )
            q_ir2 = self.cross_attn_ir(
                query=self.with_pos(q_ir, query_pos),
                reference_points=reference_points,
                input_flatten=memory_ir,
                input_spatial_shapes=spatial_shapes_ir,
                input_level_start_index=level_start_ir,
            )

        q_rgb = q_rgb + self.dropout_ca_rgb(q_rgb2)
        q_rgb = self.norm_ca_rgb(q_rgb)
        q_ir = q_ir + self.dropout_ca_ir(q_ir2)
        q_ir = self.norm_ca_ir(q_ir)

        # ======== Step 3: Cross-modal query interaction ========
        # RGB asks IR: "what do you see at this location?"
        q_rgb2, _ = self.cross_modal_rgb2ir(
            query=q_rgb, key=q_ir, value=q_ir
        )
        q_rgb = q_rgb + self.dropout_cm_rgb(q_rgb2)
        q_rgb = self.norm_cm_rgb(q_rgb)

        # IR asks RGB: "what do you see at this location?"
        q_ir2, _ = self.cross_modal_ir2rgb(
            query=q_ir, key=q_rgb, value=q_rgb
        )
        q_ir = q_ir + self.dropout_cm_ir(q_ir2)
        q_ir = self.norm_cm_ir(q_ir)

        # ======== Step 4: Adaptive three-view fusion ========
        # Concatenate three views for gate prediction
        concat_views = torch.cat([q_rgb, q_ir, q_motion], dim=-1)  # [B, N, 3*d]
        gate_logits = self.fusion_gate(concat_views)  # [B, N, 3]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, N, 3]

        # Weighted combination
        g_rgb = gate_weights[:, :, 0:1]   # [B, N, 1]
        g_ir = gate_weights[:, :, 1:2]
        g_mot = gate_weights[:, :, 2:3]

        fused_input = g_rgb * q_rgb + g_ir * q_ir + g_mot * q_motion
        q_fused2 = self.fusion_proj(fused_input)
        q_fused = q_fused + self.dropout_fusion(q_fused2)
        q_fused = self.norm_fusion(q_fused)

        # FFN
        q_fused2 = self.ffn(q_fused)
        q_fused = q_fused + self.dropout_ffn(q_fused2)
        q_fused = self.norm_ffn(q_fused)

        # Average scale params from both modalities
        scale_params = None
        if scale_params_rgb is not None and scale_params_ir is not None:
            scale_params = (scale_params_rgb + scale_params_ir) / 2.0

        return {
            'q_rgb': q_rgb,
            'q_ir': q_ir,
            'q_motion': q_motion,
            'q_fused': q_fused,
        }, gate_weights, scale_params


class ModalityAwareDecoder(nn.Module):
    """
    Stack of Modality-Aware Decoder layers with prediction heads.

    The decoder processes MTUQ queries (four views per query) through
    multiple layers of cross-modal fusion, producing final predictions
    from the fused view while maintaining modality-specific views
    for auxiliary losses (CMC, to be added in Stage 3).
    """

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 n_levels=4, n_heads=8, n_points=4, num_layers=6,
                 num_classes=7):
        super().__init__()

        self.layers = nn.ModuleList([
            ModalityAwareDecoderLayer(
                d_model, d_ffn, dropout, n_levels, n_heads, n_points
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_levels = n_levels

        # Prediction heads (applied to q_fused)
        self.class_head = nn.Linear(d_model, num_classes)
        self.bbox_head = MLP(d_model, d_model, 4, num_layers=3)

        # Reference point projection
        self.reference_point_head = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        # Critical: initialize class head bias with prior probability
        # This ensures initial predictions are low (~0.01), matching focal loss expectations
        # Without this, focal loss pushes all scores to zero during early training
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        constant_(self.class_head.bias, bias_value)

    def forward(self, queries, query_pos,
                memory_rgb, spatial_shapes_rgb, level_start_rgb,
                memory_ir, spatial_shapes_ir, level_start_ir):
        """
        Args:
            queries: dict with 'q_rgb', 'q_ir', 'q_motion', 'q_fused' [B, N_q, d]
            query_pos: [B, N_q, d]
            memory_rgb: [B, sum(Hi*Wi), d] RGB encoder output
            spatial_shapes_rgb: [n_levels, 2]
            level_start_rgb: [n_levels]
            memory_ir, spatial_shapes_ir, level_start_ir: same for IR

        Returns:
            queries: updated query dict
            outputs_class: [B, N_q, num_classes]
            outputs_coord: [B, N_q, 4]
            reference_points: [B, N_q, 2]
            all_gate_weights: list of [B, N_q, 3] per layer
        """
        # Generate reference points from query position embedding
        reference_points = self.reference_point_head(query_pos).sigmoid()
        ref_points_input = reference_points[:, :, None, :].repeat(
            1, 1, self.n_levels, 1
        )

        all_gate_weights = []
        all_scale_params = []
        aux_outputs_class = []
        aux_outputs_coord = []

        for layer_idx, layer in enumerate(self.layers):
            queries, gate_weights, scale_params = layer(
                queries, query_pos, ref_points_input,
                memory_rgb, spatial_shapes_rgb, level_start_rgb,
                memory_ir, spatial_shapes_ir, level_start_ir,
            )
            all_gate_weights.append(gate_weights)
            if scale_params is not None:
                all_scale_params.append(scale_params)

            # Auxiliary predictions from intermediate layers
            q_fused_l = queries['q_fused']
            cls_l = self.class_head(q_fused_l)
            bbox_off_l = self.bbox_head(q_fused_l)
            coord_l = torch.cat([
                (reference_points + bbox_off_l[..., :2]).sigmoid(),
                bbox_off_l[..., 2:].sigmoid()
            ], dim=-1)
            aux_outputs_class.append(cls_l)
            aux_outputs_coord.append(coord_l)

        # Final predictions = last layer's predictions
        outputs_class = aux_outputs_class[-1]
        outputs_coord = aux_outputs_coord[-1]

        # Average scale params across layers for loss computation
        avg_scale_params = None
        if all_scale_params:
            avg_scale_params = torch.stack(all_scale_params).mean(0)

        return queries, outputs_class, outputs_coord, reference_points, all_gate_weights, avg_scale_params, aux_outputs_class, aux_outputs_coord


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

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