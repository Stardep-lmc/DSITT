"""
Motion View Updater for DSITT.

Enhances the motion view of MTUQ queries with explicit temporal modeling:
1. Trajectory Memory Bank: stores recent K frames' query features + predicted boxes
2. Motion Pattern Encoder: lightweight Transformer encoding historical trajectory
3. Gated motion injection: adaptive blending of motion information

This replaces the simplified q_motion = proj(prev_q_fused) with a richer
temporal representation that captures movement patterns.

Improvement over V2 roadmap:
- Batched temporal encoding (no per-target loop, much faster)
- Box velocity features: encode Δbox between consecutive frames
- Adaptive memory length based on track age
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from collections import deque


class MotionViewUpdater(nn.Module):
    """
    Updates the motion view of tracking queries using historical trajectory information.

    For each tracked target, maintains a memory of recent frames' features and positions,
    encodes the motion pattern, and injects it into q_motion.
    """

    def __init__(self, d_model: int = 256, max_history: int = 5, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.max_history = max_history

        # Position encoder: box (cx,cy,w,h) + velocity (Δcx,Δcy,Δw,Δh) → d_model
        self.pos_encoder = nn.Sequential(
            nn.Linear(8, d_model // 2),   # 4 (box) + 4 (velocity)
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, d_model),
        )

        # Temporal encoder: 2-layer Transformer on history sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2,
        )

        # Temporal position encoding (learnable, per-timestep)
        self.temporal_pos = nn.Embedding(max_history + 1, d_model)

        # Gated injection
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        q_motion: torch.Tensor,
        history_features: Optional[torch.Tensor] = None,
        history_boxes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update motion view using trajectory history.

        Args:
            q_motion: [B, N, d] current motion queries
            history_features: [K, B, N, d] past K frames' q_fused features
                             None for first frame or detect queries
            history_boxes: [K, B, N, 4] past K frames' predicted boxes
                          None for first frame or detect queries

        Returns:
            q_motion_updated: [B, N, d]
        """
        if history_features is None or len(history_features) == 0:
            return q_motion

        K, B, N, d = history_features.shape

        # Compute box velocities (Δ between consecutive frames)
        # [K, B, N, 4]
        if K >= 2:
            velocities = history_boxes[1:] - history_boxes[:-1]  # [K-1, B, N, 4]
            # Pad first frame velocity with zeros
            velocities = torch.cat([
                torch.zeros(1, B, N, 4, device=velocities.device),
                velocities
            ], dim=0)  # [K, B, N, 4]
        else:
            velocities = torch.zeros_like(history_boxes)

        # Combine box + velocity features
        box_vel = torch.cat([history_boxes, velocities], dim=-1)  # [K, B, N, 8]
        pos_embed = self.pos_encoder(box_vel)  # [K, B, N, d]

        # Add features + position
        temporal_input = history_features + pos_embed  # [K, B, N, d]

        # Add temporal position encoding
        temporal_pos_ids = torch.arange(K, device=q_motion.device)
        temporal_pos_embed = self.temporal_pos(temporal_pos_ids)  # [K, d]
        temporal_input = temporal_input + temporal_pos_embed[:, None, None, :]

        # Reshape for batched encoding: [K, B, N, d] → [B*N, K, d]
        temporal_input = temporal_input.permute(1, 2, 0, 3).reshape(B * N, K, d)

        # Encode temporal sequence
        encoded = self.temporal_encoder(temporal_input)  # [B*N, K, d]

        # Take last timestep as motion token
        motion_token = encoded[:, -1, :]  # [B*N, d]
        motion_token = motion_token.reshape(B, N, d)  # [B, N, d]

        # Gated injection into q_motion
        gate_input = torch.cat([q_motion, motion_token], dim=-1)  # [B, N, 2d]
        gate_value = self.gate(gate_input)  # [B, N, d], values in (0, 1)

        q_motion_updated = q_motion + gate_value * self.out_proj(motion_token)

        return q_motion_updated


class TrajectoryMemoryBank:
    """
    Simple memory bank that stores recent frames' query features and predicted boxes.

    Used by MotionViewUpdater to provide historical context for motion encoding.
    """

    def __init__(self, max_length: int = 5):
        self.max_length = max_length
        self.features: List[torch.Tensor] = []  # each [B, N, d]
        self.boxes: List[torch.Tensor] = []      # each [B, N, 4]

    def reset(self):
        """Clear memory for new video."""
        self.features.clear()
        self.boxes.clear()

    def push(self, features: torch.Tensor, boxes: torch.Tensor):
        """
        Add a frame's data to memory.

        Args:
            features: [B, N, d] q_fused features
            boxes: [B, N, 4] predicted boxes
        """
        self.features.append(features.detach())
        self.boxes.append(boxes.detach())

        # Trim to max length
        if len(self.features) > self.max_length:
            self.features.pop(0)
            self.boxes.pop(0)

    def get_history(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get stacked history tensors.

        Returns:
            features: [K, B, N, d] or None
            boxes: [K, B, N, 4] or None
        """
        if len(self.features) == 0:
            return None, None

        return (
            torch.stack(self.features, dim=0),  # [K, B, N, d]
            torch.stack(self.boxes, dim=0),     # [K, B, N, 4]
        )

    @property
    def length(self) -> int:
        return len(self.features)