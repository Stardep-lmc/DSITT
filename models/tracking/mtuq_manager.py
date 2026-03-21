"""
MTUQ (Modality-Temporal Unified Query) Manager for DSITT v2.

Extends TrackQueryManager to handle structured multi-view queries:
  {q_rgb, q_ir, q_motion, q_fused}

Key differences from single-vector TrackQueryManager:
1. Detect queries: four independent learnable embeddings
2. Track queries: inherited from previous frame's four-view outputs
3. QIM projects all four views for next frame
4. TALA operates on q_fused (compatible with original logic)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .track_manager import (
    QueryInteractionModule,
    TrajectoryAwareLabelAssignment,
)


class MTUQQueryInteractionModule(nn.Module):
    """
    Extended QIM that projects all four query views for the next frame.
    """

    def __init__(self, d_model: int = 256, p_drop: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.p_drop = p_drop

        # Project each view independently
        self.proj_rgb = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.proj_ir = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.proj_motion = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.proj_fused = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.proj_pos = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, queries: Dict[str, torch.Tensor],
                query_pos: torch.Tensor,
                scores: torch.Tensor,
                training: bool = True):
        """
        Args:
            queries: dict with q_rgb, q_ir, q_motion, q_fused [B, N, d]
            query_pos: [B, N, d]
            scores: [B, N] confidence scores

        Returns:
            track_queries: dict of projected query views
            track_pos: [B, N, d]
            active_mask: [B, N] boolean
        """
        B, N, _ = queries['q_fused'].shape

        if training:
            keep_mask = torch.rand(B, N, device=scores.device) > self.p_drop
            score_mask = scores > 0.5
            active_mask = keep_mask & score_mask
        else:
            active_mask = scores > 0.5

        track_queries = {
            'q_rgb': self.proj_rgb(queries['q_rgb']),
            'q_ir': self.proj_ir(queries['q_ir']),
            'q_motion': self.proj_motion(queries['q_fused']),  # motion = prev fused
            'q_fused': self.proj_fused(queries['q_fused']),
        }
        track_pos = self.proj_pos(query_pos)

        return track_queries, track_pos, active_mask


class MTUQManager(nn.Module):
    """
    Manages MTUQ queries lifecycle: creation, propagation, update.

    Detect queries: learnable four-view embeddings (used for new targets)
    Track queries: propagated from previous frame via MTUQ-QIM
    """

    def __init__(self, d_model: int = 256, num_queries: int = 300,
                 p_drop: float = 0.1, match_cost_type: str = 'nwd',
                 nwd_constant: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # Learnable detect queries — four views
        self.detect_q_rgb = nn.Embedding(num_queries, d_model)
        self.detect_q_ir = nn.Embedding(num_queries, d_model)
        self.detect_q_motion = nn.Embedding(num_queries, d_model)
        self.detect_q_fused = nn.Embedding(num_queries, d_model)
        self.detect_query_pos = nn.Embedding(num_queries, d_model)

        # MTUQ-QIM
        self.qim = MTUQQueryInteractionModule(d_model, p_drop)

        # TALA (operates on q_fused, same as before)
        self.tala = TrajectoryAwareLabelAssignment(
            match_cost_type=match_cost_type,
            nwd_constant=nwd_constant,
        )

        # State
        self._track_queries: Optional[Dict[str, torch.Tensor]] = None
        self._track_query_pos: Optional[torch.Tensor] = None
        self._track_assignment: Optional[Dict] = None

    def reset(self):
        """Reset tracking state for new video."""
        self._track_queries = None
        self._track_query_pos = None
        self._track_assignment = None

    @property
    def num_track_queries(self) -> int:
        if self._track_queries is None:
            return 0
        return self._track_queries['q_fused'].shape[1]

    def get_queries(self, device: torch.device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get combined (track + detect) MTUQ queries.

        Returns:
            queries: dict with q_rgb, q_ir, q_motion, q_fused [1, N_total, d]
            query_pos: [1, N_total, d]
        """
        # Detect queries
        detect_queries = {
            'q_rgb': self.detect_q_rgb.weight.unsqueeze(0),
            'q_ir': self.detect_q_ir.weight.unsqueeze(0),
            'q_motion': self.detect_q_motion.weight.unsqueeze(0),
            'q_fused': self.detect_q_fused.weight.unsqueeze(0),
        }
        detect_pos = self.detect_query_pos.weight.unsqueeze(0)

        if self._track_queries is not None and self.num_track_queries > 0:
            # Concatenate track + detect
            queries = {
                k: torch.cat([self._track_queries[k], detect_queries[k]], dim=1)
                for k in detect_queries
            }
            query_pos = torch.cat([self._track_query_pos, detect_pos], dim=1)
        else:
            queries = detect_queries
            query_pos = detect_pos

        return queries, query_pos

    def update(
        self,
        queries: Dict[str, torch.Tensor],
        query_pos: torch.Tensor,
        outputs_class: torch.Tensor,
        outputs_coord: torch.Tensor,
        targets: Optional[Dict] = None,
        training: bool = True,
    ) -> Optional[Dict]:
        """
        Update track queries after MAD decoder processing.

        Uses q_fused for scoring and label assignment (compatible with TALA).
        Propagates all four views via MTUQ-QIM.
        """
        scores = outputs_class.sigmoid().max(dim=-1)[0]  # [B, N_q]

        # Label assignment (training only)
        assignment = None
        if training and targets is not None:
            assignment = self.tala.assign(
                outputs_class, outputs_coord, targets,
                self.num_track_queries, self._track_assignment
            )
            self._track_assignment = assignment['track_assignment']

        # Generate track queries for next frame
        track_queries, track_pos, active_mask = self.qim(
            queries, query_pos, scores, training
        )

        if training and assignment is not None:
            matched_q = assignment['matched_query_indices']
            if len(matched_q) > 0:
                self._track_queries = {
                    k: v[:, matched_q, :] for k, v in track_queries.items()
                }
                self._track_query_pos = track_pos[:, matched_q, :]
            else:
                self._track_queries = None
                self._track_query_pos = None
        else:
            if active_mask.any():
                active_idx = active_mask[0].nonzero(as_tuple=True)[0]
                self._track_queries = {
                    k: v[:, active_idx, :] for k, v in track_queries.items()
                }
                self._track_query_pos = track_pos[:, active_idx, :]
            else:
                self._track_queries = None
                self._track_query_pos = None

        return assignment