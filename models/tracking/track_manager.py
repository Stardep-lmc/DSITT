"""
Track Query Manager for DSITT.
Implements:
- Query Interaction Module (QIM): generates track queries from decoder hidden states
- Trajectory-Aware Label Assignment (TALA): assigns labels considering track identity
- Enter-and-Leave mechanism: handles newborn and disappeared targets

Reference: MOTR (Zeng et al., ECCV 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Tuple


class QueryInteractionModule(nn.Module):
    """
    Query Interaction Module (QIM).
    Generates track queries for the next frame from current decoder hidden states.
    Includes dropout (simulating target disappearance) and insertion (simulating FP).
    """

    def __init__(self, d_model: int = 256, p_drop: float = 0.1,
                 p_insert: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.p_drop = p_drop
        self.p_insert = p_insert

        # Project hidden state to track query
        self.track_query_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        # Project hidden state to track query position embedding
        self.track_pos_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_pos: torch.Tensor,
        scores: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate track queries for the next frame.

        Args:
            hidden_states: [B, N_q, d_model] decoder output hidden states
            query_pos: [B, N_q, d_model] query position embeddings
            scores: [B, N_q] classification confidence scores (max over classes)
            training: whether in training mode

        Returns:
            track_queries: [B, N_active, d_model] track queries for next frame
            track_pos: [B, N_active, d_model] track query position embeddings
            active_mask: [B, N_q] boolean mask of active tracks
        """
        B, N_q, _ = hidden_states.shape

        # Determine active tracks based on score threshold
        # During training, use dropout to simulate disappearance
        if training:
            # Random dropout of track queries
            keep_mask = torch.rand(B, N_q, device=hidden_states.device) > self.p_drop
            # Only keep queries with reasonable scores
            score_mask = scores > 0.5
            active_mask = keep_mask & score_mask
        else:
            # During inference, use score threshold
            active_mask = scores > 0.5

        # Project to track queries
        track_queries = self.track_query_proj(hidden_states)
        track_pos = self.track_pos_proj(query_pos)

        return track_queries, track_pos, active_mask


class TrajectoryAwareLabelAssignment:
    """
    Trajectory-Aware Label Assignment (TALA).

    For detect queries: Hungarian matching with newborn targets only.
    For track queries: identity-consistent assignment from previous frame.
    """

    def __init__(self, cls_weight: float = 2.0, l1_weight: float = 5.0,
                 giou_weight: float = 2.0):
        self.cls_weight = cls_weight
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight

    @torch.no_grad()
    def assign(
        self,
        outputs_class: torch.Tensor,
        outputs_coord: torch.Tensor,
        targets: Dict,
        num_track_queries: int,
        prev_track_assignment: Optional[Dict] = None,
    ) -> Dict:
        """
        Perform trajectory-aware label assignment.

        Args:
            outputs_class: [B, N_q, num_classes] predicted class logits
            outputs_coord: [B, N_q, 4] predicted boxes (cx, cy, w, h)
            targets: dict with keys:
                'labels': [M] class labels for all targets in this frame
                'boxes': [M, 4] ground truth boxes
                'track_ids': [M] track IDs
            num_track_queries: number of track queries (first N are track, rest are detect)
            prev_track_assignment: previous frame's assignment {track_id: query_idx}

        Returns:
            assignment: dict with:
                'matched_query_indices': tensor of matched query indices
                'matched_gt_indices': tensor of matched GT indices
                'track_assignment': {track_id: query_idx} for next frame
        """
        B = outputs_class.shape[0]
        assert B == 1, "TALA currently supports batch_size=1"

        outputs_class = outputs_class[0]  # [N_q, num_classes]
        outputs_coord = outputs_coord[0]  # [N_q, 4]

        gt_labels = targets['labels']      # [M]
        gt_boxes = targets['boxes']        # [M, 4]
        gt_track_ids = targets['track_ids']  # [M]

        num_queries = outputs_class.shape[0]
        num_gts = gt_labels.shape[0]

        if num_gts == 0:
            return {
                'matched_query_indices': torch.tensor([], dtype=torch.long),
                'matched_gt_indices': torch.tensor([], dtype=torch.long),
                'track_assignment': {},
            }

        # Separate track queries and detect queries
        track_query_outputs_class = outputs_class[:num_track_queries]
        track_query_outputs_coord = outputs_coord[:num_track_queries]
        detect_query_outputs_class = outputs_class[num_track_queries:]
        detect_query_outputs_coord = outputs_coord[num_track_queries:]

        # --- Track query assignment (identity-consistent) ---
        track_matched_q = []
        track_matched_g = []
        tracked_gt_indices = set()

        if prev_track_assignment is not None and num_track_queries > 0:
            for track_id, prev_q_idx in prev_track_assignment.items():
                # Find this track_id in current GT
                gt_mask = (gt_track_ids == track_id)
                if gt_mask.any():
                    gt_idx = gt_mask.nonzero(as_tuple=True)[0][0].item()
                    # The track query index in current frame
                    # (it should map to the same relative position)
                    if prev_q_idx < num_track_queries:
                        track_matched_q.append(prev_q_idx)
                        track_matched_g.append(gt_idx)
                        tracked_gt_indices.add(gt_idx)

        # --- Detect query assignment (Hungarian matching with newborn targets) ---
        # Newborn targets = GT targets not already tracked
        newborn_gt_indices = [
            i for i in range(num_gts) if i not in tracked_gt_indices
        ]

        detect_matched_q = []
        detect_matched_g = []

        if len(newborn_gt_indices) > 0 and detect_query_outputs_class.shape[0] > 0:
            newborn_gt_labels = gt_labels[newborn_gt_indices]
            newborn_gt_boxes = gt_boxes[newborn_gt_indices]

            # Compute matching cost
            cost_class = -detect_query_outputs_class[:, newborn_gt_labels].softmax(-1)
            cost_l1 = torch.cdist(
                detect_query_outputs_coord, newborn_gt_boxes, p=1
            )
            cost_giou = -self._generalized_box_iou(
                detect_query_outputs_coord, newborn_gt_boxes
            )

            cost_matrix = (
                self.cls_weight * cost_class +
                self.l1_weight * cost_l1 +
                self.giou_weight * cost_giou
            )

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(
                cost_matrix.cpu().numpy()
            )

            for r, c in zip(row_ind, col_ind):
                detect_matched_q.append(r + num_track_queries)  # offset by track queries
                detect_matched_g.append(newborn_gt_indices[c])

        # Combine assignments
        all_matched_q = track_matched_q + detect_matched_q
        all_matched_g = track_matched_g + detect_matched_g

        # Build track assignment for next frame
        new_track_assignment = {}
        for q_idx, g_idx in zip(all_matched_q, all_matched_g):
            tid = gt_track_ids[g_idx].item()
            new_track_assignment[tid] = q_idx

        device = outputs_class.device
        return {
            'matched_query_indices': torch.tensor(all_matched_q, dtype=torch.long, device=device),
            'matched_gt_indices': torch.tensor(all_matched_g, dtype=torch.long, device=device),
            'track_assignment': new_track_assignment,
        }

    @staticmethod
    def _generalized_box_iou(boxes1, boxes2):
        """
        Compute generalized IoU between two sets of boxes.
        boxes format: (cx, cy, w, h)
        """
        # Convert to (x1, y1, x2, y2)
        b1 = torch.stack([
            boxes1[:, 0] - boxes1[:, 2] / 2,
            boxes1[:, 1] - boxes1[:, 3] / 2,
            boxes1[:, 0] + boxes1[:, 2] / 2,
            boxes1[:, 1] + boxes1[:, 3] / 2,
        ], dim=-1)
        b2 = torch.stack([
            boxes2[:, 0] - boxes2[:, 2] / 2,
            boxes2[:, 1] - boxes2[:, 3] / 2,
            boxes2[:, 0] + boxes2[:, 2] / 2,
            boxes2[:, 1] + boxes2[:, 3] / 2,
        ], dim=-1)

        # Intersection
        inter_x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
        inter_y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
        inter_x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
        inter_y2 = torch.min(b1[:, None, 3], b2[None, :, 3])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Union
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        union = area1[:, None] + area2[None, :] - inter

        iou = inter / (union + 1e-6)

        # Enclosing box
        enclose_x1 = torch.min(b1[:, None, 0], b2[None, :, 0])
        enclose_y1 = torch.min(b1[:, None, 1], b2[None, :, 1])
        enclose_x2 = torch.max(b1[:, None, 2], b2[None, :, 2])
        enclose_y2 = torch.max(b1[:, None, 3], b2[None, :, 3])
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

        giou = iou - (enclose_area - union) / (enclose_area + 1e-6)
        return giou


class TrackQueryManager(nn.Module):
    """
    Manages the lifecycle of detect queries and track queries.
    Combines QIM and TALA.
    """

    def __init__(self, d_model: int = 256, num_queries: int = 300,
                 p_drop: float = 0.1, p_insert: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # Learnable detect queries
        self.detect_query_embed = nn.Embedding(num_queries, d_model)
        self.detect_query_pos = nn.Embedding(num_queries, d_model)

        # QIM
        self.qim = QueryInteractionModule(d_model, p_drop, p_insert)

        # TALA
        self.tala = TrajectoryAwareLabelAssignment()

        # State
        self._track_queries = None       # [1, N_track, d_model]
        self._track_query_pos = None     # [1, N_track, d_model]
        self._track_assignment = None    # {track_id: query_idx}

    def reset(self):
        """Reset tracking state (for new video sequence)."""
        self._track_queries = None
        self._track_query_pos = None
        self._track_assignment = None

    @property
    def num_track_queries(self) -> int:
        if self._track_queries is None:
            return 0
        return self._track_queries.shape[1]

    def get_queries(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get combined (track + detect) queries for current frame.

        Returns:
            tgt: [1, N_track + N_detect, d_model]
            query_pos: [1, N_track + N_detect, d_model]
        """
        detect_tgt = torch.zeros(
            1, self.num_queries, self.d_model, device=device
        )
        detect_pos = self.detect_query_pos.weight.unsqueeze(0)  # [1, N, d]

        if self._track_queries is not None and self._track_queries.shape[1] > 0:
            tgt = torch.cat([self._track_queries, detect_tgt], dim=1)
            query_pos = torch.cat([self._track_query_pos, detect_pos], dim=1)
        else:
            tgt = detect_tgt
            query_pos = detect_pos

        return tgt, query_pos

    def update(
        self,
        hidden_states: torch.Tensor,
        query_pos: torch.Tensor,
        outputs_class: torch.Tensor,
        outputs_coord: torch.Tensor,
        targets: Optional[Dict] = None,
        training: bool = True,
    ) -> Optional[Dict]:
        """
        Update track queries after decoder processing.

        Args:
            hidden_states: [B, N_q, d_model]
            query_pos: [B, N_q, d_model]
            outputs_class: [B, N_q, num_classes]
            outputs_coord: [B, N_q, 4]
            targets: GT annotations (required during training)
            training: whether in training mode

        Returns:
            assignment: label assignment dict (during training)
        """
        # Get max classification scores
        scores = outputs_class.sigmoid().max(dim=-1)[0]  # [B, N_q]

        # Label assignment (training only)
        assignment = None
        if training and targets is not None:
            assignment = self.tala.assign(
                outputs_class, outputs_coord, targets,
                self.num_track_queries, self._track_assignment
            )
            self._track_assignment = assignment['track_assignment']

        # Generate track queries for next frame via QIM
        track_queries, track_pos, active_mask = self.qim(
            hidden_states, query_pos, scores, training
        )

        if training and assignment is not None:
            # During training, use assignment to determine active tracks
            matched_q = assignment['matched_query_indices']
            if len(matched_q) > 0:
                self._track_queries = track_queries[:, matched_q, :]
                self._track_query_pos = track_pos[:, matched_q, :]
            else:
                self._track_queries = None
                self._track_query_pos = None
        else:
            # During inference, use active_mask
            if active_mask.any():
                active_indices = active_mask[0].nonzero(as_tuple=True)[0]
                self._track_queries = track_queries[:, active_indices, :]
                self._track_query_pos = track_pos[:, active_indices, :]
            else:
                self._track_queries = None
                self._track_query_pos = None

        return assignment