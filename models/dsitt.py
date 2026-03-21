"""
DSITT: Dual-Stream Infrared Tiny Target Tracker

Main model that assembles:
- Backbone (ResNet-50 + FPN)
- Deformable Transformer Encoder
- Deformable Transformer Decoder
- Track Query Manager (QIM + TALA)
- Loss computation (CAL)

This is the baseline (single-modality) version.
Multi-modal MTUQ will be built on top of this in later stages.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .backbone.resnet import build_backbone
from .encoder.deformable_encoder import DeformableTransformerEncoder
from .decoder.deformable_decoder import DeformableTransformerDecoder
from .tracking.track_manager import TrackQueryManager
from .loss.losses import DSITTLoss


class DSITT(nn.Module):
    """
    DSITT baseline model (single modality, IR or RGB).
    Processes a video clip frame-by-frame with track query propagation.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_feature_levels: int = 4,
        num_queries: int = 300,
        num_classes: int = 7,
        backbone_pretrained: bool = True,
        p_drop: float = 0.1,
        p_insert: float = 0.1,
        box_loss_type: str = 'giou',
        nwd_constant: float = 4.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.num_queries = num_queries

        # Backbone
        self.backbone = build_backbone(d_model=d_model, pretrained=backbone_pretrained)

        # Encoder
        self.encoder = DeformableTransformerEncoder(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=4,
            num_layers=num_encoder_layers,
        )

        # Decoder
        self.decoder = DeformableTransformerDecoder(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=4,
            num_layers=num_decoder_layers,
            num_classes=num_classes,
        )

        # Track Query Manager (with NWD-aware matching when configured)
        self.track_manager = TrackQueryManager(
            d_model=d_model,
            num_queries=num_queries,
            p_drop=p_drop,
            p_insert=p_insert,
            match_cost_type=box_loss_type,
            nwd_constant=nwd_constant,
        )

        # Loss (with NWD support)
        self.criterion = DSITTLoss(
            num_classes=num_classes,
            box_loss_type=box_loss_type,
            nwd_constant=nwd_constant,
        )

    def forward_single_frame(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a single frame.

        Args:
            image: [B, 3, H, W]

        Returns:
            hidden_states: [B, N_q, d_model]
            query_pos: [B, N_q, d_model]
            outputs_class: [B, N_q, num_classes]
            outputs_coord: [B, N_q, 4]
        """
        # Extract features
        srcs, pos_embeds = self.backbone(image)

        # Encode
        memory, spatial_shapes, level_start_index = self.encoder(srcs, pos_embeds)

        # Get queries (track + detect)
        tgt, query_pos = self.track_manager.get_queries(image.device)

        # Decode
        hidden_states, outputs_class, outputs_coord, ref_points = self.decoder(
            tgt, query_pos, memory, spatial_shapes, level_start_index
        )

        return hidden_states, query_pos, outputs_class, outputs_coord

    def forward(
        self,
        frames: List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Process a video clip (list of frames).

        Args:
            frames: list of [B, 3, H, W] tensors, one per frame
            targets: list of per-frame target dicts (training only)

        Returns:
            During training: dict with 'loss' and loss components
            During inference: list of per-frame predictions
        """
        training = self.training
        self.track_manager.reset()

        frame_outputs = []
        frame_assignments = []

        for t, frame in enumerate(frames):
            # Forward single frame
            hidden_states, query_pos, outputs_class, outputs_coord = \
                self.forward_single_frame(frame)

            # Store outputs
            frame_output = {
                'pred_logits': outputs_class,
                'pred_boxes': outputs_coord,
                'hidden_states': hidden_states,
            }
            frame_outputs.append(frame_output)

            # Update track queries
            frame_target = targets[t] if targets is not None else None
            assignment = self.track_manager.update(
                hidden_states, query_pos,
                outputs_class, outputs_coord,
                frame_target, training
            )
            frame_assignments.append(assignment)

        if training:
            # Compute collective average loss
            valid_assignments = [a for a in frame_assignments if a is not None]
            loss_dict = self.criterion(frame_outputs, targets, valid_assignments)
            return loss_dict
        else:
            # Return per-frame predictions for evaluation
            predictions = []
            for output in frame_outputs:
                scores = output['pred_logits'].sigmoid().max(dim=-1)
                pred = {
                    'scores': scores[0][0],     # [N_q]
                    'labels': scores[1][0],     # [N_q]
                    'boxes': output['pred_boxes'][0],  # [N_q, 4]
                }
                predictions.append(pred)
            return {'predictions': predictions}


def build_dsitt(config: Optional[Dict] = None) -> DSITT:
    """Build DSITT model from config."""
    if config is None:
        config = {}

    model_cfg = config.get('model', {})
    tracking_cfg = config.get('tracking', {})

    loss_cfg = config.get('loss', {})

    return DSITT(
        d_model=model_cfg.get('d_model', 256),
        nhead=model_cfg.get('nhead', 8),
        num_encoder_layers=model_cfg.get('num_encoder_layers', 6),
        num_decoder_layers=model_cfg.get('num_decoder_layers', 6),
        dim_feedforward=model_cfg.get('dim_feedforward', 1024),
        dropout=model_cfg.get('dropout', 0.1),
        num_feature_levels=model_cfg.get('num_feature_levels', 4),
        num_queries=model_cfg.get('num_queries', 300),
        num_classes=model_cfg.get('num_classes', 7),
        backbone_pretrained=True,
        p_drop=tracking_cfg.get('p_drop', 0.1),
        p_insert=tracking_cfg.get('p_insert', 0.1),
        box_loss_type=loss_cfg.get('box_loss_type', 'giou'),
        nwd_constant=loss_cfg.get('nwd_constant', 4.0),
    )
