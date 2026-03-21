"""
DSITT v2: Dual-Stream Infrared Tiny Target Tracker with MTUQ.

Key differences from v1 (dsitt.py):
- Dual-stream backbone (RGB + IR)
- Dual-stream encoder (independent per modality)
- Modality-Aware Decoder (MAD) with four-view queries
- MTUQ query management
- Supports both single-modality and dual-modality modes

This model implements the core innovation of the paper:
"Query-level cross-modal fusion" via Modality-Temporal Unified Queries.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .backbone.dual_stream import build_dual_stream_backbone
from .backbone.resnet import build_backbone
from .encoder.deformable_encoder import DeformableTransformerEncoder
from .decoder.modality_aware_decoder import ModalityAwareDecoder
from .tracking.mtuq_manager import MTUQManager
from .tracking.motion_view import MotionViewUpdater, TrajectoryMemoryBank
from .loss.losses import DSITTLoss
from .loss.cmc_loss import CMCLoss
from .decoder.scale_adaptive_attn import scale_diversity_loss


class DSITTv2(nn.Module):
    """
    DSITT v2 with Modality-Temporal Unified Queries (MTUQ).

    Supports:
    - modality='both': dual-stream RGB+IR with MAD decoder
    - modality='ir'/'rgb': single-stream with MAD (other modality zeroed)
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
        box_loss_type: str = 'nwd',
        nwd_constant: float = 0.1,
        modality_dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_classes = num_classes
        self.num_queries = num_queries

        # Dual-stream backbone
        self.dual_backbone = build_dual_stream_backbone(
            d_model=d_model,
            pretrained=backbone_pretrained,
            modality_dropout=modality_dropout,
        )

        # Dual-stream encoders (independent)
        self.encoder_rgb = DeformableTransformerEncoder(
            d_model=d_model, d_ffn=dim_feedforward, dropout=dropout,
            n_levels=num_feature_levels, n_heads=nhead, n_points=4,
            num_layers=num_encoder_layers,
        )
        self.encoder_ir = DeformableTransformerEncoder(
            d_model=d_model, d_ffn=dim_feedforward, dropout=dropout,
            n_levels=num_feature_levels, n_heads=nhead, n_points=4,
            num_layers=num_encoder_layers,
        )

        # Modality-Aware Decoder (MAD)
        self.decoder = ModalityAwareDecoder(
            d_model=d_model, d_ffn=dim_feedforward, dropout=dropout,
            n_levels=num_feature_levels, n_heads=nhead, n_points=4,
            num_layers=num_decoder_layers, num_classes=num_classes,
        )

        # MTUQ Query Manager
        self.mtuq_manager = MTUQManager(
            d_model=d_model, num_queries=num_queries,
            p_drop=p_drop, match_cost_type=box_loss_type,
            nwd_constant=nwd_constant,
        )

        # Loss (reuse from v1, compatible)
        self.criterion = DSITTLoss(
            num_classes=num_classes,
            box_loss_type=box_loss_type,
            nwd_constant=nwd_constant,
        )

        # CMC Loss (cross-modal consistency)
        self.cmc_criterion = CMCLoss(
            consistency_weight=1.0,
            contrastive_weight=0.5,
            temperature=0.07,
        )
        self.use_cmc = True  # can be disabled for ablation

        # Motion View Updater (Stage 5)
        self.motion_updater = MotionViewUpdater(
            d_model=d_model, max_history=5, n_heads=4,
        )
        self.memory_bank = TrajectoryMemoryBank(max_length=5)

    def forward_single_frame(
        self,
        img_rgb: torch.Tensor,
        img_ir: torch.Tensor,
    ):
        """
        Process a single frame pair (RGB + IR).

        Args:
            img_rgb: [B, 3, H, W]
            img_ir: [B, 3, H, W]

        Returns:
            queries: dict of four query views
            query_pos: [B, N_q, d]
            outputs_class: [B, N_q, num_classes]
            outputs_coord: [B, N_q, 4]
            gate_weights: list of [B, N_q, 3] per decoder layer
        """
        # 1. Dual-stream feature extraction
        srcs_rgb, pos_rgb, srcs_ir, pos_ir = self.dual_backbone(img_rgb, img_ir)

        # 2. Dual-stream encoding
        memory_rgb, shapes_rgb, starts_rgb = self.encoder_rgb(srcs_rgb, pos_rgb)
        memory_ir, shapes_ir, starts_ir = self.encoder_ir(srcs_ir, pos_ir)

        # 3. Get MTUQ queries
        queries, query_pos = self.mtuq_manager.get_queries(img_rgb.device)

        # 4. MAD decoding (with SAS scale params)
        queries, outputs_class, outputs_coord, ref_points, gate_weights, scale_params = \
            self.decoder(
                queries, query_pos,
                memory_rgb, shapes_rgb, starts_rgb,
                memory_ir, shapes_ir, starts_ir,
            )

        return queries, query_pos, outputs_class, outputs_coord, gate_weights, scale_params

    def forward(
        self,
        frames_rgb: List[torch.Tensor],
        frames_ir: List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Process a video clip (list of RGB+IR frame pairs).

        Args:
            frames_rgb: list of [B, 3, H, W] RGB tensors
            frames_ir: list of [B, 3, H, W] IR tensors
            targets: list of per-frame target dicts (training only)

        Returns:
            During training: loss dict
            During inference: predictions list
        """
        training = self.training
        self.mtuq_manager.reset()
        self.memory_bank.reset()

        frame_outputs = []
        frame_assignments = []

        for t in range(len(frames_rgb)):
            img_rgb = frames_rgb[t]
            img_ir = frames_ir[t]

            # Forward single frame
            queries, query_pos, outputs_class, outputs_coord, gate_weights, scale_params = \
                self.forward_single_frame(img_rgb, img_ir)

            # Motion view update from trajectory memory
            # Only track queries benefit from motion history
            n_track = self.mtuq_manager.num_track_queries
            if n_track > 0 and self.memory_bank.length > 0:
                hist_feats, hist_boxes = self.memory_bank.get_history()
                if hist_feats is not None and hist_feats.shape[2] == n_track:
                    # Track count matches memory → can use motion updater
                    q_motion_track = queries['q_motion'][:, :n_track]
                    q_motion_updated = self.motion_updater(
                        q_motion_track, hist_feats, hist_boxes
                    )
                    queries['q_motion'] = torch.cat([
                        q_motion_updated,
                        queries['q_motion'][:, n_track:]
                    ], dim=1)
                else:
                    # Track count changed → reset memory
                    self.memory_bank.reset()

            # Push track queries to memory (only if we have tracks)
            if n_track > 0:
                self.memory_bank.push(
                    queries['q_fused'][:, :n_track].detach(),
                    outputs_coord[:, :n_track].detach(),
                )
            else:
                self.memory_bank.reset()

            frame_output = {
                'pred_logits': outputs_class,
                'pred_boxes': outputs_coord,
                'queries': queries,
                'gate_weights': gate_weights,
                'scale_params': scale_params,
            }
            frame_outputs.append(frame_output)

            # Update MTUQ queries
            frame_target = targets[t] if targets is not None else None
            assignment = self.mtuq_manager.update(
                queries, query_pos,
                outputs_class, outputs_coord,
                frame_target, training
            )
            frame_assignments.append(assignment)

        if training:
            valid_assignments = [a for a in frame_assignments if a is not None]
            loss_dict = self.criterion(frame_outputs, targets, valid_assignments)

            # CMC Loss
            if self.use_cmc and len(valid_assignments) > 0:
                cmc_dict = self.cmc_criterion(
                    frame_outputs, valid_assignments,
                    class_head=self.decoder.class_head,
                    bbox_head=self.decoder.bbox_head,
                )
                loss_dict['loss'] = loss_dict['loss'] + cmc_dict['loss_cmc']
                loss_dict.update(cmc_dict)

            # Scale diversity loss (SAS regularization)
            all_scale_params = [fo['scale_params'] for fo in frame_outputs
                                if fo['scale_params'] is not None]
            if all_scale_params:
                loss_scale_div = sum(
                    scale_diversity_loss(sp) for sp in all_scale_params
                ) / len(all_scale_params)
                loss_dict['loss'] = loss_dict['loss'] + 0.1 * loss_scale_div
                loss_dict['loss_scale_div'] = loss_scale_div

                # Log average scale param for monitoring
                avg_scale = torch.cat([sp.mean(dim=1) for sp in all_scale_params]).mean()
                loss_dict['avg_scale'] = avg_scale

            # Add average gate weights for monitoring
            all_gates = []
            for fo in frame_outputs:
                for gw in fo['gate_weights']:
                    all_gates.append(gw.mean(dim=(0, 1)))  # [3]
            if all_gates:
                avg_gates = torch.stack(all_gates).mean(0)
                loss_dict['gate_rgb'] = avg_gates[0]
                loss_dict['gate_ir'] = avg_gates[1]
                loss_dict['gate_motion'] = avg_gates[2]

            return loss_dict
        else:
            predictions = []
            for output in frame_outputs:
                scores = output['pred_logits'].sigmoid().max(dim=-1)
                pred = {
                    'scores': scores[0][0],
                    'labels': scores[1][0],
                    'boxes': output['pred_boxes'][0],
                    'gate_weights': output['gate_weights'][-1][0],  # last layer gates
                }
                predictions.append(pred)
            return {'predictions': predictions}


def build_dsitt_v2(config: Optional[Dict] = None) -> DSITTv2:
    """Build DSITT v2 model from config."""
    if config is None:
        config = {}

    model_cfg = config.get('model', {})
    tracking_cfg = config.get('tracking', {})
    loss_cfg = config.get('loss', {})

    return DSITTv2(
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
        box_loss_type=loss_cfg.get('box_loss_type', 'nwd'),
        nwd_constant=loss_cfg.get('nwd_constant', 0.1),
        modality_dropout=model_cfg.get('modality_dropout', 0.1),
    )