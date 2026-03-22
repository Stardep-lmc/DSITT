#!/usr/bin/env python3
"""
DSITT Evaluation Script.

Runs inference on test sequences and computes detection + tracking metrics.

Usage:
    python tools/eval.py --config configs/dsitt_full.yaml --checkpoint outputs/checkpoints/checkpoint_0010.pth
    python tools/eval.py --config configs/dsitt_full.yaml --checkpoint outputs/checkpoints/checkpoint_0010.pth --visualize
"""

import sys
import os
import argparse
import time
import yaml
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

from models.dsitt import build_dsitt
from models.dsitt_v2 import build_dsitt_v2
from datasets.rgbt_tiny import build_rgbt_tiny_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='DSITT Evaluation')
    parser.add_argument('--config', type=str, default='configs/dsitt_full.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='data/rgbt_tiny')
    parser.add_argument('--output_dir', type=str, default='outputs/eval')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--score_threshold', type=float, default=0.3)
    parser.add_argument('--visualize', action='store_true')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def box_cxcywh_to_xyxy(boxes):
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def compute_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes in xyxy format."""
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-6)


class MOTMetrics:
    """Simple MOT metrics calculator."""

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.total_gt = 0
        self.total_pred = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_id_switches = 0
        self.prev_matches = {}  # gt_id -> pred_id mapping from previous frame
        self.frame_count = 0

    def update(self, pred_boxes, pred_scores, pred_labels,
               gt_boxes, gt_labels, gt_track_ids,
               score_threshold=0.3):
        """Update metrics for one frame."""
        self.frame_count += 1

        # Filter by score
        mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]

        num_gt = gt_boxes.shape[0]
        num_pred = pred_boxes.shape[0]
        self.total_gt += num_gt
        self.total_pred += num_pred

        if num_gt == 0 and num_pred == 0:
            return
        if num_gt == 0:
            self.total_fp += num_pred
            return
        if num_pred == 0:
            self.total_fn += num_gt
            return

        # Compute IoU
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)
        iou_matrix = compute_iou(pred_xyxy, gt_xyxy)

        # Greedy matching
        matched_gt = set()
        matched_pred = set()
        current_matches = {}  # gt_id -> pred_idx

        # Sort by IoU (descending)
        iou_flat = iou_matrix.flatten()
        sorted_idx = iou_flat.argsort(descending=True)

        for flat_idx in sorted_idx:
            pred_idx = flat_idx // num_gt
            gt_idx = flat_idx % num_gt
            pred_idx = pred_idx.item()
            gt_idx = gt_idx.item()

            if pred_idx in matched_pred or gt_idx in matched_gt:
                continue
            if iou_matrix[pred_idx, gt_idx] < self.iou_threshold:
                break

            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
            gt_id = gt_track_ids[gt_idx].item()
            current_matches[gt_id] = pred_idx

        tp = len(matched_gt)
        fp = num_pred - tp
        fn = num_gt - tp

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn

        # Count ID switches
        for gt_id, pred_idx in current_matches.items():
            if gt_id in self.prev_matches:
                if self.prev_matches[gt_id] != pred_idx:
                    self.total_id_switches += 1

        self.prev_matches = current_matches

    def compute(self):
        """Compute final metrics."""
        precision = self.total_tp / max(self.total_tp + self.total_fp, 1)
        recall = self.total_tp / max(self.total_tp + self.total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        # MOTA = 1 - (FN + FP + IDS) / GT
        mota = 1.0 - (self.total_fn + self.total_fp + self.total_id_switches) / max(self.total_gt, 1)

        # IDF1 approximation (simplified)
        idf1 = 2 * self.total_tp / max(2 * self.total_tp + self.total_fp + self.total_fn, 1)

        return {
            'MOTA': mota,
            'IDF1': idf1,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'TP': self.total_tp,
            'FP': self.total_fp,
            'FN': self.total_fn,
            'IDS': self.total_id_switches,
            'GT': self.total_gt,
            'Pred': self.total_pred,
            'Frames': self.frame_count,
        }


@torch.no_grad()
def evaluate(model, dataloader, device, score_threshold=0.3):
    """Run evaluation on test set."""
    model.eval()
    metrics = MOTMetrics(iou_threshold=0.5)

    total_time = 0
    num_frames = 0

    for batch_idx, (frames, targets) in enumerate(dataloader):
        # Move to device
        if isinstance(frames[0], (tuple, list)):
            frames_rgb = [f[0].to(device) for f in frames]
            frames_ir = [f[1].to(device) for f in frames]
        else:
            frames_moved = [f.to(device) for f in frames]
            frames_rgb = frames_moved
            frames_ir = frames_moved

        t0 = time.time()

        # Forward
        if hasattr(model, 'dual_backbone'):
            outputs = model(frames_rgb, frames_ir)
        else:
            outputs = model(frames_rgb)

        t1 = time.time()
        total_time += (t1 - t0)

        # Process predictions
        predictions = outputs.get('predictions', [])
        for t, pred in enumerate(predictions):
            scores = pred['scores']
            labels = pred['labels']
            boxes = pred['boxes']

            # Get GT
            if t < len(targets):
                gt = targets[t]
                gt_boxes = gt['boxes']
                gt_labels = gt['labels']
                gt_track_ids = gt['track_ids']
            else:
                gt_boxes = torch.zeros(0, 4)
                gt_labels = torch.zeros(0, dtype=torch.long)
                gt_track_ids = torch.zeros(0, dtype=torch.long)

            metrics.update(
                boxes.cpu(), scores.cpu(), labels.cpu(),
                gt_boxes, gt_labels, gt_track_ids,
                score_threshold=score_threshold,
            )
            num_frames += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Evaluated {batch_idx + 1}/{len(dataloader)} sequences "
                  f"({num_frames} frames)")

    # Compute final metrics
    results = metrics.compute()
    if num_frames > 0:
        results['FPS'] = num_frames / max(total_time, 1e-6)
    else:
        results['FPS'] = 0.0

    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get('data', {})

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build model
    print("\n=== Building Model ===")
    model_version = config.get('model', {}).get('version', 'v1')
    if model_version == 'v2':
        model = build_dsitt_v2(config)
    else:
        model = build_dsitt(config)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {num_params:.1f}M")

    # Build test dataset
    print("\n=== Building Test Dataset ===")
    modality = data_cfg.get('modality', 'both')
    dataset, dataloader = build_rgbt_tiny_dataset(
        data_root=args.data_root,
        split='test',
        modality=modality,
        clip_length=2,
        batch_size=1,
        num_workers=0,
    )

    # Evaluate
    print(f"\n=== Evaluating ({len(dataset)} sequences) ===")
    results = evaluate(model, dataloader, device,
                       score_threshold=args.score_threshold)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  MOTA:      {results['MOTA']:.4f}")
    print(f"  IDF1:      {results['IDF1']:.4f}")
    print(f"  Precision: {results['Precision']:.4f}")
    print(f"  Recall:    {results['Recall']:.4f}")
    print(f"  F1:        {results['F1']:.4f}")
    print(f"  IDS:       {results['IDS']}")
    print(f"  TP/FP/FN:  {results['TP']}/{results['FP']}/{results['FN']}")
    print(f"  GT:        {results['GT']}")
    print(f"  FPS:       {results['FPS']:.1f}")
    print(f"  Frames:    {results['Frames']}")
    print("=" * 60)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()