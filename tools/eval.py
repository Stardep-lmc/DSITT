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
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def hungarian_match(iou_matrix, iou_threshold):
    """Hungarian matching on IoU matrix. Returns matched (pred_idx, gt_idx) pairs."""
    if iou_matrix.numel() == 0:
        return [], [], set(), set()
    cost = 1.0 - iou_matrix.numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_pred, matched_gt = [], []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matched_pred.append(r)
            matched_gt.append(c)
    return matched_pred, matched_gt, set(matched_pred), set(matched_gt)


class MOTMetrics:
    """MOT metrics: MOTA, IDF1, HOTA, IDS with Hungarian matching."""

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
        self.prev_matches = {}  # gt_id -> pred_id
        self.frame_count = 0
        # For IDF1: track-level TP counts
        self.gt_id_tp = defaultdict(int)   # gt_id -> matched frame count
        self.gt_id_total = defaultdict(int) # gt_id -> total frame count
        self.pred_id_tp = defaultdict(int)  # pred_id -> matched frame count
        self.pred_id_total = defaultdict(int)
        # For HOTA: per-frame (DetA, AssA) at multiple thresholds
        self.hota_thresholds = np.arange(0.05, 1.0, 0.05)
        self.hota_per_thresh = {t: {'det_tp': 0, 'det_fp': 0, 'det_fn': 0,
                                     'ass_scores': []} for t in self.hota_thresholds}
        # Association tracking: gt_id -> set of matched pred_ids across frames
        self.gt_pred_history = defaultdict(lambda: defaultdict(int))  # gt_id -> {pred_id: count}
        self.gt_frame_count = defaultdict(int)

    def update(self, pred_boxes, pred_scores, pred_labels,
               gt_boxes, gt_labels, gt_track_ids,
               score_threshold=0.3):
        """Update metrics for one frame."""
        self.frame_count += 1

        mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]

        num_gt = gt_boxes.shape[0]
        num_pred = pred_boxes.shape[0]
        self.total_gt += num_gt
        self.total_pred += num_pred

        # Track GT presence
        for i in range(num_gt):
            gid = gt_track_ids[i].item()
            self.gt_id_total[gid] += 1
            self.gt_frame_count[gid] += 1

        if num_gt == 0 and num_pred == 0:
            return
        if num_gt == 0:
            self.total_fp += num_pred
            # HOTA: all preds are FP at every threshold
            for t in self.hota_thresholds:
                self.hota_per_thresh[t]['det_fp'] += num_pred
            return
        if num_pred == 0:
            self.total_fn += num_gt
            for t in self.hota_thresholds:
                self.hota_per_thresh[t]['det_fn'] += num_gt
            return

        # IoU matrix
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)
        iou_matrix = compute_iou(pred_xyxy, gt_xyxy)

        # Hungarian matching at primary threshold
        m_pred, m_gt, m_pred_set, m_gt_set = hungarian_match(
            iou_matrix, self.iou_threshold)

        tp = len(m_pred)
        fp = num_pred - tp
        fn = num_gt - tp
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn

        # Build current gt_id -> pred_idx mapping
        current_matches = {}
        for pi, gi in zip(m_pred, m_gt):
            gid = gt_track_ids[gi].item()
            current_matches[gid] = pi
            self.gt_id_tp[gid] += 1
            self.gt_pred_history[gid][pi] += 1

        # ID switches
        for gid, pid in current_matches.items():
            if gid in self.prev_matches and self.prev_matches[gid] != pid:
                self.total_id_switches += 1
        self.prev_matches = current_matches

        # HOTA: evaluate at multiple IoU thresholds
        for t in self.hota_thresholds:
            mp, mg, _, _ = hungarian_match(iou_matrix, t)
            t_tp = len(mp)
            self.hota_per_thresh[t]['det_tp'] += t_tp
            self.hota_per_thresh[t]['det_fp'] += num_pred - t_tp
            self.hota_per_thresh[t]['det_fn'] += num_gt - t_tp
            # Per-match association score (simplified: use IoU as proxy)
            for pi, gi in zip(mp, mg):
                self.hota_per_thresh[t]['ass_scores'].append(
                    iou_matrix[pi, gi].item())

    def compute(self):
        """Compute final metrics."""
        precision = self.total_tp / max(self.total_tp + self.total_fp, 1)
        recall = self.total_tp / max(self.total_tp + self.total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        # MOTA
        mota = 1.0 - (self.total_fn + self.total_fp + self.total_id_switches) / max(self.total_gt, 1)

        # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
        # IDTP: sum of min(gt_tp, pred_tp) for each gt-pred pair
        # Simplified: use gt_id_tp as IDTP proxy
        idtp = sum(self.gt_id_tp.values())
        idfn = sum(self.gt_id_total.values()) - idtp
        idfp = self.total_pred - idtp
        idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)

        # HOTA = mean over thresholds of sqrt(DetA * AssA)
        hota_values = []
        for t in self.hota_thresholds:
            h = self.hota_per_thresh[t]
            det_tp = h['det_tp']
            deta = det_tp / max(det_tp + h['det_fp'] + h['det_fn'], 1)
            # AssA: average association score for matched pairs
            if h['ass_scores']:
                assa = np.mean(h['ass_scores'])
            else:
                assa = 0.0
            hota_values.append(np.sqrt(deta * assa))
        hota = np.mean(hota_values) if hota_values else 0.0

        # DetA and AssA at primary threshold (0.5)
        h50 = self.hota_per_thresh[0.5] if 0.5 in self.hota_per_thresh else None
        if h50:
            deta = h50['det_tp'] / max(h50['det_tp'] + h50['det_fp'] + h50['det_fn'], 1)
            assa = np.mean(h50['ass_scores']) if h50['ass_scores'] else 0.0
        else:
            deta, assa = 0.0, 0.0

        return {
            'HOTA': float(hota),
            'MOTA': mota,
            'IDF1': idf1,
            'DetA': float(deta),
            'AssA': float(assa),
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


CLASSES = ['ship', 'car', 'cyclist', 'pedestrian', 'bus', 'drone', 'plane']
# Distinct colors for up to 20 track IDs (cycled)
TRACK_COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#FF8000', '#8000FF', '#0080FF', '#FF0080', '#80FF00', '#00FF80',
    '#FF4040', '#40FF40', '#4040FF', '#FFA500', '#A500FF', '#00A5FF',
    '#FF6666', '#66FF66',
]


def visualize_frame(image_tensor, pred_boxes, pred_scores, pred_labels,
                    gt_boxes, gt_labels, score_threshold, save_path,
                    img_w=640, img_h=512):
    """
    Draw predicted and GT boxes on an image and save to file.

    Args:
        image_tensor: [3, H, W] normalized tensor (or None for dummy)
        pred_boxes: [N, 4] (cx, cy, w, h) normalized [0,1]
        pred_scores: [N] confidence scores
        pred_labels: [N] class indices
        gt_boxes: [M, 4] (cx, cy, w, h) normalized [0,1]
        gt_labels: [M] class indices
        score_threshold: filter predictions below this
        save_path: output file path
        img_w, img_h: image dimensions for denormalization
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Try to show the image (denormalize)
    if image_tensor is not None and image_tensor.dim() == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = image_tensor.cpu() * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        img_h, img_w = img.shape[:2]
    else:
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.set_facecolor('black')

    # Draw GT boxes (green dashed)
    for i in range(gt_boxes.shape[0]):
        cx, cy, w, h = gt_boxes[i].tolist()
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        bw = w * img_w
        bh = h * img_h
        cls_name = CLASSES[gt_labels[i].item()] if gt_labels[i].item() < len(CLASSES) else '?'
        rect = patches.Rectangle((x1, y1), bw, bh, linewidth=1.5,
                                  edgecolor='lime', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(x1, y1 - 2, f'GT:{cls_name}', fontsize=6, color='lime',
                backgroundcolor=(0, 0, 0, 0.4))

    # Draw predicted boxes (colored by index, solid)
    mask = pred_scores >= score_threshold
    for i in range(pred_boxes.shape[0]):
        if not mask[i]:
            continue
        cx, cy, w, h = pred_boxes[i].tolist()
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        bw = w * img_w
        bh = h * img_h
        color = TRACK_COLORS[i % len(TRACK_COLORS)]
        cls_name = CLASSES[pred_labels[i].item()] if pred_labels[i].item() < len(CLASSES) else '?'
        score = pred_scores[i].item()
        rect = patches.Rectangle((x1, y1), bw, bh, linewidth=2,
                                  edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 2, f'{cls_name}:{score:.2f}', fontsize=6, color=color,
                backgroundcolor=(0, 0, 0, 0.4))

    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, dataloader, device, score_threshold=0.3,
             visualize=False, vis_dir=None):
    """Run evaluation on test set."""
    model.eval()
    metrics = MOTMetrics(iou_threshold=0.5)

    if visualize and vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

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

            # Visualization
            if visualize and vis_dir and num_frames < 200:
                # Get image tensor for visualization
                img_tensor = None
                if isinstance(frames[0], (tuple, list)):
                    img_tensor = frames[t][0][0].cpu() if t < len(frames) else None
                elif t < len(frames):
                    img_tensor = frames[t][0].cpu()
                vis_path = os.path.join(vis_dir, f'frame_{num_frames:05d}.png')
                visualize_frame(
                    img_tensor, boxes.cpu(), scores.cpu(), labels.cpu(),
                    gt_boxes, gt_labels, score_threshold, vis_path,
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
    vis_dir = os.path.join(args.output_dir, 'visualizations') if args.visualize else None
    print(f"\n=== Evaluating ({len(dataset)} sequences) ===")
    if args.visualize:
        print(f"  Visualization enabled, saving to {vis_dir} (first 200 frames)")
    results = evaluate(model, dataloader, device,
                       score_threshold=args.score_threshold,
                       visualize=args.visualize, vis_dir=vis_dir)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  HOTA:      {results['HOTA']:.4f}")
    print(f"  MOTA:      {results['MOTA']:.4f}")
    print(f"  IDF1:      {results['IDF1']:.4f}")
    print(f"  DetA:      {results['DetA']:.4f}")
    print(f"  AssA:      {results['AssA']:.4f}")
    print(f"  Precision: {results['Precision']:.4f}")
    print(f"  Recall:    {results['Recall']:.4f}")
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