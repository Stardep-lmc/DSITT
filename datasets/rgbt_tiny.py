"""
RGBT-Tiny Dataset Loader for DSITT.

RGBT-Tiny dataset structure (expected):
    rgbt_tiny/
    ├── train/
    │   ├── sequence_001/
    │   │   ├── visible/          # RGB images
    │   │   │   ├── 000001.jpg
    │   │   │   ├── 000002.jpg
    │   │   │   └── ...
    │   │   ├── infrared/         # IR images
    │   │   │   ├── 000001.jpg
    │   │   │   └── ...
    │   │   └── annotations.json  # or annotations.txt
    │   └── ...
    └── test/
        └── ...

This module supports:
- Single modality (IR only or RGB only)
- Dual modality (RGB + IR paired)
- Video clip sampling for training
"""

import os
import json
import glob
import random
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


class RGBTTinyDataset(Dataset):
    """
    RGBT-Tiny dataset for multi-object tracking.

    Loads video clips (consecutive frames) with annotations including
    bounding boxes, class labels, and track IDs.
    """

    # Category mapping for RGBT-Tiny (7 classes)
    CLASSES = ['ship', 'car', 'cyclist', 'pedestrian', 'bus', 'drone', 'plane']
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        modality: str = 'ir',         # 'ir', 'rgb', or 'both'
        clip_length: int = 2,
        img_size_min: int = 800,
        img_size_max: int = 1536,
        transforms: Optional[object] = None,
        sample_interval: int = 1,      # frame sampling interval
        max_sample_interval: int = 5,  # max random interval
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.modality = modality
        self.clip_length = clip_length
        self.img_size_min = img_size_min
        self.img_size_max = img_size_max
        self.transforms = transforms
        self.sample_interval = sample_interval
        self.max_sample_interval = max_sample_interval

        # Load sequence information
        self.sequences = self._load_sequences()
        print(f"[RGBTTinyDataset] Loaded {len(self.sequences)} sequences "
              f"from {split} split, modality={modality}, clip_length={clip_length}")

    def _load_sequences(self) -> List[Dict]:
        """Scan dataset directory and load sequence metadata."""
        split_dir = os.path.join(self.data_root, self.split)
        sequences = []

        if not os.path.exists(split_dir):
            print(f"[WARNING] Dataset directory not found: {split_dir}")
            print(f"[WARNING] Using dummy sequences for development.")
            return self._create_dummy_sequences()

        # Scan for sequence directories
        seq_dirs = sorted(glob.glob(os.path.join(split_dir, '*')))
        for seq_dir in seq_dirs:
            if not os.path.isdir(seq_dir):
                continue

            seq_name = os.path.basename(seq_dir)

            # Find image directories
            ir_dir = self._find_image_dir(seq_dir, ['infrared', 'ir', 'thermal'])
            rgb_dir = self._find_image_dir(seq_dir, ['visible', 'rgb', 'vis'])

            if ir_dir is None and rgb_dir is None:
                continue

            # Find annotation file
            ann_file = self._find_annotation_file(seq_dir)

            # Count frames
            img_dir = ir_dir if ir_dir else rgb_dir
            frame_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                                 glob.glob(os.path.join(img_dir, '*.png')))
            num_frames = len(frame_files)

            if num_frames < self.clip_length:
                continue

            # Load annotations
            annotations = self._load_annotations(ann_file, num_frames) if ann_file else {}

            sequences.append({
                'name': seq_name,
                'ir_dir': ir_dir,
                'rgb_dir': rgb_dir,
                'num_frames': num_frames,
                'frame_files': frame_files,
                'annotations': annotations,
            })

        if len(sequences) == 0:
            print(f"[WARNING] No valid sequences found. Using dummy sequences.")
            return self._create_dummy_sequences()

        return sequences

    def _find_image_dir(self, seq_dir: str, candidates: List[str]) -> Optional[str]:
        """Find image directory from candidate names."""
        for name in candidates:
            path = os.path.join(seq_dir, name)
            if os.path.isdir(path):
                return path
        return None

    def _find_annotation_file(self, seq_dir: str) -> Optional[str]:
        """Find annotation file in sequence directory."""
        for name in ['annotations.json', 'annotations.txt', 'gt.txt',
                      'groundtruth.txt', 'labels.json']:
            path = os.path.join(seq_dir, name)
            if os.path.exists(path):
                return path
        return None

    def _load_annotations(self, ann_file: str, num_frames: int) -> Dict:
        """
        Load annotations from file.
        Returns: {frame_idx: [{'bbox': [x,y,w,h], 'label': int, 'track_id': int}, ...]}
        """
        annotations = {}

        if ann_file.endswith('.json'):
            with open(ann_file, 'r') as f:
                data = json.load(f)
            # Parse JSON format (format may vary, adapt as needed)
            if isinstance(data, list):
                for item in data:
                    frame_idx = item.get('frame_id', item.get('frame', 0))
                    if frame_idx not in annotations:
                        annotations[frame_idx] = []
                    annotations[frame_idx].append({
                        'bbox': item.get('bbox', [0, 0, 10, 10]),
                        'label': item.get('category_id', item.get('label', 0)),
                        'track_id': item.get('track_id', item.get('id', 0)),
                    })
        elif ann_file.endswith('.txt'):
            # MOT format: frame_id, track_id, x, y, w, h, conf, class, ...
            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    frame_idx = int(parts[0]) - 1  # 0-indexed
                    track_id = int(parts[1])
                    x, y, w, h = float(parts[2]), float(parts[3]), \
                                 float(parts[4]), float(parts[5])
                    label = int(parts[7]) if len(parts) > 7 else 0

                    if frame_idx not in annotations:
                        annotations[frame_idx] = []
                    annotations[frame_idx].append({
                        'bbox': [x, y, w, h],
                        'label': label,
                        'track_id': track_id,
                    })

        return annotations

    def _create_dummy_sequences(self) -> List[Dict]:
        """Create dummy sequences for development/testing without real data."""
        sequences = []
        for i in range(10):
            num_frames = random.randint(30, 100)
            annotations = {}
            num_targets = random.randint(3, 8)
            for f in range(num_frames):
                annotations[f] = []
                for t in range(num_targets):
                    # Random bbox that moves slightly each frame
                    cx = 0.3 + 0.4 * (t / num_targets) + 0.001 * f
                    cy = 0.3 + 0.4 * ((t + 1) / num_targets) + 0.001 * f
                    w = random.uniform(0.01, 0.05)
                    h = random.uniform(0.01, 0.05)
                    annotations[f].append({
                        'bbox': [cx - w/2, cy - h/2, w, h],
                        'label': random.randint(0, 6),
                        'track_id': t,
                    })

            sequences.append({
                'name': f'dummy_seq_{i:03d}',
                'ir_dir': None,
                'rgb_dir': None,
                'num_frames': num_frames,
                'frame_files': [],
                'annotations': annotations,
            })
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def set_clip_length(self, clip_length: int):
        """Update clip length (for progressive training schedule)."""
        self.clip_length = clip_length

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[Dict]]:
        """
        Sample a video clip from a sequence.

        Returns:
            frames: list of [3, H, W] tensors
            targets: list of target dicts per frame
        """
        seq = self.sequences[idx % len(self.sequences)]

        # Sample start frame and interval
        interval = random.randint(1, min(self.max_sample_interval, self.sample_interval))
        max_start = seq['num_frames'] - self.clip_length * interval
        start_frame = random.randint(0, max(0, max_start))

        frame_indices = [
            min(start_frame + t * interval, seq['num_frames'] - 1)
            for t in range(self.clip_length)
        ]

        frames = []
        targets = []

        for frame_idx in frame_indices:
            # Load image
            img = self._load_image(seq, frame_idx)

            # Get annotations for this frame
            target = self._get_frame_target(seq, frame_idx, img.shape[-2:])

            frames.append(img)
            targets.append(target)

        return frames, targets

    def _load_image(self, seq: Dict, frame_idx: int) -> torch.Tensor:
        """Load image for given sequence and frame index."""
        # Determine which modality to load
        if self.modality == 'ir' and seq['ir_dir']:
            img_dir = seq['ir_dir']
        elif self.modality == 'rgb' and seq['rgb_dir']:
            img_dir = seq['rgb_dir']
        elif seq['ir_dir']:
            img_dir = seq['ir_dir']
        elif seq['rgb_dir']:
            img_dir = seq['rgb_dir']
        else:
            # Dummy mode: generate random image
            return torch.randn(3, self.img_size_min, self.img_size_min)

        # Get frame file
        if frame_idx < len(seq['frame_files']):
            img_path = seq['frame_files'][frame_idx]
        else:
            # Try to construct path
            for ext in ['.jpg', '.png']:
                img_path = os.path.join(img_dir, f'{frame_idx + 1:06d}{ext}')
                if os.path.exists(img_path):
                    break

        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            img = TF.to_tensor(img)  # [3, H, W], values in [0, 1]
            # Normalize with ImageNet stats
            img = TF.normalize(img,
                               mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
            return img
        else:
            # Fallback: random image
            return torch.randn(3, self.img_size_min, self.img_size_min)

    def _get_frame_target(self, seq: Dict, frame_idx: int,
                          img_size: Tuple[int, int]) -> Dict:
        """Get target annotations for a specific frame."""
        H, W = img_size
        anns = seq['annotations'].get(frame_idx, [])

        if len(anns) == 0:
            return {
                'labels': torch.zeros(0, dtype=torch.long),
                'boxes': torch.zeros(0, 4, dtype=torch.float32),
                'track_ids': torch.zeros(0, dtype=torch.long),
            }

        labels = []
        boxes = []
        track_ids = []

        for ann in anns:
            x, y, w, h = ann['bbox']

            # Normalize to [0, 1] if not already
            if x > 1 or y > 1 or w > 1 or h > 1:
                x, y, w, h = x / W, y / W, w / W, h / H  # rough normalization

            # Convert from (x, y, w, h) to (cx, cy, w, h)
            cx = x + w / 2
            cy = y + h / 2

            # Clamp to valid range
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.001, min(1.0, w))
            h = max(0.001, min(1.0, h))

            boxes.append([cx, cy, w, h])
            labels.append(ann['label'])
            track_ids.append(ann['track_id'])

        return {
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'track_ids': torch.tensor(track_ids, dtype=torch.long),
        }


def collate_fn(batch):
    """
    Custom collate function for video clips.
    Each item in batch is (frames, targets) where frames is a list of tensors.

    Since we use batch_size=1 for video tracking, this is straightforward.
    """
    frames_list, targets_list = zip(*batch)

    # For batch_size=1, just take the first item
    # For batch_size>1, we'd need padding (not implemented yet)
    if len(frames_list) == 1:
        frames = [f.unsqueeze(0) for f in frames_list[0]]  # add batch dim
        targets = targets_list[0]
        return frames, targets

    # Batch size > 1: stack frames (requires same size)
    clip_length = len(frames_list[0])
    frames = []
    for t in range(clip_length):
        frame_batch = torch.stack([frames_list[b][t] for b in range(len(frames_list))])
        frames.append(frame_batch)

    # Targets remain as list of lists
    targets = []
    for t in range(clip_length):
        targets.append([targets_list[b][t] for b in range(len(frames_list))])

    return frames, targets


def build_rgbt_tiny_dataset(
    data_root: str = 'data/rgbt_tiny',
    split: str = 'train',
    modality: str = 'ir',
    clip_length: int = 2,
    batch_size: int = 1,
    num_workers: int = 4,
) -> Tuple[Dataset, DataLoader]:
    """Build RGBT-Tiny dataset and dataloader."""

    dataset = RGBTTinyDataset(
        data_root=data_root,
        split=split,
        modality=modality,
        clip_length=clip_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train'),
    )

    return dataset, dataloader