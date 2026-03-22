"""
RGBT-Tiny Dataset Loader for DSITT.

RGBT-Tiny dataset structure:
    data/rgbt_tiny/
    ├── images/
    │   ├── DJI_0022_1/
    │   │   ├── 00/  (RGB, 640x512, 3ch)
    │   │   │   ├── 00000.jpg
    │   │   │   └── ...
    │   │   └── 01/  (IR, 640x512, 1ch grayscale)
    │   │       ├── 00000.jpg
    │   │       └── ...
    │   └── ...
    ├── annotations/
    │   ├── instances_00_train2017.json  (RGB train, COCO format)
    │   ├── instances_00_test2017.json
    │   ├── instances_01_train2017.json  (IR train, COCO format)
    │   └── instances_01_test2017.json
    ├── 00_train.txt / 00_test.txt  (RGB image lists)
    ├── 01_train.txt / 01_test.txt  (IR image lists)
    └── train.txt / test.txt         (combined)

Modalities: 00 = RGB (3ch), 01 = IR (1ch grayscale)
Categories: 0=ship, 1=car, 2=cyclist, 3=pedestrian, 4=bus, 5=drone, 6=plane
Annotations: COCO format with tracking_id for MOT
"""

import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


class RGBTTinyDataset(Dataset):
    """
    RGBT-Tiny dataset for multi-object tracking.

    Supports single modality (ir/rgb) or dual modality (both).
    Uses split files and COCO annotations.
    """

    CLASSES = ['ship', 'car', 'cyclist', 'pedestrian', 'bus', 'drone', 'plane']
    NUM_CLASSES = 7

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        modality: str = 'both',       # 'ir', 'rgb', or 'both'
        clip_length: int = 2,
        dummy_img_size: int = 320,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.modality = modality
        self.clip_length = clip_length
        self.dummy_img_size = dummy_img_size
        self.img_dir = os.path.join(data_root, 'images')

        # Try loading real data
        self.is_dummy = False
        self.sequences = []
        self.seq_annotations = {}  # seq_name -> {frame_idx -> [annotations]}

        split_file = os.path.join(data_root, f'00_{split}.txt')
        if os.path.exists(split_file) and os.path.isdir(self.img_dir):
            self._load_real_data(split)
        else:
            print(f"[WARNING] Dataset not found at {data_root}, using dummy data")
            self.is_dummy = True
            self.sequences = self._create_dummy_sequences()

        print(f"[RGBTTinyDataset] {len(self.sequences)} sequences, "
              f"split={split}, modality={modality}, clip_length={clip_length}"
              f"{' (DUMMY)' if self.is_dummy else ''}")

    def _load_real_data(self, split: str):
        """Load real dataset from split files and annotations."""
        # 1. Parse split file to get sequence -> frame list mapping
        split_file = os.path.join(self.data_root, f'00_{split}.txt')
        seq_frames = defaultdict(list)  # seq_name -> [frame_idx, ...]

        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: DJI_0022_1/00/00000
                parts = line.split('/')
                seq_name = parts[0]
                frame_name = parts[2]  # e.g., '00000'
                frame_idx = int(frame_name)
                seq_frames[seq_name].append(frame_idx)

        # Sort frames within each sequence
        for seq_name in seq_frames:
            seq_frames[seq_name] = sorted(set(seq_frames[seq_name]))

        # 2. Load COCO annotations (streaming - only keep what we need)
        ann_file = os.path.join(
            self.data_root, 'annotations',
            f'instances_00_{split}2017.json'
        )
        print(f"[RGBTTinyDataset] Loading annotations from {ann_file}...")
        self._load_coco_annotations(ann_file, seq_frames)

        # 3. Build sequence list
        for seq_name, frames in sorted(seq_frames.items()):
            if len(frames) < self.clip_length:
                continue
            self.sequences.append({
                'name': seq_name,
                'frames': frames,
                'num_frames': len(frames),
                'is_dummy': False,
            })

    def _load_coco_annotations(self, ann_file: str, seq_frames: dict):
        """Load COCO annotations and organize by sequence/frame."""
        if not os.path.exists(ann_file):
            print(f"[WARNING] Annotation file not found: {ann_file}")
            return

        with open(ann_file, 'r') as f:
            data = json.load(f)

        # Build image_id -> (seq_name, frame_idx) mapping
        img_id_to_info = {}
        for img in data['images']:
            file_name = img['file_name']  # e.g., "DJI_0022_1/00/00000.jpg"
            parts = file_name.replace('.jpg', '').split('/')
            seq_name = parts[0]
            frame_idx = int(parts[2])
            img_id_to_info[img['id']] = (seq_name, frame_idx)

        # Organize annotations by sequence and frame
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_id_to_info:
                continue
            seq_name, frame_idx = img_id_to_info[img_id]

            if seq_name not in self.seq_annotations:
                self.seq_annotations[seq_name] = {}
            if frame_idx not in self.seq_annotations[seq_name]:
                self.seq_annotations[seq_name][frame_idx] = []

            self.seq_annotations[seq_name][frame_idx].append({
                'bbox': ann['bbox'],  # [x, y, w, h] in pixels
                'category_id': ann['category_id'],
                'tracking_id': int(ann.get('tracking_id', 0)),
                'area': ann.get('area', 0),
            })

        print(f"[RGBTTinyDataset] Loaded {len(data['annotations'])} annotations "
              f"for {len(data['images'])} images")
        del data  # Free memory

    def _create_dummy_sequences(self) -> List[Dict]:
        """Create dummy sequences for development/testing."""
        sequences = []
        for i in range(10):
            num_frames = random.randint(30, 100)
            sequences.append({
                'name': f'dummy_seq_{i:03d}',
                'frames': list(range(num_frames)),
                'num_frames': num_frames,
                'is_dummy': True,
            })
            # Create dummy annotations
            num_targets = random.randint(3, 8)
            self.seq_annotations[f'dummy_seq_{i:03d}'] = {}
            for f in range(num_frames):
                anns = []
                for t in range(num_targets):
                    cx = 200 + 40 * t + 0.5 * f
                    cy = 150 + 40 * t + 0.3 * f
                    w = random.uniform(5, 30)
                    h = random.uniform(5, 20)
                    anns.append({
                        'bbox': [cx - w/2, cy - h/2, w, h],
                        'category_id': random.randint(0, 6),
                        'tracking_id': t,
                        'area': w * h,
                    })
                self.seq_annotations[f'dummy_seq_{i:03d}'][f] = anns
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def set_clip_length(self, clip_length: int):
        self.clip_length = clip_length

    def __getitem__(self, idx: int) -> Tuple[List, List[Dict]]:
        seq = self.sequences[idx % len(self.sequences)]

        # Sample consecutive frames
        max_start = seq['num_frames'] - self.clip_length
        start = random.randint(0, max(0, max_start))
        frame_indices = seq['frames'][start:start + self.clip_length]

        # Pad if not enough frames
        while len(frame_indices) < self.clip_length:
            frame_indices.append(frame_indices[-1])

        frames = []
        targets = []
        W, H = 640, 512  # RGBT-Tiny image size

        for frame_idx in frame_indices:
            # Load image(s)
            img = self._load_image(seq, frame_idx)
            # Get annotations
            target = self._get_target(seq['name'], frame_idx, W, H)
            frames.append(img)
            targets.append(target)

        return frames, targets

    def _load_image(self, seq: Dict, frame_idx: int):
        """Load image(s) for a frame."""
        if seq.get('is_dummy', False):
            size = self.dummy_img_size
            if self.modality == 'both':
                return (torch.randn(3, size, size), torch.randn(3, size, size))
            return torch.randn(3, size, size)

        seq_name = seq['name']
        fname = f'{frame_idx:05d}.jpg'

        if self.modality == 'both':
            rgb_path = os.path.join(self.img_dir, seq_name, '00', fname)
            ir_path = os.path.join(self.img_dir, seq_name, '01', fname)
            rgb_img = self._read_image(rgb_path, is_rgb=True)
            ir_img = self._read_image(ir_path, is_rgb=False)
            return (rgb_img, ir_img)
        elif self.modality == 'rgb':
            path = os.path.join(self.img_dir, seq_name, '00', fname)
            return self._read_image(path, is_rgb=True)
        else:  # ir
            path = os.path.join(self.img_dir, seq_name, '01', fname)
            return self._read_image(path, is_rgb=False)

    def _read_image(self, path: str, is_rgb: bool = True) -> torch.Tensor:
        """Read and preprocess a single image."""
        if not os.path.exists(path):
            return torch.randn(3, self.dummy_img_size, self.dummy_img_size)

        img = Image.open(path)
        if is_rgb:
            img = img.convert('RGB')
        else:
            # IR is grayscale, convert to 3-channel for backbone compatibility
            img = img.convert('L')
            img = Image.merge('RGB', [img, img, img])

        img = TF.to_tensor(img)  # [3, H, W]
        img = TF.normalize(img,
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        return img

    def _get_target(self, seq_name: str, frame_idx: int,
                    W: int, H: int) -> Dict:
        """Get normalized target annotations for a frame."""
        anns = self.seq_annotations.get(seq_name, {}).get(frame_idx, [])

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

            # Convert pixel coords to normalized [0,1] (cx, cy, w, h)
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H

            # Clamp
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))

            boxes.append([cx, cy, nw, nh])
            labels.append(ann['category_id'])
            track_ids.append(ann['tracking_id'])

        return {
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'track_ids': torch.tensor(track_ids, dtype=torch.long),
        }


def collate_fn(batch):
    """Custom collate for video clips with single/dual modality."""
    frames_list, targets_list = zip(*batch)

    if len(frames_list) == 1:
        raw_frames = frames_list[0]
        if isinstance(raw_frames[0], (tuple, list)):
            frames = [(rgb.unsqueeze(0), ir.unsqueeze(0))
                      for rgb, ir in raw_frames]
        else:
            frames = [f.unsqueeze(0) for f in raw_frames]
        targets = targets_list[0]
        return frames, targets

    clip_length = len(frames_list[0])
    is_dual = isinstance(frames_list[0][0], (tuple, list))

    frames = []
    for t in range(clip_length):
        if is_dual:
            rgb_batch = torch.stack([frames_list[b][t][0] for b in range(len(frames_list))])
            ir_batch = torch.stack([frames_list[b][t][1] for b in range(len(frames_list))])
            frames.append((rgb_batch, ir_batch))
        else:
            frame_batch = torch.stack([frames_list[b][t] for b in range(len(frames_list))])
            frames.append(frame_batch)

    targets = []
    for t in range(clip_length):
        targets.append([targets_list[b][t] for b in range(len(frames_list))])

    return frames, targets


def build_rgbt_tiny_dataset(
    data_root: str = 'data/rgbt_tiny',
    split: str = 'train',
    modality: str = 'both',
    clip_length: int = 2,
    batch_size: int = 1,
    num_workers: int = 4,
    dummy_img_size: int = 320,
) -> Tuple[Dataset, DataLoader]:
    """Build RGBT-Tiny dataset and dataloader."""
    dataset = RGBTTinyDataset(
        data_root=data_root,
        split=split,
        modality=modality,
        clip_length=clip_length,
        dummy_img_size=dummy_img_size,
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