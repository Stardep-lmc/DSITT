# DSITT: Dual-Stream Infrared Tiny Target Tracker

A query-centric multi-modal multi-object tracking framework for infrared tiny targets, built on Deformable DETR with modality-temporal unified queries.

## 🏗️ Architecture Overview

```
(img_rgb, img_ir) × T frames
      ↓
[DualStreamBackbone + ModalityDropout]        ← Stage 2
  → (F_rgb, F_ir) × 4 scales
      ↓
[DualStreamEncoder] × 6 layers               ← Stage 0
  → (memory_rgb, memory_ir)
      ↓
[MTUQManager.get_queries]                     ← Stage 2
  → {q_rgb, q_ir, q_motion, q_fused} + pos
      ↓
[MotionViewUpdater(memory_bank)]              ← Stage 5
  → q_motion enriched with trajectory history
      ↓
[ModalityAwareDecoder] × 6 layers            ← Stage 2/4
  Step 1: Self-attention (q_fused)
  Step 2: SAS Cross-attention (q_rgb↔F_rgb, q_ir↔F_ir)
  Step 3: Cross-modal interaction (q_rgb↔q_ir)
  Step 4: Adaptive 3-view fusion → q_fused
      ↓
[Prediction Heads]
  → cls, box (from q_fused)
  → box_rgb, box_ir (auxiliary, for CMC)      ← Stage 3
      ↓
[Loss]
  = FocalLoss + L1 + NWD                      ← Stage 0+1
  + CMC(consistency + contrastive)             ← Stage 3
  + ScaleDiversityLoss                         ← Stage 4
      ↓
[MTUQManager.update + MemoryBank.push]        ← Stage 2+5
  → track queries for next frame
```

**Parameters**: 81.8M | **Loss terms**: 6 | **Classes**: 7 (RGBT-Tiny)

## 📂 Project Structure

```
DSITT/
├── models/
│   ├── dsitt.py              # v1 baseline (single-stream)
│   ├── dsitt_v2.py           # v2 full model (dual-stream + all innovations)
│   ├── backbone/
│   │   ├── resnet.py         # ResNet-50 backbone
│   │   └── dual_stream.py    # Dual-stream backbone with modality dropout
│   ├── encoder/
│   │   └── deformable_encoder.py  # Multi-scale deformable encoder
│   ├── decoder/
│   │   ├── deformable_decoder.py      # v1 decoder
│   │   ├── modality_aware_decoder.py  # MAD: 4-view query fusion
│   │   └── scale_adaptive_attn.py     # SAS: learnable scale sampling
│   ├── tracking/
│   │   ├── track_manager.py    # v1 track management
│   │   ├── mtuq_manager.py     # MTUQ: unified query lifecycle
│   │   └── motion_view.py      # Trajectory encoder + memory bank
│   ├── loss/
│   │   ├── losses.py           # Main loss (Focal + L1 + NWD/GIoU)
│   │   ├── nwd_loss.py         # Normalized Wasserstein Distance
│   │   └── cmc_loss.py         # Cross-Modal Consistency loss
│   └── ops/
│       └── ms_deform_attn.py   # Multi-scale deformable attention
├── datasets/
│   └── rgbt_tiny.py           # RGBT-Tiny dataset loader (single/dual modality)
├── configs/
│   ├── dsitt_base.yaml        # v1 baseline config
│   ├── dsitt_nwd.yaml         # v1 + NWD config
│   ├── dsitt_mtuq.yaml        # v2 MTUQ config
│   └── dsitt_full.yaml        # v2 full config (all stages)
├── tools/
│   ├── train.py               # Training script (v1/v2 auto-detect)
│   └── test_model.py          # Model smoke test
├── analysis/                  # Implementation notes per stage
└── outputs/                   # Checkpoints & logs
```

## 🚀 Quick Start

### Requirements

```bash
pip install torch torchvision pyyaml tensorboard
```

### Development Test (no dataset needed)

```bash
# v2 full model (dual-stream, all innovations)
python tools/train.py --config configs/dsitt_full.yaml --dummy --epochs 5 --print_freq 1

# v1 baseline (single-stream)
python tools/train.py --config configs/dsitt_base.yaml --dummy --epochs 5 --print_freq 1
```

### Training with RGBT-Tiny Dataset

```bash
# Download RGBT-Tiny dataset and place in data/rgbt_tiny/
python tools/train.py --config configs/dsitt_full.yaml --data_root data/rgbt_tiny --epochs 200
```

### Resume Training

```bash
python tools/train.py --config configs/dsitt_full.yaml --resume outputs/checkpoints/checkpoint_0100.pth
```

## 🔬 Key Innovations

### 1. Normalized Wasserstein Distance (NWD) Loss
Replaces GIoU for tiny targets where bbox overlap is unreliable. Models boxes as 2D Gaussians and computes Wasserstein distance.

### 2. Modality-Temporal Unified Queries (MTUQ)
Four-view query system: `q_rgb`, `q_ir`, `q_motion`, `q_fused`. Queries carry modality-specific and temporal information across frames with adaptive gated fusion.

### 3. Modality-Aware Decoder (MAD)
6-layer decoder with per-view cross-attention and learnable 3-view fusion gates. Enables query-level (not feature-level) cross-modal fusion.

### 4. Cross-Modal Consistency (CMC) Loss
Auxiliary per-modality prediction heads with:
- **Consistency loss**: L1 between RGB and IR box predictions
- **Contrastive loss**: InfoNCE pulling matched query pairs closer

### 5. Scale-Adaptive Sampling (SAS)
Learnable per-query scale parameters replacing fixed multi-scale sampling points. Regularized by scale diversity loss to prevent collapse.

### 6. Motion View Enhancement
Trajectory memory bank storing last K frames of track queries. 2-layer Transformer encodes temporal patterns with box velocity features, injected via gating.

## 📊 Implementation Stages

| Stage | Component | Params | Status |
|-------|-----------|--------|--------|
| 0 | Baseline (Deformable DETR + tracking) | 40.1M | ✅ |
| 1 | + NWD Loss | 40.1M | ✅ |
| 2 | + MTUQ + MAD Decoder | 79.2M | ✅ |
| 3 | + CMC Loss | 79.5M | ✅ |
| 4 | + SAS Attention | 80.5M | ✅ |
| 5 | + Motion View | 81.8M | ✅ |
| 6 | System Integration | 81.8M | ✅ |

## 📋 Training Configuration

Key hyperparameters (see `configs/dsitt_full.yaml`):

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| Encoder/Decoder layers | 6/6 |
| Feature levels | 4 |
| Queries | 300 |
| NWD constant | 0.1 |
| CMC temperature | 0.07 |
| Scale diversity weight | 0.1 |
| Motion memory length | 5 |
| Modality dropout | 0.1 |
| Base LR | 2e-4 |

### Progressive Clip Schedule
| Epoch | Clip Length |
|-------|------------|
| 1 | 2 |
| 50 | 3 |
| 90 | 4 |
| 150 | 5 |

## 📝 License

This project is for research purposes.