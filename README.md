# DSITT: Dual-Stream Infrared Tiny Target Tracker

Query-level cross-modal fusion via Modality-Temporal Unified Queries for RGBT tiny object tracking.

## 🚀 Quick Start: Environment Setup & Training

### 1. 系统要求

- **GPU**: NVIDIA GPU with ≥24GB VRAM (tested on 32GB)
- **OS**: Linux (tested on Ubuntu)
- **Python**: 3.10+
- **CUDA**: 11.8+ (with PyTorch 2.0+)

### 2. 克隆仓库

```bash
git clone https://github.com/Stardep-lmc/DSITT.git
cd DSITT
```

### 3. 安装依赖

```bash
# 创建 conda 环境 (推荐)
conda create -n dsitt python=3.12 -y
conda activate dsitt

# 安装 PyTorch (根据你的 CUDA 版本选择)
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install pyyaml tensorboard scipy pillow
```

### 4. 准备数据集

RGBT-Tiny 数据集包含三个压缩包：
- `images.zip` (2.3GB) — RGB+IR 图像
- `annotations_coco.zip` (37MB) — COCO 格式标注 
- `data_split.zip` (419KB) — 训练/测试划分

```bash
# 在项目根目录下创建数据目录
mkdir -p data/rgbt_tiny

# 将三个 zip 文件放入项目根目录, 然后解压:
unzip -q data_split.zip -d data/rgbt_tiny/
unzip -q images.zip -d data/rgbt_tiny/images/
unzip -q annotations_coco.zip -d data/rgbt_tiny/annotations/
```

解压后的目录结构应该是:
```
data/rgbt_tiny/
├── images/
│   ├── DJI_0022_1/
│   │   ├── 00/          ← RGB (640x512, 3通道)
│   │   │   ├── 00000.jpg
│   │   │   └── ...
│   │   └── 01/          ← IR  (640x512, 1通道灰度)
│   │       ├── 00000.jpg
│   │       └── ...
│   └── ... (共 115 个序列)
├── annotations/
│   ├── instances_00_train2017.json  (RGB train, COCO格式)
│   ├── instances_00_test2017.json
│   ├── instances_01_train2017.json  (IR train)
│   └── instances_01_test2017.json
├── 00_train.txt     (RGB 训练图像列表)
├── 00_test.txt
├── 01_train.txt     (IR 训练图像列表)
├── 01_test.txt
├── train.txt
└── test.txt
```

**验证数据集:**
```bash
python -c "
from datasets.rgbt_tiny import RGBTTinyDataset
ds = RGBTTinyDataset('data/rgbt_tiny', split='train', modality='both')
print(f'Sequences: {len(ds.sequences)}')  # 应该输出 85
frames, targets = ds[0]
print(f'Frame type: {type(frames[0])}, shape: {frames[0][0].shape}')  # (3, 512, 640)
print(f'Targets: {targets[0][\"labels\"].shape[0]} objects')
"
```

### 5. 验证模型构建

```bash
# Dummy 数据快速测试 (不需要真实数据集)
python tools/train.py --dummy --epochs 2 --print_freq 1 --config configs/dsitt_full.yaml
```

应该看到:
```
Using DSITTv2 (MTUQ + MAD + SAS + Motion)
Parameters: 81.5M (trainable: 81.4M)
...
Epoch [1] Complete. Avg Loss: X.XX
```

### 6. 开始训练

```bash
# 完整训练 (200 epochs, ~4小时 on 32GB GPU)
python tools/train.py \
    --config configs/dsitt_full.yaml \
    --data_root data/rgbt_tiny \
    --epochs 200 \
    --print_freq 40 \
    --save_freq 20 \
    --output_dir outputs/train_200ep \
    --num_workers 0

# 使用 AMP 加速 (可选, 节省约 30% 显存)
python tools/train.py \
    --config configs/dsitt_full.yaml \
    --data_root data/rgbt_tiny \
    --epochs 200 \
    --print_freq 40 \
    --save_freq 20 \
    --output_dir outputs/train_200ep \
    --num_workers 0 \
    --amp
```

**从 checkpoint 恢复训练:**
```bash
python tools/train.py \
    --config configs/dsitt_full.yaml \
    --data_root data/rgbt_tiny \
    --epochs 200 \
    --output_dir outputs/train_200ep \
    --resume outputs/train_200ep/checkpoints/checkpoint_0100.pth
```

### 7. 评估

```bash
python tools/eval.py \
    --config configs/dsitt_full.yaml \
    --checkpoint outputs/train_200ep/checkpoints/checkpoint_0200.pth \
    --data_root data/rgbt_tiny \
    --score_threshold 0.3
```

---

## 📂 项目结构

```
DSITT/
├── models/
│   ├── dsitt_v2.py               # 主模型 (双流+所有创新)
│   ├── dsitt.py                  # v1 基线 (单流)
│   ├── backbone/
│   │   ├── resnet.py             # ResNet-50
│   │   └── dual_stream.py        # 双流骨干 + 模态 Dropout
│   ├── encoder/
│   │   └── deformable_encoder.py # 多尺度可变形编码器
│   ├── decoder/
│   │   ├── modality_aware_decoder.py  # MAD: 4步查询级融合 + 辅助解码损失
│   │   └── scale_adaptive_attn.py     # SAS: 尺度自适应采样
│   ├── tracking/
│   │   ├── mtuq_manager.py       # MTUQ: 四视图查询管理
│   │   ├── track_manager.py      # TALA: 轨迹感知标签分配
│   │   └── motion_view.py        # 运动视图编码器 + 记忆库
│   ├── loss/
│   │   ├── losses.py             # 主损失 (Focal + L1 + NWD + 辅助)
│   │   ├── nwd_loss.py           # 归一化 Wasserstein 距离
│   │   └── cmc_loss.py           # 跨模态一致性损失
│   └── ops/
│       └── ms_deform_attn.py     # 多尺度可变形注意力
├── datasets/
│   └── rgbt_tiny.py              # RGBT-Tiny 数据集 (COCO格式)
├── configs/
│   ├── dsitt_full.yaml           # 完整 v2 配置
│   ├── dsitt_base.yaml           # v1 基线配置
│   ├── dsitt_nwd.yaml            # v1 + NWD
│   └── dsitt_mtuq.yaml           # v2 MTUQ 配置
├── tools/
│   ├── train.py                  # 训练脚本 (支持 v1/v2, AMP, resume)
│   └── eval.py                   # 评估脚本 (MOTA/IDF1/IDS)
├── paper/
│   ├── dsitt_paper.tex           # 论文 LaTeX
│   └── references.bib            # BibTeX 参考文献
└── analysis/                     # 各阶段实现笔记
```

## 🏗️ 架构

```
(img_rgb, img_ir) × T frames
      ↓
[DualStreamBackbone + ModalityDropout]
  → (F_rgb, F_ir) × 4 scales
      ↓
[DualStreamEncoder] × 6 layers
  → (M_rgb, M_ir)
      ↓
[MTUQ: {q_rgb, q_ir, q_mot, q_fused}]
      ↓
[MotionViewUpdater(memory_bank)]
      ↓
[MAD Decoder] × 6 layers (each with auxiliary loss)
  Step 1: Self-attention (q_fused)
  Step 2: SAS Cross-attention (q_rgb↔M_rgb, q_ir↔M_ir)
  Step 3: Cross-modal exchange (q_rgb↔q_ir)
  Step 4: Gated 3-view fusion → q_fused
      ↓
[Prediction Heads] → cls, box
[CMC Auxiliary Heads] → box_rgb, box_ir
      ↓
[Loss] = Focal + L1 + NWD + CMC + SAS_div + Aux(layers 1-5)
```

## ⚙️ 关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 256 | 特征维度 |
| num_queries | 100 | 检测查询数 |
| num_encoder/decoder_layers | 6/6 | 编解码器层数 |
| cls_weight | 5.0 | 分类损失权重 |
| nwd_constant | 0.1 | NWD 归一化常数 |
| modality_dropout | 0.1 | 模态随机丢弃率 |
| base_lr | 2e-4 | 基础学习率 |
| lr_drop_epoch | 100 | LR 下降 epoch |

## 📝 License

Research use only.