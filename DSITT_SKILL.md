# DSITT / QFTrack 项目全局知识库 (Skill Document)

> 本文档是项目的"大脑"——每次开新窗口时首先阅读此文件，快速恢复对整个代码库的理解。
> 持续维护：发现新问题、完成新功能后请更新对应章节。

---

## 1. 项目定位

DSITT（代码名）/ QFTrack（论文名）是一个面向 **RGBT 微小目标多目标跟踪 (MOT)** 的端到端框架。

核心创新点：**查询级跨模态融合**——不同于现有方法在特征层融合 RGB 和 IR，本方法在解码器内部的查询层面进行融合，每个跟踪查询维护独立的模态视图。

目标会议：CVPR 2026。数据集：RGBT-Tiny（115 序列，640×512，7 类，目标最小 4×4 像素）。

---

## 2. 架构总览

```
输入: (img_rgb, img_ir) × T 帧
       ↓
[DualStreamBackbone]  两个独立 ResNet-50 + FPN，模态 Dropout(p=0.1)
  → (F_rgb, F_ir) × 4 尺度 (stride 4/8/16/32)
       ↓
[DualStreamEncoder]  两个独立 6 层 Deformable Transformer Encoder
  → (M_rgb, M_ir)  编码后的多尺度记忆
       ↓
[MTUQ Manager]  构造四视图查询 {q_rgb, q_ir, q_motion, q_fused}
       ↓
[MotionViewUpdater]  轨迹记忆库 → 时序 Transformer → 门控注入 q_motion
       ↓
[MAD Decoder] × 6 层 (自注意力 → SAS交叉注意力 → 跨模态交换 → 门控融合)
  + 每层辅助预测 + 迭代参考点精炼
       ↓
[预测头] q_fused → cls_head + bbox_head
[损失] = Focal + L1 + NWD + CMC + SAS_div + Aux
```

---

## 3. 目录结构与文件职责

```
DSITT/
├── models/
│   ├── __init__.py              # 导出 build_dsitt + build_dsitt_v2
│   ├── dsitt.py                 # v1 基线模型 (单模态)
│   ├── dsitt_v2.py              # v2 完整模型 (双模态 MTUQ)
│   ├── backbone/
│   │   ├── resnet.py            # ResNet-50 + FPN + FrozenBatchNorm2d
│   │   └── dual_stream.py       # 双流骨干 + 模态 Dropout
│   ├── encoder/
│   │   └── deformable_encoder.py
│   ├── decoder/
│   │   ├── deformable_decoder.py      # v1 解码器 (含迭代精炼)
│   │   ├── modality_aware_decoder.py  # v2 MAD 解码器
│   │   └── scale_adaptive_attn.py     # SAS 注意力
│   ├── tracking/
│   │   ├── track_manager.py     # v1 QIM + TALA
│   │   ├── mtuq_manager.py      # v2 MTUQ-QIM + TALA
│   │   └── motion_view.py       # 运动视图 + 记忆库
│   ├── loss/
│   │   ├── losses.py            # Focal + L1 + NWD/GIoU + 辅助损失
│   │   ├── nwd_loss.py          # NWD 距离
│   │   └── cmc_loss.py          # 跨模态一致性损失
│   └── ops/
│       └── ms_deform_attn.py    # 纯 PyTorch 可变形注意力
├── datasets/
│   ├── rgbt_tiny.py             # 数据集加载
│   └── transforms.py            # 双模态增强
├── configs/                     # 4 个 YAML 配置
├── tools/
│   ├── train.py                 # 训练 (warmup + AMP + 梯度累积)
│   ├── eval.py                  # 评估 (HOTA/MOTA/IDF1 + 可视化)
│   ├── test_model.py            # v1 冒烟测试
│   └── test_model_v2.py         # v2 冒烟测试
└── paper/                       # 论文 LaTeX
```

---

## 4. 关键模块设计逻辑

### 4.1 双流骨干
- 两个独立 ResNet-50 + FPN，BN 全部替换为 FrozenBatchNorm2d
- 模态 Dropout(p=0.1): 训练时随机置零一个模态

### 4.2 可变形编码器
- 4 尺度 FPN 特征展平 + level embedding，6 层可变形自注意力
- RGB/IR 各自独立编码

### 4.3 MTUQ 查询管理
- 检测查询: 4 组 Embedding + 位置 Embedding
- 跟踪查询: 上一帧经 5 个 MLP 投影，q_motion 来自 prev q_fused
- QIM 筛选后重映射 track_assignment 为连续索引
- p_insert 假阳性注入

### 4.4 MAD 解码器
- 每层: 自注意力 → SAS 交叉注意力 → 跨模态交换 → 门控融合 → FFN
- 迭代参考点精炼 (detached)，每层辅助预测

### 4.5 SAS 注意力
- per-query scale_param ∈ (0,1)，约束采样偏移范围
- 多样性正则: ReLU(0.15 - std)

### 4.6 运动视图
- 记忆库存 K=5 帧 (q_fused, boxes)，计算速度特征
- 2 层 Temporal Transformer + 门控注入
- track 数量变化时跳过更新，push 前按需重置

### 4.7 损失体系

| 损失 | 权重 |
|------|------|
| Focal (α=0.25, γ=2.0) | 2.0 |
| L1 | 5.0 |
| NWD (C=0.1) | 2.0 |
| CMC 一致性 | 1.0 |
| CMC 对比 (τ=0.07) | 0.5 |
| SAS 多样性 | 0.1 |
| 辅助解码 (层1-5均值) | 1.0 |

### 4.8 数据集 (RGBTTinyDataset)
- COCO 格式标注，含 tracking_id
- 支持 ir/rgb/both 三种模态
- 每 epoch 采样 samples_per_epoch=2000 个 clip
- 坐标归一化 [0,1]，格式 (cx, cy, w, h)
- 双模态增强: 同步水平翻转 + RGB 亮度/对比度抖动

---

## 5. 训练流程

1. 加载 YAML 配置，根据 `model.version` 选择 v1 或 v2
2. 构建模型，backbone 用 0.1× 学习率
3. AdamW 优化器，SequentialLR: LinearLR warmup (1000 iter) + StepLR (epoch 100 ×0.1)
4. 每 epoch 根据 clip_schedule 调整 clip_length
5. 每个 batch: 前向 → 损失/accum_steps → 反向 → (累积完成后) 梯度裁剪(0.1) → 更新 → lr_scheduler.step()
6. 支持 AMP (fp16)、checkpoint resume、梯度累积 (--accum_steps)
7. 每 save_freq epoch 验证 + 保存 checkpoint，按 MOTA 追踪 best model

---

## 6. 评估流程

1. 加载 checkpoint，构建模型，eval 模式
2. 逐序列推理，每帧产生 (scores, labels, boxes)
3. 按 score_threshold 过滤
4. 匈牙利 IoU 匹配计算 TP/FP/FN/IDS
5. 输出 HOTA, MOTA, IDF1, DetA, AssA, Precision, Recall, FPS
6. 可选 --visualize 保存前 200 帧的 pred+GT 框叠加图

---

## 7. 配置系统

4 个 YAML 配置文件，递进关系:
- `dsitt_base.yaml`: v1 基线，单模态 IR，GIoU 损失
- `dsitt_nwd.yaml`: v1 + NWD 损失
- `dsitt_mtuq.yaml`: v2 双模态，MTUQ，NWD
- `dsitt_full.yaml`: v2 完整版，所有创新，num_queries=300，cls_weight=2.0

---

## 8. 项目完成计划

### 8.0 已完成项 (P1-P15 + 8.1 + 8.2)

所有代码级 bug 和功能缺失已修复，详见第 13 节归档和第 14 节更新日志。

当前代码状态：
- ✅ v1/v2 模型完整，含辅助解码损失
- ✅ 训练脚本：warmup + AMP + 梯度累积 + best model 保存 + resume 鲁棒性
- ✅ 评估脚本：HOTA/MOTA/IDF1 + 可视化
- ✅ CMC 损失：全双向 KL + InfoNCE 对比
- ✅ 所有 15 个初始 bug + 6 个新发现 bug + 4 个功能缺失已修复

### 8.1 遗留注意事项（无需修复，需知晓）

| # | 类型 | 描述 |
|---|------|------|
| 1 | 注意事项 | `losses.py` 辅助损失循环排除最后一层（= 主预测），v1/v2 语义一致 |
| 2 | 已知近似 | `eval.py` HOTA 的 AssA 用 IoU 近似，与 TrackEval 官方不完全一致 |
| 3 | 配置脆弱 | `build_dsitt` / `build_dsitt_v2` 未从 config 读取 `l1_weight`、`giou_weight`、`focal_gamma`，使用默认值（恰好与 config 一致）。如需调参需修改 builder |
| 4 | 架构限制 | TALA 硬编码 `assert B == 1`，仅支持 batch_size=1 |
| 5 | 架构限制 | 双流骨干参数量 81.8M，显存开销大 |
| 6 | 架构限制 | clip_length > 2 导致 OOM |
| 7 | 架构限制 | 运动视图对新目标无效（检测查询的 q_motion 是固定 Embedding） |
| 8 | 训练风险 | CMC 对比损失在匹配目标少时退化（已有 M<2 保护） |
| 9 | 训练风险 | 门控权重可能坍缩到单一模态（需监控 gate_rgb/gate_ir/gate_motion） |

### 8.2 接下来要做的事（按优先级排序）

| # | 优先级 | 任务 | 依赖 | 预计工作量 | 状态 |
|---|--------|------|------|-----------|------|
| 1 | 🔴P0 | **数据准备**：下载 RGBT-Tiny 数据集，验证 COCO 格式标注，确认 train/test 划分 | 无 | 0.5 天 | ✅ 已完成 |
| 2 | 🔴P0 | **端到端冒烟测试**：用真实数据跑 5 epoch，确认训练循环无报错 | #1 | 0.5 天 | ✅ 已完成 |
| 3 | 🔴P0 | **v1 基线训练**：dsitt_base.yaml，200 epoch，记录 HOTA/MOTA/IDF1 | #2 | 3 天 | 🔄 训练中 (fp32, epoch 11+/200) |
| 4 | 🔴P0 | **v1+NWD 训练**：dsitt_nwd.yaml，200 epoch，对比 GIoU vs NWD | #3 | 3 天 | 待开始（v1 完成后启动） |
| 5 | 🔴P0 | **v2 完整模型训练**：dsitt_full.yaml，200 epoch | #2 | 3 天 | ⏸ 暂停 (epoch 25, 有 checkpoint) |
| 6 | 🟡P1 | **消融实验**：逐一关闭 NWD/CMC/SAS/Motion，记录指标变化 | #5 | 3 天 | 待开始 |
| 7 | 🟡P1 | **填充论文实验数据**：将训练结果填入 paper/dsitt_paper.tex 的 TODO 表格 | #3-#6 | 1 天 | 待开始 |
| 8 | 🟡P1 | **生成论文图表**：5 张 FIGNEEDED 图（架构图、可视化对比、消融曲线等） | #5-#6 | 2 天 | 待开始 |
| 9 | 🟢P2 | **超参调优**：NWD C ∈ {0.05, 0.1, 0.2}，辅助损失权重 ∈ {0.5, 1.0} | #5 | 2 天 | 待开始 |
| 10 | 🟢P2 | **builder 读取完整 config**：让 build_dsitt/v2 从 YAML 读取所有损失权重 | 无 | 0.5 天 | 待开始 |
| 11 | 🟢P2 | **补充材料**：更多可视化、per-class 分析、失败案例分析 | #5 | 1 天 | 待开始 |
| 12 | 🟢P3 | **EMA 模型平均**：添加 EMA 权重平均提升稳定性 | 无 | 0.5 天 | 待开始 |
| 13 | 🟢P3 | **多 GPU 支持**：DDP 训练 | 无 | 1 天 | 待开始 |

### 8.3 训练实验计划表

| 实验 | 配置文件 | 模型 | 关键变量 | 论文表格 |
|------|----------|------|----------|----------|
| Exp-A | dsitt_base.yaml | v1 | GIoU, 单模态 IR | Table 1 (baseline) |
| Exp-B | dsitt_nwd.yaml | v1 | NWD, 单模态 IR | Table 1 (+NWD) |
| Exp-C | dsitt_mtuq.yaml | v2 | NWD, 双模态, MTUQ+MAD | Table 1 (+MTUQ) |
| Exp-D | dsitt_full.yaml | v2 | 全部创新 | Table 1 (Full) |
| Abl-1 | dsitt_full.yaml (no CMC) | v2 | use_cmc=False | Table 2 |
| Abl-2 | dsitt_full.yaml (no SAS) | v2 | 替换 SAS 为标准 deform attn | Table 2 |
| Abl-3 | dsitt_full.yaml (no Motion) | v2 | 禁用 motion_updater | Table 2 |
| Abl-4 | dsitt_nwd.yaml (GIoU) | v1 | box_loss_type=giou | Table 2 |

### 8.4 训练经验教训（重要！）

| # | 经验 | 详情 |
|---|------|------|
| 1 | **v1 + GIoU + AMP = NaN** | v1 基线用 GIoU 损失 + AMP fp16 在 epoch 40+ 出现严重 NaN（>50% batch）。原因是可变形注意力在 fp16 下数值不稳定。降低 LR 无效。**解决方案：关闭 AMP 用 fp32 训练** |
| 2 | **v2 + NWD + AMP = 稳定** | v2 完整模型用 NWD 损失 + AMP fp16 训练 25 epoch 零 NaN。NWD 比 GIoU 数值更稳定（基于高斯分布距离，无除法溢出风险） |
| 3 | **fp32 不比 AMP 慢** | v1 fp32 训练 ~17 min/epoch，AMP 训练 ~22 min/epoch。纯 PyTorch 可变形注意力的 AMP 开销（NaN 检测 + scaler）反而拖慢速度 |
| 4 | **NaN 保护机制** | train.py 已添加：NaN loss 跳过 + track_manager/mtuq_manager 状态重置 + try/except 异常捕获。TALA 匈牙利匹配前 nan_to_num 清理代价矩阵 |
| 5 | **CUDA 编号 ≠ nvidia-smi 编号** | 本机 CUDA:0=4080S(32GB), CUDA:1=4080S(32GB), CUDA:2=4060Ti(16GB), CUDA:3=5060Ti(16GB)。用 `--device cuda:X` 指定 |
| 6 | **Checkpoint 加载慢** | 从磁盘加载 465MB-939MB checkpoint 需要 3-7 分钟（D state），属正常 |
| 7 | **建议训练配置** | v1: fp32 无 AMP；v2: AMP + NWD；所有实验: clip_length=2, batch_size=1, num_workers=2 |

### 8.5 当前训练状态（2026-04-09 更新）

**GPU 映射（nvidia-smi 实际编号，2026-04-09 确认）：**
| GPU | 型号 | 显存 | 状态 |
|-----|------|------|------|
| cuda:0 | RTX 5060 Ti | 16GB | 空闲 |
| cuda:1 | RTX 4060 Ti | 16GB | 其他项目占用 6.5GB |
| cuda:2 | RTX 4080 SUPER | 32GB | v1 训练中 |
| cuda:3 | RTX 4080 SUPER | 32GB | 空闲 |

**正在运行：**
- v1 base (Exp-A): `outputs_base/`, fp32 无 AMP, 从 epoch 10 resume, 当前 epoch 12+/200, loss ~5.0, 零 NaN, cuda:2 (4080S 32GB), ~17 min/epoch
  - 预计剩余 ~53 小时（约 2.2 天）

**已死亡（需恢复）：**
- v2 full (Exp-D): `outputs_full/`, AMP, 进程在 epoch 30 iter 1 后 OOM 死亡。epoch 21→29 loss 1.85→1.74 收敛良好。最新 checkpoint: epoch 20
  - Resume 命令: `cd DSITT && nohup python -u tools/train.py --config configs/dsitt_full.yaml --data_root data/rgbt_tiny --epochs 200 --amp --save_freq 10 --print_freq 50 --num_workers 2 --output_dir outputs_full --device cuda:3 --resume outputs_full/checkpoints/checkpoint_0020.pth > outputs_full_train.log 2>&1 &`
  - 决策：等 v1 完成后再恢复

**待启动（按顺序）：**
1. v1+NWD (Exp-B): v1 完成后在同一 GPU 启动, `outputs_nwd/`, fp32
2. v2 full (Exp-D): 恢复训练（从 epoch 20 checkpoint）
3. 消融实验 (Abl-1~4): 所有主实验完成后

---

## 9. 数据流详解 (v2 训练模式)

```python
# 1. 数据加载 + 增强
frames, targets = dataset[idx]
# frames = [(rgb_tensor, ir_tensor), ...] × clip_length
# 增强: 同步水平翻转 + RGB 颜色抖动

# 2. 双流骨干 (含模态 Dropout)
srcs_rgb, pos_rgb, srcs_ir, pos_ir = dual_backbone(img_rgb, img_ir)

# 3. 双流编码
memory_rgb = encoder_rgb(srcs_rgb, pos_rgb)
memory_ir = encoder_ir(srcs_ir, pos_ir)

# 4. MTUQ 查询 (track + detect)
queries = {q_rgb, q_ir, q_motion, q_fused}

# 5. 运动视图更新 (仅 track queries)
q_motion[:, :n_track] = motion_updater(...)

# 6. MAD 解码 (6 层，含迭代参考点精炼)
for layer in decoder.layers:
    queries, gate_weights, scale_params = layer(...)
    reference_points = refine(reference_points, bbox_head)

# 7. TALA 标签分配 + QIM 生成下一帧 track queries
# 8. 损失 = 主损失 + CMC + SAS_div + 辅助解码
```

---

## 10. 论文状态

论文文件: `paper/dsitt_paper.tex`，CVPR 2026 格式。

已完成: Abstract, Introduction, Related Work, Method (全部), Experiments 框架, Conclusion

待完成:
- 所有实验数据 (`\TODO{}`) → 依赖 8.2#3-#6
- 5 张图 (`\FIGNEEDED{}`) → 依赖 8.2#8
- 补充材料 → 依赖 8.2#11

---

## 11. 快速命令参考

```bash
cd DSITT

# Dummy 冒烟测试
python tools/train.py --dummy --epochs 2 --print_freq 1 --config configs/dsitt_full.yaml

# 真实数据训练 (v2 完整模型)
python tools/train.py --config configs/dsitt_full.yaml --data_root data/rgbt_tiny --epochs 200 --amp --accum_steps 4

# 真实数据训练 (v1 基线)
python tools/train.py --config configs/dsitt_base.yaml --data_root data/rgbt_tiny --epochs 200

# 评估
python tools/eval.py --config configs/dsitt_full.yaml --checkpoint outputs/checkpoints/checkpoint_best.pth --data_root data/rgbt_tiny

# 评估 + 可视化
python tools/eval.py --config configs/dsitt_full.yaml --checkpoint outputs/checkpoints/checkpoint_best.pth --data_root data/rgbt_tiny --visualize
```

---

## 12. 关键超参速查

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 256 | 特征维度 |
| nhead | 8 | 注意力头数 |
| num_encoder/decoder_layers | 6 | 层数 |
| dim_feedforward | 1024 | FFN 中间维度 |
| num_queries | 300 | 检测查询数 |
| num_classes | 7 | 类别数 |
| modality_dropout | 0.1 | 模态丢弃率 |
| base_lr | 2e-4 | 基础学习率 |
| backbone_lr_factor | 0.1 | 骨干 LR 倍率 |
| warmup_iters | 1000 | warmup 步数 |
| lr_drop_epoch | 100 | LR 衰减 epoch |
| clip_max_norm | 0.1 | 梯度裁剪 |
| nwd_constant | 0.1 | NWD 常数 C |
| focal_alpha/gamma | 0.25/2.0 | Focal loss |
| cls_weight | 2.0 | 分类损失权重 |
| box_l1_weight | 5.0 | L1 框损失权重 |
| giou_weight | 2.0 | NWD/GIoU 权重 |
| cmc_consistency/contrastive | 1.0/0.5 | CMC 权重 |
| cmc_temperature | 0.07 | 对比学习温度 |
| scale_div_weight | 0.1 | SAS 多样性权重 |
| memory_len | 5 | 轨迹记忆长度 |
| p_drop/p_insert | 0.1/0.1 | QIM 丢弃/插入率 |
| max_offset | 0.5 | SAS 最大偏移 |
| target_std | 0.15 | SAS 目标标准差 |

---

## 13. 已修复问题归档

### 13.1 初始 15 个问题 (P1-P15)

| 序号 | 问题简述 | 修复摘要 |
|------|----------|----------|
| P1 | BN 冻结失效 | resnet.py 用 FrozenBatchNorm2d 替换所有 BN |
| P2 | CMC Loss box 未加 reference_points | cmc_loss.py 从 frame_output 读取 ref_points 计算正确坐标 |
| P3 | v1 decoder 无 bias prior | deformable_decoder.py 添加 class_head bias 初始化 |
| P4 | __init__.py 未导出 v2 | 添加 `from .dsitt_v2 import build_dsitt_v2` |
| P5 | TALA prev_q_idx 映射错位 | QIM 筛选后 old_to_new 重映射 track_assignment |
| P6 | 运动记忆库 track 变化时重置 | 跳过运动更新而非重置，push 前按需重置 |
| P7 | scale_diversity_loss 与论文不一致 | 保持代码 hinge loss，修改论文公式匹配 |
| P8 | dsitt_full.yaml 超参不一致 | 统一为论文值 (cls=2.0, alpha=0.25, queries=300) |
| P9 | 无数据增强 | 新建 transforms.py，同步翻转 + RGB 颜色抖动 |
| P10 | 评估指标不完整 | eval.py 添加 HOTA/DetA/AssA，匈牙利匹配 |
| P11 | 无学习率 warmup | train.py 用 SequentialLR(LinearLR + StepLR) |
| P12 | 无 v2 冒烟测试 | 新建 test_model_v2.py |
| P13 | FrozenBatchNorm2d 未使用 | resnet.py 递归替换所有 BN 为 FrozenBN |
| P14 | p_insert 未实现 | v1/v2 QIM 均实现假阳性注入 |
| P15 | 参考点无迭代精炼 | 两个解码器均添加逐层精炼 (detached) |

### 13.2 第二轮修复 (8.1 代码 Bug + 8.2 功能缺失)

| 序号 | 问题简述 | 修复摘要 |
|------|----------|----------|
| 8.1#1 | v1 不返回辅助输出 | dsitt.py forward 传递 aux_cls/aux_coord |
| 8.1#2 | v1 解码器不收集中间层预测 | deformable_decoder.py 每层收集 cls/coord |
| 8.1#5 | LR scheduler resume 偏移 | checkpoint 保存 iters_per_epoch，resume 时检测变化重建 |
| 8.1#6 | CMC KL 双侧 detach | 移除 detach，全双向梯度 |
| 8.2#2 | 无 best model 保存 | train.py 添加验证 + MOTA 追踪 + checkpoint_best.pth |
| 8.2#3 | 无梯度累积 | train.py --accum_steps，loss/N，scheduler 仅在 step 时推进 |
| 8.2#4 | 无可视化工具 | eval.py visualize_frame()，--visualize 保存 pred+GT 框图 |

---

## 14. 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-04-07 | 初始版本：完整代码分析，架构梳理，缺陷清单 |
| 2026-04-07 | P1-P15 全部修复完成 |
| 2026-04-07 | SKILL 文档重构：归档已修复缺陷，新增 6 个新发现缺陷 |
| 2026-04-07 | 修复 8.1#1+#2：v1 辅助解码损失生效 |
| 2026-04-07 | 实现 8.2#2：best model 保存 |
| 2026-04-07 | 实现 8.2#3：梯度累积 |
| 2026-04-07 | 实现 8.2#4：可视化工具 |
| 2026-04-07 | 修复 8.1#5：LR scheduler resume 鲁棒性 |
| 2026-04-07 | 修复 8.1#6：CMC KL 全双向梯度 |
| 2026-04-07 | SKILL 全面重构：第 8 章改为项目完成计划表，新增遗留注意事项、实验计划、下一步任务清单 |
| 2026-04-08 | 8.2#1 数据准备完成：解压 RGBT-Tiny 至 data/rgbt_tiny/，验证 115 序列 (85 train + 30 test)，46701 帧 RGB+IR，431K 标注含 tracking_id，7 类，640×512 |
| 2026-04-08 | 8.2#2 端到端冒烟测试完成：conda env dsitt (Python 3.10, PyTorch 2.11+cu130, RTX 4080 SUPER)，v2 full 真实数据 2 epoch 无报错，损失正常收敛 11→5，~1.1s/iter。修复 GradScaler/autocast FutureWarning |
| 2026-04-08 | 8.2#3 v1 基线训练启动：dsitt_base.yaml, 单模态 IR, 200 epoch, AMP, cuda:0 (4080S 32GB)。修复所有配置 clip_schedule 为固定 clip_length=2 防 OOM。添加 stdout flush 修复 nohup 日志缓冲。输出目录 outputs_base/ |
| 2026-04-08 | 8.2#5 v2 完整模型训练启动：dsitt_full.yaml, 双模态 both, 200 epoch, AMP, cuda:1 (4080S 32GB, 24.1GB 占用)。与 v1 并行训练。输出目录 outputs_full/ |
| 2026-04-09 | v1 AMP NaN 问题：epoch 40+ GIoU+AMP 出现严重 NaN (>50% batch)。尝试降低 LR 无效。根因：可变形注意力 fp16 数值不稳定 |
| 2026-04-09 | 修复 NaN 保护：TALA 代价矩阵 nan_to_num + 训练循环 NaN loss 跳过 + track state 重置 + try/except 异常捕获 |
| 2026-04-09 | v1 base 从 epoch 10 fp32 重启，零 NaN，~17 min/epoch。v2 full 暂停 (epoch 25 checkpoint)，改为单 GPU 串行训练 |
| 2026-04-09 | SKILL 新增 8.4 训练经验教训 + 8.5 当前训练状态。记录 CUDA 编号映射、AMP 兼容性、建议训练配置 |

---

> ⚠️ 维护提醒：每次修复 bug、添加功能、完成实验后，请更新本文档对应章节。
> 特别是第 8 节（计划表）和第 14 节（更新日志）。
> 新窗口开始工作前，先阅读本文件恢复上下文。
