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
  - 检测查询: 4 组独立 Embedding (num_queries 个)
  - 跟踪查询: 上一帧输出经 per-view MLP 投影
       ↓
[MotionViewUpdater]  轨迹记忆库 → 时序 Transformer → 门控注入 q_motion
       ↓
[MAD Decoder] × 6 层，每层 4 步:
  Step 1: q_fused 自注意力 (查询间交互)
  Step 2: SAS 可变形交叉注意力 (q_rgb↔M_rgb, q_ir↔M_ir)
  Step 3: 双向跨模态交换 (q_rgb↔q_ir)
  Step 4: 门控三视图融合 → q_fused
  + 每层产生辅助预测 (auxiliary decoding loss)
       ↓
[预测头] q_fused → cls_head(Linear) + bbox_head(3层MLP)
[CMC 辅助头] q_rgb/q_ir → 共享 cls_head/bbox_head
       ↓
[损失] = Focal + L1 + NWD + CMC(一致性+对比) + SAS_div + Aux(层1-5)
```

---

## 3. 目录结构与文件职责

```
DSITT/
├── models/
│   ├── __init__.py              # 导出 build_dsitt (v1) + build_dsitt_v2 (v2)
│   ├── dsitt.py                 # v1 基线模型 (单模态，单向量查询)
│   ├── dsitt_v2.py              # v2 完整模型 (双模态，MTUQ 四视图查询)
│   ├── backbone/
│   │   ├── resnet.py            # ResNet-50 + FPN + 正弦位置编码
│   │   └── dual_stream.py       # 双流骨干，两个独立 Backbone 实例 + 模态 Dropout
│   ├── encoder/
│   │   └── deformable_encoder.py # 多尺度可变形自注意力编码器
│   ├── decoder/
│   │   ├── deformable_decoder.py      # v1 标准可变形解码器
│   │   ├── modality_aware_decoder.py  # v2 MAD 解码器 (核心创新)
│   │   └── scale_adaptive_attn.py     # SAS 尺度自适应可变形注意力
│   ├── tracking/
│   │   ├── track_manager.py     # v1 的 QIM + TALA
│   │   ├── mtuq_manager.py      # v2 的 MTUQ-QIM + TALA (四视图查询管理)
│   │   └── motion_view.py       # 运动视图更新器 + 轨迹记忆库
│   ├── loss/
│   │   ├── losses.py            # 主损失: Focal + L1 + NWD/GIoU + 辅助解码损失
│   │   ├── nwd_loss.py          # NWD 距离计算 (高斯建模 bbox)
│   │   └── cmc_loss.py          # 跨模态一致性损失 (预测一致性 + 对比学习)
│   └── ops/
│       └── ms_deform_attn.py    # 纯 PyTorch 多尺度可变形注意力 (无 CUDA 算子)
├── datasets/
│   └── rgbt_tiny.py             # RGBT-Tiny 数据集加载 (COCO 标注, 支持 dummy)
├── configs/
│   ├── dsitt_full.yaml          # v2 完整配置 (num_queries=300, cls_weight=2.0)
│   ├── dsitt_base.yaml          # v1 基线 (num_queries=300, cls_weight=2.0)
│   ├── dsitt_nwd.yaml           # v1 + NWD
│   └── dsitt_mtuq.yaml          # v2 MTUQ (num_queries=300)
├── tools/
│   ├── train.py                 # 训练脚本 (支持 v1/v2, AMP, resume, clip schedule)
│   ├── eval.py                  # 评估脚本 (简化 MOTA/IDF1/IDS)
│   └── test_model.py            # v1 冒烟测试
├── paper/
│   ├── dsitt_paper.tex          # 论文 LaTeX (CVPR 格式, 大量 TODO 待填)
│   └── references.bib           # 参考文献
└── analysis/                    # 各阶段设计笔记和路线图
```

---

## 4. 关键模块设计逻辑

### 4.1 双流骨干 (DualStreamBackbone)

- 两个完全独立的 ResNet-50 + FPN，不共享权重
- 模态 Dropout: 训练时以 p/2 概率将 RGB 置零，p/2 概率将 IR 置零
- 目的: 迫使模型学会在单模态退化时仍能工作
- IR 图像是灰度的，通过复制 3 通道来兼容 ResNet 输入

### 4.2 可变形编码器 (DeformableTransformerEncoder)

- 将 4 个尺度的 FPN 特征展平拼接，加上可学习的 level embedding
- 生成归一化参考点 (0~1)，每个空间位置在所有尺度上都有参考点
- 6 层可变形自注意力，每层: MSDeformAttn → LayerNorm → FFN → LayerNorm
- RGB 和 IR 各自独立编码，不共享参数

### 4.3 MTUQ 查询管理 (MTUQManager)

- 检测查询: 4 组独立 `nn.Embedding`，加 1 组位置 Embedding
- 跟踪查询: 上一帧匹配到 GT 的查询，经 5 个独立 MLP 投影 (rgb/ir/motion/fused/pos)
- `q_motion` 的初始化: 来自上一帧的 `q_fused`（而非 `q_motion`），这是有意设计
- 查询拼接顺序: [track_queries | detect_queries]，track 在前
- TALA 标签分配: 跟踪查询按 track_id 一致性分配，检测查询用匈牙利匹配

### 4.4 MAD 解码器 (ModalityAwareDecoder)

每层 4 步:
1. **自注意力**: 仅在 q_fused 上做标准 MHA，实现查询间空间排斥
2. **模态交叉注意力**: q_rgb 用 SAS 注意 M_rgb，q_ir 用 SAS 注意 M_ir
3. **跨模态交换**: q_rgb 以 q_ir 为 KV 做 MHA，q_ir 以 q_rgb 为 KV 做 MHA
4. **门控融合**: concat(q_rgb, q_ir, q_motion) → MLP → softmax → 加权求和 → 投影 → 残差到 q_fused → FFN

参考点: 从 query_pos 经线性层 + sigmoid 生成，在所有层间共享（无迭代精炼）。

辅助解码损失: 每层都产生 cls/box 预测，前 5 层的损失取平均加到总损失。

### 4.5 SAS 尺度自适应注意力 (ScaleAdaptiveDeformableAttn)

- 每个查询预测一个 scale_param ∈ (0,1)
- 采样偏移: `tanh(raw_offset) * scale_param * max_offset`
- 小 scale_param → 小采样范围 → 适合微小目标
- 多样性正则: `L_div = ReLU(target_std - std(scale_params))`，鼓励查询间尺度多样性

### 4.6 运动视图 (MotionViewUpdater + TrajectoryMemoryBank)

- 记忆库存储最近 K=5 帧的 (q_fused, pred_boxes)
- 计算帧间速度: Δbox = box[t] - box[t-1]
- 位置编码: MLP(concat(box, velocity)) → d_model
- 时序编码: 2 层 Transformer Encoder，取最后时刻输出
- 门控注入: gate = sigmoid(W·concat(q_motion, motion_token))，q_motion += gate * proj(motion_token)

### 4.7 损失函数体系

| 损失 | 公式 | 权重 | 作用 |
|------|------|------|------|
| Focal Loss | sigmoid focal, α=0.25, γ=2.0 | λ_cls=2.0 | 分类 |
| L1 Loss | \|pred - gt\|₁ | λ_l1=5.0 | 框回归 |
| NWD Loss | 1 - exp(-W₂/C), C=0.1 | λ_nwd=2.0 | 框回归(微小目标友好) |
| CMC 一致性 | L1(box_rgb, box_ir) + sym_KL(cls_rgb, cls_ir) | λ_con=1.0 | 跨模态预测一致 |
| CMC 对比 | InfoNCE(q_rgb, q_ir) | λ_ctr=0.5 | 跨模态特征对齐 |
| SAS 多样性 | ReLU(0.15 - std(scale_params)) | λ_div=0.1 | 防止尺度坍缩 |
| 辅助解码 | 中间层 cls+l1+nwd 的平均 | 1.0 | 深层监督 |

所有损失在 clip 内所有帧上累积，除以总匹配目标数 (Collective Average Loss)。

### 4.8 数据集 (RGBTTinyDataset)

- COCO 格式标注，含 tracking_id
- 支持 ir/rgb/both 三种模态模式
- 每 epoch 从全部可用 clip 中采样 samples_per_epoch=2000 个
- clip_length 可动态调整 (训练 schedule)
- 坐标归一化到 [0,1]，格式 (cx, cy, w, h)
- IR 灰度图复制为 3 通道，ImageNet 标准化

---

## 5. 训练流程

1. 加载 YAML 配置，根据 `model.version` 选择 v1 或 v2
2. 构建模型，backbone 用 0.1× 学习率
3. AdamW 优化器，StepLR 在 lr_drop_epoch 处 ×0.1
4. 每 epoch 根据 clip_schedule 调整 clip_length
5. 每个 batch: 移动到 GPU → 前向 → 损失 → 反向 → 梯度裁剪(0.1) → 更新
6. 支持 AMP (fp16)、checkpoint resume
7. TensorBoard 日志记录

---

## 6. 评估流程

1. 加载 checkpoint，构建模型，设为 eval 模式
2. 逐序列推理，每帧产生 (scores, labels, boxes)
3. 按 score_threshold 过滤 (建议 0.1)
4. 贪心 IoU 匹配计算 TP/FP/FN/IDS
5. 输出 MOTA, IDF1, Precision, Recall, FPS

---

## 7. 配置系统

4 个 YAML 配置文件，递进关系:
- `dsitt_base.yaml`: v1 基线，单模态 IR，GIoU 损失，num_queries=300
- `dsitt_nwd.yaml`: v1 + NWD 损失替换 GIoU
- `dsitt_mtuq.yaml`: v2 双模态，MTUQ，NWD，num_queries=300
- `dsitt_full.yaml`: v2 完整版，所有创新，num_queries=300，cls_weight=2.0

---

## 8. 已知缺陷与待修复问题

### 8.0 修复优先级总表（按顺序逐个修复）

| 序号 | 来源 | 问题简述 | 状态 |
|------|------|----------|------|
| P1 | 8.1#2 | BN 冻结失效：model.train() 会覆盖 BN 的 eval 状态 | ✅ 已修复 (2026-04-07) |
| P2 | 8.1#3 | CMC Loss 的 box 预测未加 reference_points，语义错误 | ✅ 已修复 (2026-04-07，重新应用) |
| P3 | 8.1#7 | v1 解码器 class_head 无 bias prior，focal loss 初期不稳定 | ✅ 已修复 (2026-04-07，重新应用) |
| P4 | 8.1#1 | models/__init__.py 未导出 build_dsitt_v2 | ✅ 已修复 (2026-04-07) |
| P5 | 8.1#5 | TALA prev_q_idx 映射在 QIM 筛选后可能错位 | ✅ 已修复 (2026-04-07) |
| P6 | 8.1#4 | 运动视图记忆库在 track 数量变化时被重置 | ✅ 已修复 (2026-04-07) |
| P7 | 8.1#8 | scale_diversity_loss 实现与论文公式不一致 | ✅ 已修复 (2026-04-07) |
| P8 | 8.1#6 | dsitt_full.yaml 超参与论文声称不一致 | ✅ 已修复 (2026-04-07) |
| P9 | 8.2#1 | 无数据增强 | ✅ 已修复 (2026-04-07) |
| P10 | 8.2#2 | 评估指标不完整 (HOTA 未实现, IDF1 近似) | ✅ 已修复 (2026-04-07) |
| P11 | 8.2#4 | 无学习率 warmup | ⬜ 待修复 |
| P12 | 8.2#7 | test_model.py 仅测试 v1，无 v2 冒烟测试 | ⬜ 待修复 |
| P13 | 8.2#9 | FrozenBatchNorm2d 定义但未使用 | ⬜ 待修复 |
| P14 | 8.2#10 | p_insert 参数接受但未实现 | ⬜ 待修复 |
| P15 | 8.3#1 | 参考点无迭代精炼 | ⬜ 待修复 |

> 修复规则：每次开新窗口修一个 P，修完后将状态改为 ✅ 已修复 并记录日期。

---

### 8.0.1 各 P 项详细修复指南

#### P1 — BN 冻结失效 ✅ 已修复

- 文件: `DSITT/models/backbone/resnet.py`
- 原因: `Backbone.__init__` 中对 BN 调用 `module.eval()`，但 PyTorch 的 `model.train()` 会递归将所有子模块切回 train 模式
- 修复: 添加 `self.freeze_bn = freeze_bn` 属性，重写 `train()` 方法:
```python
def train(self, mode: bool = True):
    super().train(mode)
    if mode and self.freeze_bn:
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    return self
```

---

#### P2 — CMC Loss box 预测语义错误 ✅ 已修复

- 文件: `DSITT/models/loss/cmc_loss.py`
- 涉及行: 第 151-226 行 `CMCLoss.forward()` 方法
- 原错误代码 (第 196-197 行):
```python
q_rgb_boxes = bbox_head(q_rgb)     # [B, N_q, 4]  ← 这是原始偏移量，不是坐标！
q_ir_boxes = bbox_head(q_ir)
```
- 正确逻辑参考 `modality_aware_decoder.py` 第 312-318 行:
```python
# 解码器中的正确做法:
reference_points = self.reference_point_head(query_pos).sigmoid()
bbox_off = self.bbox_head(q_fused)
coord = torch.cat([
    (reference_points + bbox_off[..., :2]).sigmoid(),
    bbox_off[..., 2:].sigmoid()
], dim=-1)
```
- 修复步骤:
  1. `CMCLoss.forward()` 签名添加 `reference_points` 参数
  2. 将 box 计算改为:
  ```python
  q_rgb_offset = bbox_head(q_rgb)
  q_rgb_boxes = torch.cat([
      (reference_points + q_rgb_offset[..., :2]).sigmoid(),
      q_rgb_offset[..., 2:].sigmoid()
  ], dim=-1)
  # 同理 q_ir
  q_ir_offset = bbox_head(q_ir)
  q_ir_boxes = torch.cat([
      (reference_points + q_ir_offset[..., :2]).sigmoid(),
      q_ir_offset[..., 2:].sigmoid()
  ], dim=-1)
  ```
  3. `dsitt_v2.py` 第 246-252 行调用处需要传入 reference_points:
  ```python
  # 当前代码:
  cmc_dict = self.cmc_criterion(
      frame_outputs, valid_assignments,
      class_head=self.decoder.class_head,
      bbox_head=self.decoder.bbox_head,
  )
  # 改为:
  cmc_dict = self.cmc_criterion(
      frame_outputs, valid_assignments,
      class_head=self.decoder.class_head,
      bbox_head=self.decoder.bbox_head,
      ref_point_head=self.decoder.reference_point_head,
  )
  ```
  4. `CMCLoss.forward()` 内部需要从 `frame_outputs` 中的 `queries` 获取 `query_pos`，然后用 `ref_point_head` 计算 reference_points。或者更简单的方案：让 `dsitt_v2.py` 在 `frame_output` dict 中存储 `reference_points`（MAD decoder 已经返回了），然后 CMCLoss 直接从 `frame_output` 中取用
  5. 最简方案：在 `dsitt_v2.py` 的 `frame_output` dict 中添加 `'reference_points': ref_points`，然后 CMCLoss 从中读取
- 修复实施 (2026-04-07):
  - 采用最简方案 (方案 5)
  - `dsitt_v2.py`: `forward_single_frame()` 返回 `ref_points`；`frame_output` dict 添加 `'reference_points': ref_points`
  - `cmc_loss.py`: `CMCLoss.forward()` 从 `output['reference_points']` 读取参考点，box 计算改为 `(ref + offset[:2]).sigmoid()` + `offset[2:].sigmoid()`，与解码器逻辑一致

---

#### P3 — v1 解码器 class_head 无 bias prior ✅ 已修复

- 文件: `DSITT/models/decoder/deformable_decoder.py`
- 涉及行: 第 118-121 行 `_reset_parameters` 方法
- 当前代码:
```python
def _reset_parameters(self):
    for p in self.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)
```
- 对比 MAD 解码器 (`modality_aware_decoder.py` 第 260-269 行) 已有正确实现:
```python
def _reset_parameters(self):
    for p in self.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)
    prior_prob = 0.01
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    constant_(self.class_head.bias, bias_value)
```
- 修复: 在 `deformable_decoder.py` 的 `_reset_parameters` 末尾添加:
```python
import math
from torch.nn.init import constant_
# ... 在方法末尾添加:
prior_prob = 0.01
bias_value = -math.log((1 - prior_prob) / prior_prob)
constant_(self.class_head.bias, bias_value)
```
- 注意: `math` 和 `constant_` 已在文件顶部导入，只需在方法内添加 3 行

---

#### P4 — models/__init__.py 未导出 v2

- 文件: `DSITT/models/__init__.py`
- 当前内容 (预计只有): `from .dsitt import build_dsitt`
- 修复: 添加一行:
```python
from .dsitt_v2 import build_dsitt_v2
```
- 验证: 修复后应能 `from models import build_dsitt, build_dsitt_v2`

---

#### P5 — TALA prev_q_idx 映射错位

- 文件: `DSITT/models/tracking/track_manager.py`
- 涉及行: 第 164-180 行 `TrajectoryAwareLabelAssignment.assign()` 中的 track query 分配逻辑
- 问题详解:
  - 帧 t: 有 5 个 track queries (index 0-4)，TALA 匹配后 `track_assignment = {tid_A: 0, tid_B: 2, tid_C: 3}`
  - QIM 筛选后只保留 matched 的 3 个 → 新的 track queries index 变为 0, 1, 2
  - 帧 t+1: `prev_track_assignment` 仍然是 `{tid_A: 0, tid_B: 2, tid_C: 3}`
  - 但实际 track queries 只有 3 个 (index 0-2)，`tid_B: 2` 和 `tid_C: 3` 的 index 已经不对了
- 当前错误代码 (第 170-180 行):
```python
for track_id, prev_q_idx in prev_track_assignment.items():
    gt_mask = (gt_track_ids == track_id)
    if gt_mask.any():
        gt_idx = gt_mask.nonzero(as_tuple=True)[0][0].item()
        if prev_q_idx < num_track_queries:  # ← 这里 prev_q_idx 可能是旧的绝对 index
            track_matched_q.append(prev_q_idx)
            track_matched_g.append(gt_idx)
            tracked_gt_indices.add(gt_idx)
```
- 修复方案: 在 `update()` 方法 (第 395-413 行) 中构建 `new_track_assignment` 时，将 value 改为在 matched_q 列表中的相对位置:
```python
# 当前代码 (第 232-236 行):
new_track_assignment = {}
for q_idx, g_idx in zip(all_matched_q, all_matched_g):
    tid = gt_track_ids[g_idx].item()
    new_track_assignment[tid] = q_idx  # ← 绝对 index

# 改为: 只存储 track queries 的相对位置
# 在 update() 中，matched_q 经过筛选后会被重新编号
# 所以 track_assignment 应该存储 track_id → 在新 track queries 中的序号
```
- 实际修复 (2026-04-07): 在 `track_manager.py` 和 `mtuq_manager.py` 的 `update()` 中，QIM 筛选 `matched_q` 后，用 dict comprehension 构建 `old_to_new` 映射并重写 `_track_assignment`:
```python
# 在 self._track_queries = ... 之后:
old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(matched_q)}
remapped = {}
for tid, old_q_idx in self._track_assignment.items():
    if old_q_idx in old_to_new:
        remapped[tid] = old_to_new[old_q_idx]
self._track_assignment = remapped
```
- 同时在 `len(matched_q) == 0` 分支中将 `_track_assignment` 置为 `{}`

---

#### P6 — 运动视图记忆库 track 数量变化时重置

- 文件: `DSITT/models/dsitt_v2.py`
- 涉及行: 第 195-209 行
- 当前错误代码:
```python
if n_track > 0 and self.memory_bank.length > 0:
    hist_feats, hist_boxes = self.memory_bank.get_history()
    if hist_feats is not None and hist_feats.shape[2] == n_track:
        # Track count matches → can use motion updater
        ...
    else:
        # Track count changed → reset memory  ← 太激进！
        self.memory_bank.reset()
```
- 问题: 任何 track 数量变化（新目标出现、旧目标消失）都会清空整个记忆库
- 修复方案 (简化版): 当 shape 不匹配时，跳过运动更新但不重置记忆库:
```python
if n_track > 0 and self.memory_bank.length > 0:
    hist_feats, hist_boxes = self.memory_bank.get_history()
    if hist_feats is not None and hist_feats.shape[2] == n_track:
        q_motion_track = queries['q_motion'][:, :n_track]
        q_motion_updated = self.motion_updater(
            q_motion_track, hist_feats, hist_boxes
        )
        queries['q_motion'] = torch.cat([
            q_motion_updated,
            queries['q_motion'][:, n_track:]
        ], dim=1)
    # else: shape 不匹配时跳过运动更新，但不重置记忆库
    # 记忆库会在下面的 push 中自然更新为新的 track 数量
```
- 同时修改记忆库 push 逻辑 (第 211-218 行): 当 track 数量变化时，重置记忆库再 push（因为旧的 per-track 特征与新的 track 不对应）:
```python
if n_track > 0:
    # 如果 track 数量变化，需要重置记忆库（旧特征与新 track 不对应）
    if self.memory_bank.length > 0:
        hist_feats, _ = self.memory_bank.get_history()
        if hist_feats is not None and hist_feats.shape[2] != n_track:
            self.memory_bank.reset()
    self.memory_bank.push(
        queries['q_fused'][:, :n_track].detach(),
        outputs_coord[:, :n_track].detach(),
    )
else:
    self.memory_bank.reset()
```
- 更彻底的修复 (进阶版): 记忆库按 track_id 存储，而非按位置。这需要重构 `TrajectoryMemoryBank` 为 dict-based 存储，复杂度较高，可作为后续优化

---

#### P7 — scale_diversity_loss 与论文不一致

- 文件: `DSITT/models/decoder/scale_adaptive_attn.py`
- 涉及行: 第 154-170 行
- 论文公式 (dsitt_paper.tex 第 248 行): `L_div = -Var({s_i})`
- 代码实现:
```python
def scale_diversity_loss(scale_params, target_std=0.15):
    std = scale_params.squeeze(-1).std(dim=-1).mean()
    return F.relu(target_std - std)  # 只在 std < target_std 时惩罚
```
- 分析: 代码实现其实比论文公式更合理（有上界，不会无限鼓励方差导致训练不稳定）
- 修复建议: 保持代码实现不变，修改论文公式以匹配代码:
```latex
% 将论文中的:
\mathcal{L}_\text{div} = -\text{Var}(\{s_i\}_{i=1}^N)
% 改为:
\mathcal{L}_\text{div} = \max(0, \sigma_\text{target} - \text{std}(\{s_i\}_{i=1}^N))
```
- 如果选择改代码匹配论文:
```python
def scale_diversity_loss(scale_params):
    var = scale_params.squeeze(-1).var(dim=-1).mean()
    return -var  # 始终鼓励多样性
```

---

#### P8 — dsitt_full.yaml 超参与论文不一致

- 文件: `DSITT/configs/dsitt_full.yaml`
- 不一致项:

| 参数 | 论文值 | 代码值 | 建议 |
|------|--------|--------|------|
| cls_weight (λ_c) | 2.0 | 5.0 | 以实验效果为准，但需统一 |
| focal_alpha | 0.25 | 0.5 | 同上 |
| num_queries | 300 (论文 §4.1) | 100 | 论文说 300，代码用 100 |

- 修复: 先用论文值跑一轮实验，再用代码值跑一轮，取效果好的那组。然后统一论文和代码
- 如果选择统一为论文值:
```yaml
loss:
  cls_weight: 2.0
  focal_alpha: 0.25
model:
  num_queries: 300
```

---

#### P9 — 无数据增强

- 需新建文件: `DSITT/datasets/transforms.py`
- 需修改文件: `DSITT/datasets/rgbt_tiny.py`
- 实现内容:
```python
# transforms.py
import random
import torch
import torchvision.transforms.functional as TF

class DualModalityTransform:
    """对 RGB 和 IR 做同步几何变换，仅对 RGB 做颜色变换"""

    def __init__(self, train=True):
        self.train = train

    def __call__(self, img_rgb, img_ir, boxes):
        """
        img_rgb: [3, H, W] tensor
        img_ir: [3, H, W] tensor
        boxes: [N, 4] (cx, cy, w, h) 归一化坐标
        """
        if not self.train:
            return img_rgb, img_ir, boxes

        # 1. 随机水平翻转 (RGB + IR 同步)
        if random.random() > 0.5:
            img_rgb = TF.hflip(img_rgb)
            img_ir = TF.hflip(img_ir)
            boxes[:, 0] = 1.0 - boxes[:, 0]  # cx 翻转

        # 2. 颜色抖动 (仅 RGB)
        if random.random() > 0.5:
            img_rgb = TF.adjust_brightness(img_rgb, random.uniform(0.8, 1.2))
            img_rgb = TF.adjust_contrast(img_rgb, random.uniform(0.8, 1.2))

        return img_rgb, img_ir, boxes
```
- 在 `rgbt_tiny.py` 的 `__getitem__` 中调用 transform

---

#### P10 — 评估指标不完整

- 文件: `DSITT/tools/eval.py`
- 修复方案 A (推荐): 集成 TrackEval 库
```bash
pip install trackeval
```
- 修复方案 B: 自行实现 HOTA
- HOTA 核心算法:
```python
def compute_hota(pred_tracks, gt_tracks, iou_thresholds=np.arange(0.05, 1.0, 0.05)):
    """
    对每个 IoU 阈值 α:
      1. 计算 DetA(α) = TP / (TP + FP + FN)  (检测准确率)
      2. 计算 AssA(α) = 平均(每个匹配对的关联准确率)  (关联准确率)
      3. HOTA(α) = sqrt(DetA(α) * AssA(α))
    最终 HOTA = mean(HOTA(α) for α in thresholds)
    """
```
- 同时需要将贪心匹配改为匈牙利匹配 (用 `scipy.optimize.linear_sum_assignment`)

---

#### P11 — 无学习率 warmup

- 文件: `DSITT/tools/train.py`
- 涉及行: 第 262-263 行 (scheduler 构建处)
- 当前代码:
```python
lr_drop = train_cfg.get('lr_drop_epoch', 100)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop, gamma=0.1)
```
- 修复:
```python
from torch.optim.lr_scheduler import LinearLR, StepLR, SequentialLR

warmup_iters = train_cfg.get('warmup_iters', 1000)
lr_drop = train_cfg.get('lr_drop_epoch', 100)

warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
main_scheduler = StepLR(optimizer, lr_drop, gamma=0.1)
lr_scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler],
                            milestones=[warmup_iters])
```
- 注意: SequentialLR 的 milestones 是按 step 计数的，需要确认 warmup_iters 是 iteration 还是 epoch。建议用 iteration (1000 步 ≈ 0.5 epoch)

---

#### P12 — 无 v2 冒烟测试

- 文件: 扩展 `DSITT/tools/test_model.py` 或新建 `DSITT/tools/test_model_v2.py`
- 需要测试:
```python
from models.dsitt_v2 import build_dsitt_v2

model = build_dsitt_v2()
model.train()

# 双模态输入
frames_rgb = [torch.randn(1, 3, 320, 320)]
frames_ir = [torch.randn(1, 3, 320, 320)]
targets = [{'labels': torch.randint(0, 7, (5,)),
            'boxes': torch.rand(5, 4) * 0.5 + 0.25,
            'track_ids': torch.arange(5)}]

loss_dict = model(frames_rgb, frames_ir, targets)
# 验证 loss_dict 包含: loss, loss_cls, loss_l1, loss_nwd, loss_cmc, gate_rgb, gate_ir, gate_motion

model.eval()
with torch.no_grad():
    result = model(frames_rgb, frames_ir)
# 验证 result['predictions'][0] 包含: scores, labels, boxes, gate_weights
```

---

#### P13 — FrozenBatchNorm2d 未使用

- 文件: `DSITT/models/backbone/resnet.py`
- 状态: P1 已通过重写 `train()` 解决了核心问题
- 可选进阶修复: 将 ResNet 中所有 `BatchNorm2d` 替换为 `FrozenBatchNorm2d`，这样即使有人忘记调用 `train()` 也不会出问题
- 实现: 在 `Backbone.__init__` 中，构建 ResNet 后遍历替换:
```python
if freeze_bn:
    self._freeze_bn_to_frozen(backbone)

def _freeze_bn_to_frozen(self, module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features)
            frozen.weight.copy_(child.weight)
            frozen.bias.copy_(child.bias)
            frozen.running_mean.copy_(child.running_mean)
            frozen.running_var.copy_(child.running_var)
            setattr(module, name, frozen)
        else:
            self._freeze_bn_to_frozen(child)
```
- 低优先级，当前 P1 的修复已足够

---

#### P14 — p_insert 未实现

- 文件: `DSITT/models/tracking/track_manager.py` QIM 类
- 设计意图: 训练时随机插入假阳性 track queries，模拟推理时的误检，增强模型对 FP 的鲁棒性
- 实现思路:
```python
# 在 QIM.forward() 中，active_mask 确定后:
if training and self.p_insert > 0:
    n_insert = int(self.p_insert * N_q)
    if n_insert > 0:
        # 随机选择 n_insert 个非活跃查询，强制激活
        inactive = (~active_mask[0]).nonzero(as_tuple=True)[0]
        if len(inactive) > 0:
            insert_idx = inactive[torch.randperm(len(inactive))[:n_insert]]
            active_mask[0, insert_idx] = True
```
- 低优先级，MOTR 原版也未必使用

---

#### P15 — 参考点无迭代精炼

- 文件: `DSITT/models/decoder/modality_aware_decoder.py` 和 `deformable_decoder.py`
- 参考: Deformable DETR 的 `with_box_refine` 模式
- 当前代码 (MAD decoder 第 291-294 行):
```python
reference_points = self.reference_point_head(query_pos).sigmoid()
ref_points_input = reference_points[:, :, None, :].repeat(1, 1, self.n_levels, 1)
# 所有层共享同一个 reference_points
```
- 修复: 每层解码后用预测的 cx, cy 偏移更新 reference_points:
```python
for layer_idx, layer in enumerate(self.layers):
    queries, gate_weights, scale_params = layer(...)

    # 用当前层的预测更新参考点
    q_fused_l = queries['q_fused']
    bbox_off_l = self.bbox_head(q_fused_l)
    new_ref = (reference_points + bbox_off_l[..., :2]).sigmoid().detach()
    reference_points = new_ref
    ref_points_input = reference_points[:, :, None, :].repeat(1, 1, self.n_levels, 1)
```
- 注意: `.detach()` 很重要，防止梯度通过参考点回传导致训练不稳定
- 这是一个较大的改动，建议在基础 bug 修完后再做

---

### 8.1 代码级 Bug / 不一致

| # | 严重度 | 位置 | 问题描述 |
|---|--------|------|----------|
| 1 | 🔴高 | `models/__init__.py` | 仅导出 `build_dsitt` (v1)，未导出 `build_dsitt_v2`。虽然 train/eval 直接 import，但模块接口不完整 |
| 2 | 🔴高 | `resnet.py` Backbone | BN 冻结不彻底：`__init__` 中调用 `module.eval()`，但 `model.train()` 会递归将所有子模块切回 train 模式，导致 BN 统计量在训练中被更新。需要重写 `train()` 方法或使用 `FrozenBatchNorm2d` |
| 3 | 🔴高 | `cmc_loss.py` | CMC 的 box 预测直接用 `bbox_head(q_rgb)` 得到原始偏移量，但实际 box 坐标需要加上 reference_points 并 sigmoid。CMC 比较的不是真实坐标，而是原始 MLP 输出，语义不一致 |
| 4 | 🟡中 | `dsitt_v2.py` L197 | 运动视图更新条件 `hist_feats.shape[2] == n_track`：任何跟踪数量变化（新目标出现/旧目标消失）都会导致记忆库重置，严重限制运动视图的实际效用 |
| 5 | 🟡中 | `track_manager.py` TALA | `prev_q_idx` 映射问题：上一帧的 query index 在经过 QIM 筛选后会被重新编号，但 TALA 直接用原始 index 做 `if prev_q_idx < num_track_queries`，可能导致错误的 track-GT 关联 |
| 6 | 🟡中 | 配置不一致 | `dsitt_full.yaml` 的 `num_queries=100` 与其他配置的 300 不一致；`cls_weight=5.0, focal_alpha=0.5` 与论文声称的 λ_c=2 不一致 |
| 7 | 🟡中 | `deformable_decoder.py` | v1 解码器的 `_reset_parameters` 未设置 class_head 的 bias prior (MAD 解码器有)，可能导致 v1 训练初期 focal loss 不稳定 |
| 8 | 🟡中 | `scale_adaptive_attn.py` | `scale_diversity_loss` 用 `ReLU(target_std - std)` 只在 std 低于阈值时惩罚，而论文写的是 `L_div = -Var({s_i})`（始终鼓励多样性），两者行为不同 |

### 8.2 功能缺失

| # | 优先级 | 描述 |
|---|--------|------|
| 1 | 🔴高 | **无数据增强**: 没有随机翻转、裁剪、颜色抖动。路线图提到但未实现，对泛化能力影响大 |
| 2 | 🔴高 | **评估指标不完整**: HOTA 在配置中列出但未实现；IDF1 是近似计算；使用贪心匹配而非匈牙利匹配 |
| 3 | 🔴高 | **论文实验数据全部为 TODO**: 所有表格数据为空，需要完成训练后填充 |
| 4 | 🟡中 | **无学习率 warmup**: 直接从全 LR 开始训练，可能导致早期不稳定 |
| 5 | 🟡中 | **无 best model 保存**: 只按固定间隔保存 checkpoint，没有基于验证指标的最优模型追踪 |
| 6 | 🟡中 | **无梯度累积**: batch_size=1 且无梯度累积，有效 batch size 始终为 1 |
| 7 | 🟡中 | **test_model.py 仅测试 v1**: 没有 v2 的冒烟测试 |
| 8 | 🟢低 | **无可视化工具**: eval.py 有 `--visualize` 参数但未实现可视化逻辑 |
| 9 | 🟢低 | **FrozenBatchNorm2d 定义但未使用**: resnet.py 中定义了但实际用的是手动冻结 |
| 10 | 🟢低 | **p_insert 未实现**: QIM 接受 p_insert 参数但从未使用（假阳性查询插入） |

### 8.3 架构设计隐患

| # | 描述 | 影响 | 建议 |
|---|------|------|------|
| 1 | **参考点无迭代精炼**: 两个解码器都从 query_pos 一次性生成参考点，所有层共享。Deformable DETR 原版有 iterative refinement | 定位精度受限 | 添加逐层参考点更新 |
| 2 | **仅支持 batch_size=1**: TALA 硬编码 `assert B == 1`，整个 pipeline 假设 B=1 | 训练效率低，无法利用多 GPU 数据并行 | 重构 TALA 支持 batch>1 |
| 3 | **双流骨干参数量翻倍 (81.8M)**: 两个完全独立的 ResNet-50 | 显存和计算开销大 | 论文已提到：共享前 2 个 stage 可减少约 40% |
| 4 | **clip_length > 2 导致 OOM**: 配置注释明确说明 32GB GPU 上双流 640×512 只能跑 clip_length=2 | 时序建模能力受限 | 需要梯度检查点或混合精度优化 |
| 5 | **运动视图对新目标无效**: 检测查询的 q_motion 是固定 Embedding，没有历史信息 | 新出现目标无法利用运动线索 | 可考虑用全局运动统计初始化 |
| 6 | **跨模态交换共享注意力权重**: Step 3 中 rgb→ir 和 ir→rgb 用不同的 MHA 模块，但论文路线图 V2 中用的是同一个 | 当前实现更灵活但参数更多 | 保持当前设计，但需在论文中说明 |

### 8.4 训练稳定性风险

| # | 风险 | 应对建议 |
|---|------|----------|
| 1 | Focal loss 初期不稳定 (v1 无 bias prior) | 为 v1 decoder 添加 class_head bias 初始化 |
| 2 | CMC 对比损失在匹配目标少时退化 | 已有 `M < 2` 时返回 0 的保护，但应考虑 warmup |
| 3 | 门控权重可能坍缩到单一模态 | 监控 gate_rgb/gate_ir/gate_motion 的分布 |
| 4 | NWD 常数 C=0.1 对归一化坐标敏感 | 需要调参验证 C ∈ {0.05, 0.1, 0.2} |
| 5 | 辅助解码损失权重为 1.0 (与主损失等权) | 可能需要降低辅助损失权重 (如 0.5) |

---

## 9. 数据流详解 (v2 训练模式)

```python
# 1. 数据加载
frames, targets = dataset[idx]
# frames = [(rgb_tensor, ir_tensor), ...] × clip_length
# targets = [{'labels': [M], 'boxes': [M,4], 'track_ids': [M]}, ...]

# 2. 双流骨干
srcs_rgb, pos_rgb, srcs_ir, pos_ir = dual_backbone(img_rgb, img_ir)
# 各 4 个尺度: [B, 256, H/4, W/4], [B, 256, H/8, W/8], ...

# 3. 双流编码
memory_rgb, shapes, starts = encoder_rgb(srcs_rgb, pos_rgb)  # [B, ΣHiWi, 256]
memory_ir, shapes, starts = encoder_ir(srcs_ir, pos_ir)

# 4. MTUQ 查询
queries = {q_rgb, q_ir, q_motion, q_fused}  # 各 [1, N_track+N_detect, 256]
query_pos  # [1, N_total, 256]

# 5. 运动视图更新 (仅 track queries 部分)
q_motion[:, :n_track] = motion_updater(q_motion[:, :n_track], hist_feats, hist_boxes)

# 6. MAD 解码 (6 层)
for layer in decoder.layers:
    queries, gate_weights, scale_params = layer(queries, query_pos, ref_points,
                                                 memory_rgb, ..., memory_ir, ...)
    aux_cls.append(class_head(queries['q_fused']))
    aux_coord.append(...)

# 7. 最终预测
outputs_class = aux_cls[-1]   # [B, N_total, 7]
outputs_coord = aux_coord[-1] # [B, N_total, 4]  (cx, cy, w, h) 归一化

# 8. TALA 标签分配
assignment = tala.assign(outputs_class, outputs_coord, targets[t],
                         n_track, prev_track_assignment)
# → matched_query_indices, matched_gt_indices, track_assignment

# 9. QIM 生成下一帧跟踪查询
track_queries, track_pos, active_mask = qim(queries, query_pos, scores)
# 训练时: 用 matched_query_indices 筛选
# 推理时: 用 score > 0.5 筛选

# 10. 记忆库更新
memory_bank.push(q_fused[:, :n_track].detach(), outputs_coord[:, :n_track].detach())

# 11. 损失计算 (clip 结束后)
loss = DSITTLoss(frame_outputs, targets, assignments)  # 主损失
loss += CMCLoss(frame_outputs, assignments)             # 跨模态一致性
loss += 0.1 * scale_diversity_loss(scale_params)        # SAS 正则
```

---

## 10. 论文状态

论文文件: `paper/dsitt_paper.tex`，CVPR 2026 格式。

已完成部分:
- Abstract ✅
- Introduction ✅
- Related Work ✅
- Method (全部 6 个子节) ✅
- Experiments 框架 ✅ (表格结构已搭好)
- Conclusion ✅

待完成:
- 所有实验数据 (标记为 `\TODO{}`)
- 5 张图 (标记为 `\FIGNEEDED{}`)
  - Fig.1: 动机三子图 (IoU 曲线 + 融合范式对比 + 数据集示例)
  - Fig.2: 完整架构图
  - Fig.3: MAD 单层结构图
  - Fig.4: 门控权重分析图
  - Fig.5: 跟踪轨迹对比可视化
- 补充材料 (超参敏感性分析)

---

## 11. 快速命令参考

```bash
# 进入项目目录
cd DSITT

# Dummy 数据冒烟测试 (不需要真实数据)
python tools/train.py --dummy --epochs 2 --print_freq 1 --config configs/dsitt_full.yaml

# 真实数据训练 (v2 完整模型)
python tools/train.py \
    --config configs/dsitt_full.yaml \
    --data_root data/rgbt_tiny \
    --epochs 50 --print_freq 200 --save_freq 10 \
    --output_dir outputs/train_v4 --num_workers 0

# 从 checkpoint 恢复
python tools/train.py \
    --config configs/dsitt_full.yaml \
    --data_root data/rgbt_tiny \
    --epochs 50 --output_dir outputs/train_v4 \
    --resume outputs/train_v4/checkpoints/checkpoint_0010.pth

# 评估
python tools/eval.py \
    --config configs/dsitt_full.yaml \
    --checkpoint outputs/train_v4/checkpoints/checkpoint_0050.pth \
    --data_root data/rgbt_tiny --score_threshold 0.1

# 数据集准备
mkdir -p data/rgbt_tiny
unzip -q data_split.zip -d data/rgbt_tiny/
unzip -q images.zip -d data/rgbt_tiny/images/
unzip -q annotations_coco.zip -d data/rgbt_tiny/annotations/
```

---

## 12. 关键超参速查

| 参数 | 值 | 文件 | 说明 |
|------|-----|------|------|
| d_model | 256 | 全局 | 特征维度 |
| nhead | 8 | 全局 | 注意力头数 |
| num_encoder_layers | 6 | 全局 | 编码器层数 |
| num_decoder_layers | 6 | 全局 | 解码器层数 |
| dim_feedforward | 1024 | 全局 | FFN 中间维度 |
| num_queries | 300 | configs | 检测查询数 |
| num_classes | 7 | configs | 类别数 |
| modality_dropout | 0.1 | dsitt_full | 模态随机丢弃率 |
| base_lr | 2e-4 | train | 基础学习率 |
| backbone_lr_factor | 0.1 | train | 骨干学习率倍率 |
| lr_drop_epoch | 100 | train | LR 下降 epoch |
| clip_max_norm | 0.1 | train | 梯度裁剪 |
| nwd_constant | 0.1 | loss | NWD 归一化常数 C |
| focal_alpha | 0.25 | loss | Focal loss α |
| focal_gamma | 2.0 | loss | Focal loss γ |
| cls_weight | 2.0 | loss | 分类损失权重 |
| box_l1_weight | 5.0 | loss | L1 框损失权重 |
| giou_weight | 2.0 | loss | NWD/GIoU 框损失权重 |
| cmc_consistency_weight | 1.0 | loss | CMC 一致性权重 |
| cmc_contrastive_weight | 0.5 | loss | CMC 对比权重 |
| cmc_temperature | 0.07 | loss | 对比学习温度 |
| scale_div_weight | 0.1 | loss | SAS 多样性权重 |
| memory_len | 5 | tracking | 轨迹记忆长度 |
| p_drop | 0.1 | tracking | 查询随机丢弃率 |
| max_offset | 0.5 | SAS | 最大采样偏移 |
| target_std | 0.15 | SAS | 多样性目标标准差 |

---

## 13. 更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-04-07 | 初始版本：完整代码分析，架构梳理，缺陷清单 |
| 2026-04-07 | P1 修复：resnet.py Backbone 重写 train() 方法，freeze_bn 时强制 BN 保持 eval |
| 2026-04-07 | P2/P3 代码丢失（未保存），SKILL 状态回退为待修复 |
| 2026-04-07 | P2 重新修复：dsitt_v2.py 传递 ref_points，cmc_loss.py 用 reference_points 计算 box 坐标 |
| 2026-04-07 | P3 重新修复：deformable_decoder.py _reset_parameters 添加 class_head bias prior |
| 2026-04-07 | P4 修复：models/__init__.py 添加 `from .dsitt_v2 import build_dsitt_v2` |
| 2026-04-07 | P5 修复：track_manager.py 和 mtuq_manager.py 的 update() 中，QIM 筛选后重映射 track_assignment 为连续索引 |
| 2026-04-07 | P6 修复：dsitt_v2.py 运动视图逻辑——track 数量不匹配时跳过运动更新而非重置记忆库；push 前检查 track 数量变化再决定是否重置 |
| 2026-04-07 | P7 修复：保持代码实现（hinge loss）不变，修改论文公式为 max(0, σ_target - std) 以匹配代码 |
| 2026-04-07 | P8 修复：dsitt_full.yaml 统一为论文值 (cls_weight=2.0, focal_alpha=0.25, num_queries=300)；dsitt_v2.py 默认参数同步更新 |
| 2026-04-07 | P9 修复：新建 datasets/transforms.py (DualModalityTransform)，rgbt_tiny.py __getitem__ 中集成——随机水平翻转(同步RGB+IR+boxes) + RGB颜色抖动 |
| 2026-04-07 | P10 修复：eval.py 贪心匹配→匈牙利匹配(scipy)，新增 HOTA(19个IoU阈值)、DetA、AssA 指标，IDF1 改为基于 track-level IDTP 计算 |

---

> ⚠️ 维护提醒：每次修复 bug、添加功能、完成实验后，请更新本文档对应章节。
> 特别是第 8 节（缺陷清单）和第 13 节（更新日志）。
