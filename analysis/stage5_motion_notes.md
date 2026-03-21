# 阶段五：运动视图增强 — 实现笔记

## 日期：2026-03-21

---

## 1. 实现内容

### 新增
- `models/tracking/motion_view.py`
  - `MotionViewUpdater`: 轨迹运动编码器 (2层Transformer + 门控注入)
  - `TrajectoryMemoryBank`: 轨迹记忆库 (存储最近K帧的q_fused和预测框)

### 修改
- `models/dsitt_v2.py`: 集成运动视图更新到帧循环中

---

## 2. 关键设计

### 运动编码器
- 输入: 历史K帧的 q_fused [K,B,N,d] + 预测框 [K,B,N,4]
- 计算帧间速度: Δbox = box[t] - box[t-1]
- 位置编码: MLP(concat(box, velocity)) → d_model
- 时序编码: 2层 Transformer Encoder (batch化, 无逐目标循环)
- 门控注入: q_motion += gate * proj(motion_token)

### 记忆库策略
- 只存储 track 查询（大小固定），不存 detect 查询
- 当 track 数量变化时自动重置（避免 size mismatch）
- 最大长度 K=5，超出自动弹出最早的帧

---

## 3. Bug 修复记录

### Size Mismatch Bug
**问题**: 帧间查询数不同 (frame1: 300 detect, frame2: 300+4 track) 导致 torch.stack 失败
**修复**: memory bank 只存储 track queries (前 n_track 个), 并在 track 数变化时重置

### 改进: Batched Temporal Encoding
V2 路线图原方案是 per-target loop: `for i in range(N): encode(seq[:, i])`
改进为 reshape + batch 编码: `encode(seq.reshape(B*N, K, d))`
速度提升约 N 倍 (N=查询数)

---

## 4. 验证结果

| 指标 | 阶段四 (无运动) | **阶段五 (含运动)** |
|------|----------------|-------------------|
| 参数量 | 80.5M | **81.8M** (+1.3M) |
| 训练 loss | 60.1 | **52.0** |
| gate_motion | 0.43 | **0.32** |
| 推理 | 2帧 OK | **3帧 OK** |

---

## 5. 完整架构总结 (阶段零→五)

```
完整 DSITT v2 数据流:
                                              
(img_rgb, img_ir) × T 帧
      ↓
[DualStreamBackbone + ModalityDropout]        ← 阶段二
  → (F_rgb, F_ir) × 4 尺度
      ↓
[DualStreamEncoder] × 2                       ← 阶段零
  → (memory_rgb, memory_ir)
      ↓
[MTUQManager.get_queries]                     ← 阶段二
  → {q_rgb, q_ir, q_motion, q_fused} + pos
      ↓
[MotionViewUpdater(memory_bank)]              ← 阶段五
  → q_motion enriched with trajectory history
      ↓
[ModalityAwareDecoder] × 6 layers            ← 阶段二
  Step 1: Self-attention (q_fused)
  Step 2: SAS Cross-attention (q_rgb↔F_rgb, q_ir↔F_ir)  ← 阶段四
  Step 3: Cross-modal interaction (q_rgb↔q_ir)
  Step 4: Adaptive 3-view fusion → q_fused
      ↓
[Prediction Heads]
  → cls, box (from q_fused)
  → box_rgb, box_ir (auxiliary, for CMC)     ← 阶段三
      ↓
[Loss]
  = FocalLoss + L1 + NWD                     ← 阶段零+一
  + CMC(consistency + contrastive)            ← 阶段三  
  + ScaleDiversityLoss                       ← 阶段四
      ↓
[MTUQManager.update + MemoryBank.push]        ← 阶段二+五
  → track queries for next frame

参数量: 81.8M
损失项: 6 个 (Focal, L1, NWD, Consistency, Contrastive, ScaleDiv)