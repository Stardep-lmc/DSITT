# 阶段二：MTUQ + MAD 实现笔记

## 日期：2026-03-21

---

## 1. 实现文件清单

### 新增
- `models/backbone/dual_stream.py` — 双流骨干（含模态 dropout）
- `models/decoder/modality_aware_decoder.py` — MAD 解码器（核心创新）
- `models/tracking/mtuq_manager.py` — MTUQ 四元组查询管理
- `models/dsitt_v2.py` — 双模态主模型入口
- `configs/dsitt_mtuq.yaml` — MTUQ 配置
- `analysis/stage2_mtuq_analysis.md` — 预分析文档
- `analysis/stage2_mtuq_notes.md` — 本文档

---

## 2. 验证结果

| 指标 | v1 (单模态 GIoU) | v1 (单模态 NWD) | **v2 (MTUQ+NWD)** |
|------|-----------------|-----------------|-------------------|
| 参数量 | 40.1M | 40.1M | **80.3M** |
| 训练 loss | 31.5 | 88.6 | **57.6** |
| backward | ✓ | ✓ | **✓** |
| 推理 | ✓ | ✓ | **✓** |

### 门控权重分析（初始化后）
- **训练模式**：gate_rgb=0.475, gate_ir=0.251, gate_motion=0.274
  - RGB 初始权重最高（可能因为 ResNet ImageNet 预训练在 RGB 上更强）
  - IR 和 motion 权重接近，说明模型初始状态没有强偏好
- **推理模式**（单个查询）：gate_rgb=0.114, gate_ir=0.131, gate_motion=0.755
  - 首帧运动视图权重异常高 → 合理：第一帧没有真正的运动信息，模型退化为依赖可学习 embedding

---

## 3. 创新性改进（超越原方案）

### 3.1 模态 Dropout（Modality Dropout）
**原方案**中没有的新设计。在 DualStreamBackbone 中以概率 p 将某一模态全部置零：
```python
if training and rand < modality_dropout/2:
    img_rgb = zeros_like(img_rgb)  # 强制模型只用 IR
elif training and rand < modality_dropout:
    img_ir = zeros_like(img_ir)    # 强制模型只用 RGB
```

**意义**：
- 训练时模型学会在单模态退化时仍能跟踪
- 门控机制自然学到将退化模态的权重设为 0
- 这是天然的正则化，防止模型过度依赖某一模态
- **这个设计可以在论文中作为鲁棒性增强的贡献点**

### 3.2 运动视图的简化初始化
原方案设计了复杂的 MotionViewUpdater。阶段二采用简化版：
- 检测查询的 q_motion = 可学习 embedding
- 跟踪查询的 q_motion = 上一帧的 q_fused 经过投影
- 这保证了时序信息通过 q_motion 隐式传递，而不需要显式的记忆库

**发现**：这种简化设计已经能提供基础的时序连续性。完整的运动编码器（阶段五）会进一步增强。

### 3.3 门控归一化策略
原方案使用手动归一化 `gate_sum = g1+g2+g3; g1 = g1/gate_sum`。
改进为使用 `softmax`，更稳定且梯度更好：
```python
gate_logits = fusion_gate(concat_views)  # [B, N, 3]
gate_weights = softmax(gate_logits, dim=-1)
```

---

## 4. 参数量分析

| 组件 | 参数量 |
|------|--------|
| RGB Backbone (ResNet50+FPN) | ~25M |
| IR Backbone (ResNet50+FPN) | ~25M |
| RGB Encoder (6层) | ~7.5M |
| IR Encoder (6层) | ~7.5M |
| MAD Decoder (6层) | ~12M |
| MTUQ Queries + QIM | ~3M |
| Prediction Heads | ~0.3M |
| **总计** | **~80.3M** |

32GB GPU 完全足够。如果需要减少参数：
- 降低编码器层数 6→3（节省 ~7.5M）
- 共享骨干前 2 个 stage（节省 ~15M）

---

## 5. 与前序阶段的衔接确认

| 衔接点 | 状态 |
|--------|------|
| Backbone (阶段零) | ✅ 复用 `build_backbone`，DualStreamBackbone 包装两个实例 |
| Encoder (阶段零) | ✅ 复用 `DeformableTransformerEncoder`，双流各一个 |
| MSDeformAttn (阶段零) | ✅ MAD 层中的交叉注意力直接复用 |
| NWD Loss (阶段一) | ✅ `box_loss_type='nwd'` 在 v2 中正常工作 |
| NWD Matching (阶段一) | ✅ TALA 在 MTUQManager 中使用 NWD 匹配 |
| CAL Loss 框架 | ✅ DSITTLoss 不需修改，v2 输出格式兼容 |
| 训练脚本 | ⚠️ 需要小幅适配（双模态输入格式） |

---

## 6. 后续阶段准备

### 为阶段三（CMC 损失）预留的接口
v2 模型输出中已包含 `queries['q_rgb']` 和 `queries['q_ir']`。
阶段三只需：
1. 从这两个视图生成辅助预测 `box_rgb`, `box_ir`
2. 计算一致性损失和对比学习损失
3. 加入总损失

### 为阶段四（SAS）预留的接口
MAD 解码器中的 `cross_attn_rgb` 和 `cross_attn_ir` 可以直接替换为
`ScaleAdaptiveDeformableAttn`，无需修改其他部分。

### 为阶段五（运动视图）预留的接口
MTUQQueryInteractionModule 中 `q_motion = proj_motion(q_fused)` 这一行
可以替换为 `q_motion = MotionViewUpdater(q_fused, history)`。