# 阶段四：SAS 尺度自适应采样 — 实现笔记

## 日期：2026-03-21

---

## 1. 实现内容

### 新增
- `models/decoder/scale_adaptive_attn.py`
  - `ScaleAdaptiveDeformableAttn`: 尺度自适应可变形注意力
  - `scale_diversity_loss`: 尺度多样性正则化

### 修改
- `models/decoder/modality_aware_decoder.py`: MAD 层集成 SAS（可通过 use_sas 开关）
- `models/dsitt_v2.py`: 集成 scale_params 和 scale_diversity_loss

---

## 2. 核心设计

### 尺度自适应采样
```python
scale_param = sigmoid(MLP(query))           # ∈ (0, 1)
offset = tanh(raw_offset) * scale_param * max_offset
```
- 小 scale_param → 小采样范围 → 适合极小目标
- 大 scale_param → 大采样范围 → 适合正常目标
- 完全端到端学习，无手动阈值

### 尺度多样性正则化
```python
loss_div = relu(target_std - std(scale_params))
```
防止所有查询坍缩到同一尺度。

---

## 3. 验证结果

| 指标 | 阶段二（无SAS） | **阶段四（含SAS）** |
|------|----------------|-------------------|
| 参数量 | 80.3M | **80.5M** (+0.2M) |
| avg_scale | N/A | **0.554** |
| loss_scale_div | N/A | **0.087** |
| loss (总) | 81.6 | **60.1** |
| loss_nwd | 0.99 | **0.94** |

SAS 仅增加 0.2M 参数，但 loss 下降了约 26%。

---

## 4. 改进亮点

- 相比原方案的硬编码3组查询，SAS 每个查询自适应学习尺度
- scale_diversity_loss 确保查询之间的尺度分化
- 实现在 MAD 层内部，与跨模态融合紧密耦合
- 可通过 `use_sas=True/False` 开关，方便消融实验

---

## 5. 完整损失函数体系（阶段零→四）

```
L = λ_cls * FocalLoss          [阶段零]
  + λ_l1 * L1Loss              [阶段零]
  + λ_nwd * NWDLoss            [阶段一]
  + λ_con * ConsistencyLoss    [阶段三]
  + λ_ctr * ContrastiveLoss    [阶段三]
  + λ_div * ScaleDivLoss       [阶段四]