# 阶段三：CMC 跨模态一致性损失 — 实现笔记

## 日期：2026-03-21

---

## 1. 实现内容

### 新增
- `models/loss/cmc_loss.py` — CMC 损失模块
  - `PredictionConsistencyLoss`: 预测一致性（L1 box + 对称 KL 分类）
  - `ContrastiveAlignmentLoss`: 对比对齐（InfoNCE）
  - `CMCLoss`: 组合损失

### 修改
- `models/dsitt_v2.py` — 集成 CMC 到训练流程

---

## 2. 验证结果

| 损失项 | 值 |
|--------|-----|
| loss (总) | 81.63 |
| loss_cls | 37.96 |
| loss_nwd | 0.99 |
| loss_l1 | 0.30 |
| **loss_cmc** | **2.25** |
| - loss_consistency | 1.58 |
| - loss_contrastive | 1.34 |

CMC 贡献总 loss 的 ~2.8%，不会主导训练，但提供持续的跨模态对齐信号。

---

## 3. 改进亮点（超越原方案）

### 3.1 对称 KL 散度 + 单侧 Detach
原方案只用单向 KL。改进为对称 KL，且各自 detach 对方作为目标：
```python
kl_rgb_ir = KL(log_softmax(rgb), softmax(ir.detach()))
kl_ir_rgb = KL(log_softmax(ir), softmax(rgb.detach()))
cls_loss = (kl_rgb_ir + kl_ir_rgb) / 2
```
**好处**：防止两个视图坍缩到同一个解，保持各自模态特性。

### 3.2 共享预测头
q_rgb 和 q_ir 的辅助预测使用与 q_fused 相同的 class_head 和 bbox_head。
**好处**：
- 不增加任何新参数
- 强制三个视图在同一预测空间中对齐
- 简化实现

### 3.3 可开关设计
`model.use_cmc = True/False` 一行代码切换，方便消融实验。

---

## 4. 对后续阶段的影响

### 阶段四（SAS）
CMC 损失完全独立于注意力机制，不会与 SAS 冲突。

### 阶段五（运动视图）
运动视图目前不参与 CMC（只有 RGB 和 IR 视图）。
未来可以考虑：运动视图与融合视图的时序一致性损失。

---

## 5. 整体架构至此

```
阶段零: 基线 (ResNet50+FPN → DeformEnc → DeformDec → TALA)  40.1M
阶段一: + NWD Loss (替换 GIoU)                                 +0M
阶段二: + 双流骨干 + MAD解码器 + MTUQ                           80.3M  
阶段三: + CMC Loss (一致性 + 对比学习)                           +0M (共享预测头)
                                                              ─────
当前总计:                                                       80.3M
```

**所有阶段零→三的损失函数**：
```
L_total = λ_cls * FocalLoss 
        + λ_l1 * L1Loss
        + λ_nwd * NWDLoss
        + λ_con * ConsistencyLoss     [阶段三 CMC]
        + λ_ctr * ContrastiveLoss     [阶段三 CMC]