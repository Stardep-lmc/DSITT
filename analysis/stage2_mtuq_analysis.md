# 阶段二：MTUQ + 模态感知解码器 — 深度技术分析

## 日期：2026-03-21
## 目的：在动手编码前，仔细分析阶段二的技术方案，确保与阶段零/一无缝衔接

---

## 1. 阶段二的目标回顾

根据 `second_round_review.md` 中的 MTUQ 方案：
- 将跟踪查询从**单一向量** `q ∈ R^d` 扩展为**结构化多视图表征** `Q = {q_rgb, q_ir, q_motion, q_fused}`
- 实现**模态感知解码器（MAD）**，在查询级别完成跨模态融合
- 双流骨干网络独立处理 RGB 和 IR

---

## 2. 当前代码结构分析（阶段零+阶段一之后）

### 2.1 数据流（当前单模态）
```
image [B,3,H,W]
  → Backbone (ResNet50 + FPN)  → srcs [4×(B,256,Hi,Wi)], pos [4×(B,256,Hi,Wi)]
  → Encoder (6层 DeformAttn)   → memory [B, ΣHiWi, 256], spatial_shapes, level_idx
  → TrackManager.get_queries() → tgt [B, N_q, 256], query_pos [B, N_q, 256]
  → Decoder (6层)              → hidden [B, N_q, 256], cls [B, N_q, 7], box [B, N_q, 4]
  → TrackManager.update()      → assignment, track_queries for next frame
```

### 2.2 需要改动的模块

| 模块 | 当前状态 | 阶段二改动 |
|------|---------|-----------|
| `backbone/resnet.py` | 单流 ResNet50+FPN | 复用为双流（实例化两个 Backbone） |
| `encoder/deformable_encoder.py` | 单编码器 | 双流编码器（各处理一个模态） |
| `decoder/deformable_decoder.py` | 标准解码器 | **替换为 MAD 解码器** |
| `tracking/track_manager.py` | 单向量查询 | **扩展为 MTUQ 四元组** |
| `models/dsitt.py` | 单模态主模型 | **双模态主模型** |
| `datasets/rgbt_tiny.py` | 支持单/双模态 | 启用双模态返回 |
| `loss/losses.py` | 单预测损失 | 增加 q_rgb/q_ir 辅助损失头 |

### 2.3 不需要改动的模块
- `ops/ms_deform_attn.py` — 可变形注意力核心不变
- `loss/nwd_loss.py` — NWD 损失不变
- `tools/train.py` — 训练循环结构不变（帧级迭代）

---

## 3. 技术方案详细设计

### 3.1 双流骨干网络

**设计选择**：不创建新类，直接在主模型中实例化两个 `Backbone` 对象。

```python
# 在 DSITT.__init__ 中:
self.backbone_rgb = build_backbone(d_model, pretrained=True)
self.backbone_ir  = build_backbone(d_model, pretrained=True)
# 权重独立，不共享
```

**显存考量**：
- 单流 backbone ≈ 25M params
- 双流 ≈ 50M params
- 32GB GPU 足够（推理时约 2×前向计算）

**兼容性**：保留 `self.backbone` 用于单模态，新增 `self.backbone_rgb`/`self.backbone_ir` 用于双模态。通过 `modality` 参数切换。

### 3.2 双流编码器

**方案 A（推荐）**：两个独立编码器
```python
self.encoder_rgb = DeformableTransformerEncoder(...)
self.encoder_ir  = DeformableTransformerEncoder(...)
```

**方案 B**：共享编码器 + 模态标识嵌入
- 两个模态共用一个编码器，通过额外的模态嵌入区分
- 参数更少，但特征可能互相干扰

**决策**：采用方案 A。虽然参数翻倍，但双模态特征需要独立处理以保持各自模态的特性。后续如果显存不足，可以退回方案 B。

### 3.3 MTUQ 查询结构

**当前查询**（TrackQueryManager）：
```python
tgt: [B, N_q, 256]       # 查询特征
query_pos: [B, N_q, 256]  # 位置嵌入
```

**扩展为 MTUQ 四元组**：
```python
queries = {
    'q_rgb':    [B, N_q, 256],  # RGB 视图
    'q_ir':     [B, N_q, 256],  # IR 视图
    'q_motion': [B, N_q, 256],  # 运动视图
    'q_fused':  [B, N_q, 256],  # 融合查询（用于最终预测）
}
query_pos: [B, N_q, 256]  # 位置嵌入（四个视图共享）
```

**关键设计决策**：
- 位置嵌入在四个视图间共享（因为它们表示同一个空间位置的目标）
- 检测查询：四个视图各有独立的可学习 Embedding
- 跟踪查询：从上一帧的 MAD 输出继承四元组

### 3.4 模态感知解码器（MAD）层

每个 MAD 层的处理流程：

```
输入: {q_rgb, q_ir, q_motion, q_fused}, query_pos, F_rgb_memory, F_ir_memory

Step 1: 查询自注意力
  all_q = concat(q_fused)  # 所有查询间做自注意力
  q_fused = SelfAttn(all_q, all_q, all_q)

Step 2: 模态内交叉注意力
  q_rgb  = DeformCrossAttn(q_rgb + query_pos, F_rgb_memory)
  q_ir   = DeformCrossAttn(q_ir + query_pos, F_ir_memory)

Step 3: 跨模态查询交互
  q_rgb' = q_rgb + MultiheadAttn(query=q_rgb, key=q_ir, value=q_ir)
  q_ir'  = q_ir  + MultiheadAttn(query=q_ir, key=q_rgb, value=q_rgb)

Step 4: 三视图自适应融合
  concat = [q_rgb', q_ir', q_motion]
  gate_rgb = σ(W_rgb · concat)
  gate_ir  = σ(W_ir · concat)  
  gate_mot = σ(W_mot · concat)
  gates_normalized = softmax(gate_rgb, gate_ir, gate_mot)
  q_fused = FFN(Σ gate_i * q_i)

输出: {q_rgb', q_ir', q_motion, q_fused}
```

### 3.5 预测头

**主预测**（用 q_fused）：
```python
cls = ClassHead(q_fused)    # [B, N_q, 7]
box = BoxHead(q_fused)      # [B, N_q, 4]
```

**辅助预测**（用于后续阶段的 CMC 损失，阶段二暂不启用）：
```python
box_rgb = BoxHead(q_rgb)    # 辅助
box_ir  = BoxHead(q_ir)     # 辅助
```

### 3.6 运动视图

阶段二中，运动视图使用**简化版**：
- 第一帧：q_motion 初始化为可学习 embedding
- 后续帧：q_motion = 上一帧的 q_fused（直接传递，不做复杂的运动编码）
- 完整的 MotionViewUpdater 留到阶段五实现

---

## 4. 实现步骤（编码顺序）

### Step 2.1: 创建 `models/backbone/dual_stream.py`
- 包装两个 Backbone 实例
- 支持单模态（退化为单流）和双模态

### Step 2.2: 创建 `models/decoder/modality_aware_decoder.py`
- MAD 解码器层
- MAD 解码器堆栈
- 保留标准解码器不动（可切换）

### Step 2.3: 修改 `models/tracking/track_manager.py`
- MTUQ 查询管理：四元组的创建、传递、更新
- QIM 扩展为四元组投影

### Step 2.4: 创建 `models/dsitt_v2.py`（双模态主模型）
- 不修改原 dsitt.py，新建 v2 版本
- 支持 `modality='both'` 时使用双流+MAD
- 支持 `modality='ir'/'rgb'` 时退化为单流+标准解码器

### Step 2.5: 修改 `datasets/rgbt_tiny.py`
- 当 modality='both' 时返回 (img_rgb, img_ir) 对

### Step 2.6: 更新配置和训练脚本
- `configs/dsitt_mtuq.yaml`
- 训练脚本适配双模态输入

### Step 2.7: 验证测试
- Smoke test: 双模态前向+反向
- Dummy 训练: 2 epochs

---

## 5. 与阶段零/一的衔接点

### 5.1 骨干网络
- 阶段零的 `Backbone` 类完全复用，不做任何修改
- 双流只是实例化两个 Backbone

### 5.2 编码器
- 阶段零的 `DeformableTransformerEncoder` 完全复用
- 双流只是实例化两个 Encoder

### 5.3 可变形注意力
- `MSDeformAttn` 不变，MAD 层中的 DeformCrossAttn 直接复用

### 5.4 NWD 损失（阶段一）
- NWD 在 MAD 模型中同样使用
- `box_loss_type='nwd'` 配置在 v2 模型中继续有效
- 匹配代价同样支持 NWD

### 5.5 TALA
- 轨迹感知标签分配逻辑不变
- 只是查询从单向量变为四元组中的 q_fused

### 5.6 CAL 损失
- Collective Average Loss 的框架不变
- 增加 q_rgb/q_ir 的辅助预测输出（为阶段三 CMC 损失预留接口）

---

## 6. 风险评估

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| MAD 解码器内存溢出 | 中 | 高 | 减少解码器层数(6→4)或共享编码器 |
| 跨模态注意力训练不稳定 | 中 | 中 | 初期固定 gate_mot=0，只训练 RGB/IR |
| 四元组查询传递逻辑复杂 | 中 | 中 | 充分的单元测试 |
| 单模态退化兼容问题 | 低 | 高 | 保留 dsitt.py 不动 |

---

## 7. 预估工作量

| 步骤 | 预估时间 | 复杂度 |
|------|---------|--------|
| dual_stream.py | 30min | 低 |
| modality_aware_decoder.py | 2-3h | 高 |
| track_manager MTUQ 扩展 | 1-2h | 中 |
| dsitt_v2.py | 1-2h | 中 |
| 数据集双模态支持 | 30min | 低 |
| 配置+训练脚本 | 30min | 低 |
| 验证测试 | 1h | 中 |
| **总计** | **~8h** | |