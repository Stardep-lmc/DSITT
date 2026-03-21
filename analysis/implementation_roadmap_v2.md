# DSITT 核心模块实现路线图 V2（基于二次审视修订版）

## 修订说明

本文档是 `implementation_roadmap.md` 的升级版，基于 `second_round_review.md` 的深度反思进行了根本性调整：
- 将四个独立模块的拼凑方案 → 统一为以 **MTUQ（模态-时序统一查询）** 为核心的框架
- 新增 **跨模态一致性损失（CMC）** 作为全新创新点
- 将硬编码多粒度查询 → 改为 **尺度自适应采样（SAS）**
- 时序模块不再独立，而是作为 MTUQ 的运动视图集成

---

## 阶段零：环境与基线搭建（预计 2 周）

### Step 0.1 — 数据集准备
- 下载 RGBT-Tiny 数据集
- **关键确认**：验证数据集确实提供配对的 RGB + IR 图像
- 理解标注格式：bbox `(x, y, w, h)` + track_id + category（7 类）
- 确认划分：训练集 85 序列，测试集 30 序列
- 编写双模态 dataloader：每个样本同时返回 `(img_rgb, img_ir, annotations)`

### Step 0.2 — 复现 MOTR 基线
- 拉取 MOTR 官方仓库（https://github.com/megvii-research/MOTR）
- 适配到 RGBT-Tiny：先用单模态（仅 IR）输入跑通
- 记录基线指标：HOTA / MOTA / IDF1 / IDS / DetA / AssA
- 编写评估脚本，集成 SAFit 指标

### Step 0.3 — 搭建代码框架
- 基于 MOTR 代码结构，创建 DSITT 项目目录
- 确保训练、推理、评估流程完整可用
- 配置好 GPU 环境（确认显存是否支持双流骨干，估算约需 32GB+）

---

## 阶段一：NWD 增强损失 + 匹配代价（预计 3-5 天）

> 这是最简单的改动，但效果显著，应最先完成。

### Step 1.1 — 实现 NWD 距离计算模块

```python
# 文件: models/loss/nwd_loss.py

def bbox_to_gaussian(bbox):
    """将 (cx, cy, w, h) 建模为二维高斯 N(μ, Σ)"""
    cx, cy, w, h = bbox.unbind(-1)
    mu = torch.stack([cx, cy], dim=-1)
    sigma = torch.stack([w/2, h/2], dim=-1)  # 标准差 = 宽高的一半
    return mu, sigma

def wasserstein_distance_2d(mu_p, sigma_p, mu_g, sigma_g):
    """计算两个对角高斯分布的二阶 Wasserstein 距离"""
    w2 = ((mu_p - mu_g) ** 2).sum(-1) + ((sigma_p - sigma_g) ** 2).sum(-1)
    return w2

def nwd(pred_bbox, gt_bbox, C=4.0):
    """归一化 Wasserstein 距离，值域 (0, 1]"""
    mu_p, sigma_p = bbox_to_gaussian(pred_bbox)
    mu_g, sigma_g = bbox_to_gaussian(gt_bbox)
    w2 = wasserstein_distance_2d(mu_p, sigma_p, mu_g, sigma_g)
    return torch.exp(-w2 / (C ** 2))
```

### Step 1.2 — 替换损失函数
- 定位 MOTR 中的 GIoU Loss 计算位置
- 替换为：`L_nwd = 1 - NWD(pred_box, gt_box)`
- 总损失：`L = λ_cls * FocalLoss + λ_nwd * L_nwd + λ_l1 * L1Loss`
- 初始权重：`λ_nwd = 2.0`

### Step 1.3 — 替换匈牙利匹配代价
- 定位 MOTR 中 TALA 的匹配代价计算
- 将 IoU cost 替换为 NWD cost：`C_nwd = 1 - NWD(pred, gt)`
- 匹配代价：`C = λ_cls * C_cls + λ_nwd * C_nwd + λ_l1 * C_l1`

### Step 1.4 — 验证
- 对比 baseline（GIoU）vs NWD Loss，RGBT-Tiny 上训练
- 关注按尺度分组的检测精度（<8px / 8-16px / 16-32px）
- 调优超参 C ∈ {2, 4, 8}

---

## 阶段二：MTUQ + 模态感知解码器（核心！预计 3 周）

> 这是整个论文的灵魂。决定成败。

### Step 2.1 — 构建双流特征提取

```python
# 文件: models/backbone/dual_stream.py

class DualStreamBackbone(nn.Module):
    def __init__(self):
        self.backbone_rgb = ResNet50(pretrained=True)
        self.backbone_ir  = ResNet50(pretrained=True)
        self.fpn_rgb = FPN(in_channels=[256, 512, 1024, 2048])
        self.fpn_ir  = FPN(in_channels=[256, 512, 1024, 2048])
        # 两流独立参数，不共享权重
    
    def forward(self, img_rgb, img_ir):
        feats_rgb = self.fpn_rgb(self.backbone_rgb(img_rgb))  # {P2, P3, P4, P5}
        feats_ir  = self.fpn_ir(self.backbone_ir(img_ir))
        return feats_rgb, feats_ir
```

- 两个分支各自经过 Deformable DETR 的编码器
- 输出：`F_rgb = {F_rgb_l}` 和 `F_ir = {F_ir_l}`，l = 多尺度层

### Step 2.2 — 实现 MTUQ 查询结构

```python
# 文件: models/tracking/mtuq.py

class MTUQuery:
    """模态-时序统一查询"""
    def __init__(self, d_model=256, num_queries=300):
        # 检测查询：每个包含四个视图
        self.detect_q_rgb    = nn.Embedding(num_queries, d_model)
        self.detect_q_ir     = nn.Embedding(num_queries, d_model)
        self.detect_q_motion = nn.Embedding(num_queries, d_model)
        self.detect_q_fused  = nn.Embedding(num_queries, d_model)
    
    def init_detect_queries(self):
        """返回检测查询的四元组"""
        return {
            'q_rgb':    self.detect_q_rgb.weight,     # [N, d]
            'q_ir':     self.detect_q_ir.weight,      # [N, d]
            'q_motion': self.detect_q_motion.weight,  # [N, d]
            'q_fused':  self.detect_q_fused.weight,   # [N, d]
        }
    
    def create_track_queries(self, prev_hidden_states):
        """从上一帧的隐藏状态创建跟踪查询
        prev_hidden_states: 上一帧解码器输出的四元组
        """
        return {
            'q_rgb':    prev_hidden_states['q_rgb'].detach(),
            'q_ir':     prev_hidden_states['q_ir'].detach(),
            'q_motion': self.update_motion_view(prev_hidden_states),
            'q_fused':  prev_hidden_states['q_fused'].detach(),
        }
```

### Step 2.3 — 实现模态感知解码器（MAD）

```python
# 文件: models/decoder/modality_aware_decoder.py

class ModalityAwareDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        # Step 1: 模态内交叉注意力
        self.cross_attn_rgb = DeformableAttn(d_model, nhead)
        self.cross_attn_ir  = DeformableAttn(d_model, nhead)
        
        # Step 2: 跨模态查询交互
        self.cross_modal_attn = nn.MultiheadAttention(d_model, nhead)
        
        # Step 3: 运动视图更新
        self.motion_update = nn.MultiheadAttention(d_model, nhead)
        
        # Step 4: 统一融合
        self.fusion_gate_rgb = nn.Linear(d_model * 3, d_model)
        self.fusion_gate_ir  = nn.Linear(d_model * 3, d_model)
        self.fusion_gate_mot = nn.Linear(d_model * 3, d_model)
        self.fusion_proj     = nn.Linear(d_model, d_model)
        
        # FFN
        self.ffn = FFN(d_model)
    
    def forward(self, queries, F_rgb, F_ir, history_queries=None):
        q_rgb, q_ir, q_motion, q_fused = (
            queries['q_rgb'], queries['q_ir'], 
            queries['q_motion'], queries['q_fused']
        )
        
        # Step 1: 各模态查询与自己的特征图交互
        q_rgb = self.cross_attn_rgb(q_rgb, F_rgb)
        q_ir  = self.cross_attn_ir(q_ir, F_ir)
        
        # Step 2: 跨模态查询交互
        # RGB 视图询问 IR 视图
        q_rgb_enhanced = q_rgb + self.cross_modal_attn(
            query=q_rgb, key=q_ir, value=q_ir
        )[0]
        # IR 视图询问 RGB 视图
        q_ir_enhanced = q_ir + self.cross_modal_attn(
            query=q_ir, key=q_rgb, value=q_rgb
        )[0]
        
        # Step 3: 运动视图更新（用历史查询序列做自注意力）
        if history_queries is not None:
            q_motion = q_motion + self.motion_update(
                query=q_motion,
                key=history_queries,
                value=history_queries
            )[0]
        
        # Step 4: 自适应融合三个视图
        concat_views = torch.cat([q_rgb_enhanced, q_ir_enhanced, q_motion], dim=-1)
        gate_rgb = torch.sigmoid(self.fusion_gate_rgb(concat_views))
        gate_ir  = torch.sigmoid(self.fusion_gate_ir(concat_views))
        gate_mot = torch.sigmoid(self.fusion_gate_mot(concat_views))
        
        # 门控归一化
        gate_sum = gate_rgb + gate_ir + gate_mot + 1e-6
        gate_rgb, gate_ir, gate_mot = (
            gate_rgb / gate_sum, gate_ir / gate_sum, gate_mot / gate_sum
        )
        
        q_fused = self.fusion_proj(
            gate_rgb * q_rgb_enhanced + 
            gate_ir * q_ir_enhanced + 
            gate_mot * q_motion
        )
        q_fused = self.ffn(q_fused)
        
        return {
            'q_rgb': q_rgb_enhanced,
            'q_ir': q_ir_enhanced,
            'q_motion': q_motion,
            'q_fused': q_fused,
        }
```

### Step 2.4 — 集成到主模型

```python
# 文件: models/dsitt.py

class DSITT(nn.Module):
    def forward(self, frames_rgb, frames_ir):
        results = []
        track_queries = None
        history_buffer = []  # 存储历史查询用于运动视图
        
        for t, (rgb, ir) in enumerate(zip(frames_rgb, frames_ir)):
            # 1. 双流特征提取
            F_rgb, F_ir = self.dual_backbone(rgb, ir)
            F_rgb = self.encoder_rgb(F_rgb)
            F_ir  = self.encoder_ir(F_ir)
            
            # 2. 准备查询
            detect_queries = self.mtuq.init_detect_queries()
            if track_queries is not None:
                all_queries = concat_queries(track_queries, detect_queries)
            else:
                all_queries = detect_queries
            
            # 3. 模态感知解码
            history = stack_history(history_buffer) if history_buffer else None
            for layer in self.decoder_layers:
                all_queries = layer(all_queries, F_rgb, F_ir, history)
            
            # 4. 预测
            cls_pred = self.cls_head(all_queries['q_fused'])
            box_pred = self.box_head(all_queries['q_fused'])
            
            # 也从 q_rgb 和 q_ir 做辅助预测（用于 CMC Loss）
            box_pred_rgb = self.box_head(all_queries['q_rgb'])
            box_pred_ir  = self.box_head(all_queries['q_ir'])
            
            results.append({
                'cls': cls_pred, 'box': box_pred,
                'box_rgb': box_pred_rgb, 'box_ir': box_pred_ir,
                'queries': all_queries,
            })
            
            # 5. 更新跟踪查询和历史缓冲
            track_queries = self.mtuq.create_track_queries(all_queries)
            history_buffer.append(all_queries['q_fused'].detach())
            if len(history_buffer) > self.memory_len:
                history_buffer.pop(0)
        
        return results
```

### Step 2.5 — 验证
- 仅用 IR 模态跑通模型（rgb 输入设为 ir 的副本），确认流程正确
- 加入 RGB 模态，对比单模态 vs 双模态
- 可视化门控权重 `gate_rgb / gate_ir / gate_mot`，确认行为合理
- 对比：标准 MOTR 解码器 vs MAD 解码器（均使用双模态输入）

---

## 阶段三：跨模态一致性损失 CMC（预计 1-2 周）

> 全新创新点，审稿区分度最高。

### Step 3.1 — 实现位置/分类一致性损失

```python
# 文件: models/loss/cmc_loss.py

class CrossModalConsistencyLoss(nn.Module):
    def forward(self, results, assignments):
        L_pos = 0  # 位置一致性
        L_cls = 0  # 分类一致性
        count = 0
        
        for frame_result, assignment in zip(results, assignments):
            # 只对匹配到 GT 的查询计算一致性
            matched_idx = assignment['matched_query_indices']
            
            box_rgb = frame_result['box_rgb'][matched_idx]
            box_ir  = frame_result['box_ir'][matched_idx]
            
            # 位置一致性：两个视图对同一目标的预测应一致
            L_pos += F.l1_loss(box_rgb, box_ir)
            
            # 分类一致性：两个视图的分类概率应一致
            cls_rgb = self.cls_head(frame_result['queries']['q_rgb'][matched_idx])
            cls_ir  = self.cls_head(frame_result['queries']['q_ir'][matched_idx])
            L_cls += F.kl_div(
                F.log_softmax(cls_rgb, dim=-1),
                F.softmax(cls_ir, dim=-1),
                reduction='batchmean'
            )
            count += 1
        
        return (L_pos + L_cls) / max(count, 1)
```

### Step 3.2 — 实现跨模态对比学习损失

```python
# 文件: models/loss/cmc_loss.py (续)

class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        self.temp = temperature
    
    def forward(self, q_rgb, q_ir, matched_mask):
        """
        q_rgb: [N, d] 所有查询的 RGB 视图特征
        q_ir:  [N, d] 所有查询的 IR 视图特征
        matched_mask: [N] 布尔掩码，标记哪些查询匹配到了 GT
        """
        # 只使用匹配到 GT 的查询
        q_rgb = F.normalize(q_rgb[matched_mask], dim=-1)
        q_ir  = F.normalize(q_ir[matched_mask], dim=-1)
        M = q_rgb.shape[0]
        
        if M < 2:
            return torch.tensor(0.0)
        
        # 正样本对：同一目标的 (q_rgb_i, q_ir_i)
        # 负样本对：不同目标的 (q_rgb_i, q_ir_j), j ≠ i
        sim_matrix = torch.mm(q_rgb, q_ir.T) / self.temp  # [M, M]
        labels = torch.arange(M).to(sim_matrix.device)
        
        loss = (F.cross_entropy(sim_matrix, labels) + 
                F.cross_entropy(sim_matrix.T, labels)) / 2
        return loss
```

### Step 3.3 — 集成到总损失

```python
L_total = (
    λ_cls * FocalLoss + λ_nwd * NWDLoss + λ_l1 * L1Loss +  # 基础损失
    λ_con * L_consistency +                                   # 一致性损失
    λ_ctr * L_contrastive                                     # 对比学习损失
)
# 建议初始值：λ_con = 1.0, λ_ctr = 0.5
```

### Step 3.4 — 验证
- 消融实验：无 CMC vs 仅位置一致性 vs 仅对比学习 vs 两者都有
- **重要**：模态退化实验
  - 在测试时对 RGB 图像加入不同程度的高斯噪声 σ ∈ {0, 25, 50, 100, 255}
  - 观察 HOTA 的下降曲线
  - 对比有/无 CMC Loss 时的鲁棒性差异

---

## 阶段四：尺度自适应采样 SAS（预计 1-2 周）

### Step 4.1 — 修改可变形注意力

```python
# 文件: models/decoder/scale_adaptive_sampling.py

class ScaleAdaptiveDeformableAttn(DeformableAttn):
    """在标准可变形注意力基础上，增加尺度自适应约束"""
    
    def __init__(self, d_model, n_heads, n_points, max_offset=0.5):
        super().__init__(d_model, n_heads, n_points)
        self.scale_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        self.max_offset = max_offset
    
    def forward(self, query, reference_points, input_flatten, ...):
        # 每个查询预测自己的尺度参数
        scale_param = self.scale_predictor(query)  # [B, N, 1]
        
        # 计算采样偏移（原始方式）
        raw_offset = self.sampling_offsets(query)  # [B, N, n_heads, n_points, 2]
        
        # 尺度约束：用 tanh 限界 + 尺度缩放
        constrained_offset = (
            torch.tanh(raw_offset) * 
            scale_param.unsqueeze(-1).unsqueeze(-1) * 
            self.max_offset
        )
        
        # 后续使用 constrained_offset 而非 raw_offset
        ...
        return output, scale_param  # 同时返回 scale_param 用于损失加权
```

### Step 4.2 — 尺度感知损失加权

```python
# 在损失计算中，利用 scale_param 加权
scale_weight = 1.0 / (scale_param.squeeze(-1) + 0.01)
# 小目标对应小 scale_param → 大权重 → 梯度增强

L_nwd_weighted = (scale_weight * NWDLoss(pred, gt)).mean()
```

### Step 4.3 — 在 MAD 解码器中替换
- 将 MAD 中的 `self.cross_attn_rgb` 和 `self.cross_attn_ir` 替换为 `ScaleAdaptiveDeformableAttn`
- 确保 scale_param 在前向传播中被正确返回和使用

### Step 4.4 — 验证
- 按目标尺度分组统计检测精度
- 可视化不同尺寸目标上的采样点分布
- 对比：无约束 vs 尺度自适应约束

---

## 阶段五：运动视图完善（预计 1-2 周）

### Step 5.1 — 丰富运动视图的信息来源

在 Step 2.3 的 MAD 中，运动视图 `q_motion` 目前仅通过对历史查询做注意力来更新。增加显式的位置编码：

```python
class MotionViewUpdater(nn.Module):
    def __init__(self, d_model=256, max_history=5):
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, d_model // 2),  # 4 = (cx, cy, w, h)
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.temporal_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=512),
            num_layers=2
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, q_motion, history_queries, history_boxes):
        """
        q_motion: [N, d] 当前运动视图
        history_queries: [K, N, d] 过去K帧的 q_fused
        history_boxes: [K, N, 4] 过去K帧的预测框
        """
        if history_queries is None or len(history_queries) == 0:
            return q_motion
        
        K, N, d = history_queries.shape
        
        # 为历史查询加上位置编码
        pos_embed = self.pos_encoder(history_boxes)  # [K, N, d]
        history_input = history_queries + pos_embed   # [K, N, d]
        
        # 每个目标独立地对自己的历史做时序编码
        # 重塑为 [K, N, d] → 按N拆分为N个 [K, d] 序列
        motion_tokens = []
        for i in range(N):
            seq = history_input[:, i, :].unsqueeze(1)  # [K, 1, d]
            encoded = self.temporal_attn(seq)           # [K, 1, d]
            motion_tokens.append(encoded[-1, 0, :])     # 取最后时刻 [d]
        
        motion_token = torch.stack(motion_tokens, dim=0)  # [N, d]
        
        # 门控更新
        g = self.gate(torch.cat([q_motion, motion_token], dim=-1))
        q_motion_updated = q_motion + g * motion_token
        
        return q_motion_updated
```

### Step 5.2 — 历史缓冲区管理优化
- 存储最近 K 帧的 `(q_fused, pred_boxes)`
- 训练时 K 由视频片段长度决定（2→3→4→5 递增）
- 推理时 K 固定（建议 K=5）
- 目标消失时清除对应的历史记录

### Step 5.3 — 验证
- 关注 IDF1 和 IDS 指标的变化
- 对比有/无运动视图
- 长序列（>100帧）上的轨迹完整性分析

---

## 阶段六：系统集成、调优与完整实验（预计 3 周）

### Step 6.1 — 全系统集成
```
完整数据流：
(img_rgb, img_ir) → DualStreamBackbone → (F_rgb, F_ir)
                                              ↓
                              DualStreamEncoder → (F_rgb_enc, F_ir_enc)
                                              ↓
                     MTUQuery.init/create → {q_rgb, q_ir, q_motion, q_fused}
                                              ↓
                 MotionViewUpdater(history) → q_motion updated
                                              ↓
                     MAD Decoder (6 layers) → updated queries
                                              ↓
                         Prediction Heads → cls, box, box_rgb, box_ir
                                              ↓
                              Loss: NWD + Focal + L1 + CMC_consistency + CMC_contrastive
                                              ↓
                         TALA (NWD matching) → label assignment
```

### Step 6.2 — 分阶段训练策略

| 阶段 | Epoch | 操作 | 学习率 | 片段长度 |
|------|-------|------|--------|---------|
| Warm-up | 1-30 | 冻结双流骨干，训练解码器+损失 | 2e-4 | 2 |
| Fine-tune | 31-80 | 解冻骨干最后2个stage | 2e-5 (骨干) / 2e-4 (其他) | 3 |
| Full train | 81-150 | 全参数训练 | 2e-4 | 4 |
| Final | 151-200 | 学习率×0.1 | 2e-5 | 5 |

- 跟踪查询 drop 概率 p_drop = 0.1
- 假阳性查询 insert 概率 p_insert = 0.1
- 数据增强：随机翻转、随机裁剪、颜色抖动（仅 RGB）

### Step 6.3 — 完整消融实验

**主消融表（Table 2 in paper）**：

| # | MTUQ/MAD | CMC Loss | SAS | Motion View | NWD | HOTA | MOTA | IDF1 | IDS |
|---|----------|----------|-----|-------------|-----|------|------|------|-----|
| 1 | - (标准解码器) | - | - | - | - | baseline | | | |
| 2 | - | - | - | - | ✓ | | | | |
| 3 | ✓ | - | - | - | - | | | | |
| 4 | ✓ | - | - | - | ✓ | | | | |
| 5 | ✓ | ✓ | - | - | ✓ | | | | |
| 6 | ✓ | ✓ | ✓ | - | ✓ | | | | |
| 7 | ✓ | ✓ | ✓ | ✓ | ✓ | **full** | | | |

**融合层级对比表（Table 3）**：

| 融合策略 | 位置 | HOTA | MOTA | IDF1 |
|---------|------|------|------|------|
| 无融合（仅 IR） | - | | | |
| 输入级拼接 | 骨干前 | | | |
| 特征级交叉注意力 | 编码器后 | | | |
| **查询级融合（MTUQ）** | 解码器内 | | | |

**模态退化鲁棒性表（Table 4）**：

| 方法 | 正常 | RGB σ=50 | RGB σ=100 | RGB 全黑 | IR σ=50 |
|------|------|----------|-----------|---------|---------|
| 仅 IR | - | N/A | N/A | N/A | - |
| 仅 RGB | - | - | - | - | N/A |
| 特征级融合 | - | - | - | - | - |
| DSITT (无CMC) | - | - | - | - | - |
| **DSITT (完整)** | - | - | - | - | - |

**目的**：展示 CMC 一致性约束在模态退化时的鲁棒性优势。

### Step 6.4 — SOTA 对比实验

**主结果表（Table 1 in paper）**：

| 方法 | 类型 | HOTA↑ | MOTA↑ | IDF1↑ | IDS↓ | DetA↑ | AssA↑ | SAFit↑ |
|------|------|-------|-------|-------|------|-------|-------|--------|
| SORT | 检测-跟踪 | - | - | - | - | - | - | - |
| DeepSORT | 检测-跟踪 | - | - | - | - | - | - | - |
| ByteTrack | 检测-跟踪 | - | - | - | - | - | - | - |
| OC-SORT | 检测-跟踪 | - | - | - | - | - | - | - |
| BoT-SORT | 检测-跟踪 | - | - | - | - | - | - | - |
| MOTR | 端到端 | - | - | - | - | - | - | - |
| MOTRv2 | 端到端 | - | - | - | - | - | - | - |
| MeMOTR | 端到端 | - | - | - | - | - | - | - |
| **DSITT (Ours)** | 端到端 | **-** | **-** | **-** | **-** | **-** | **-** | **-** |

### Step 6.5 — 效率分析

| 方法 | Params (M) | FLOPs (G) | FPS | HOTA |
|------|-----------|-----------|-----|------|
| ByteTrack | - | - | - | - |
| MOTR | - | - | - | - |
| MeMOTR | - | - | - | - |
| **DSITT (Ours)** | - | - | - | - |

### Step 6.6 — 可视化结果
1. 查询多视图注意力图（q_rgb / q_ir / q_fused 分别的注意力区域）
2. 模态门控权重随时间变化曲线（展示白天→夜晚的自动切换）
3. 尺度自适应采样点分布（小目标 vs 正常目标）
4. 跟踪轨迹对比（MOTR vs DSITT，展示 ID Switch 减少）
5. 失败案例分析（诚实展示局限性）

---

## 各阶段依赖关系与时间线总览

```
Week 1-2:   阶段零 [环境+基线]  ████████████
                │
Week 3:     阶段一 [NWD Loss]   ████
                │
Week 4-6:   阶段二 [MTUQ+MAD]  ████████████████  ← 核心！
                │
Week 7-8:   阶段三 [CMC Loss]   ████████
            阶段四 [SAS]         ████████  (可与阶段三并行)
                │
Week 9:     阶段五 [运动视图]    ████████
                │
Week 10-12: 阶段六 [集成+实验]  ████████████████

总计：约 12 周
```

---

## 关键风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| RGBT-Tiny 不提供配对 RGB 图像 | 低 | 致命 | 提前确认；备选方案：用合成 RGB |
| 双流骨干显存不足 | 中 | 高 | 共享骨干前3个stage，只在后2个stage独立 |
| 查询级融合效果不如特征级 | 中 | 高 | 两者结合（特征级+查询级）作为备选 |
| CMC Loss 导致训练不稳定 | 中 | 中 | 训练前期用小权重(0.1)，后期增大(1.0) |
| 单数据集审稿被质疑 | 高 | 中 | 准备 KAIST 上的检测实验作为补充 |

---

## 修订后的项目文件结构

```
DSITT/
├── configs/
│   ├── dsitt_base.yaml            # 基线配置（MOTR on RGBT-Tiny）
│   ├── dsitt_nwd.yaml             # + NWD Loss
│   ├── dsitt_mtuq.yaml            # + MTUQ + MAD
│   ├── dsitt_cmc.yaml             # + CMC Loss
│   └── dsitt_full.yaml            # 完整模型
├── models/
│   ├── backbone/
│   │   ├── resnet.py
│   │   ├── fpn.py
│   │   └── dual_stream.py         # 双流骨干 [新]
│   ├── encoder/
│   │   └── deformable_encoder.py
│   ├── decoder/
│   │   ├── modality_aware_decoder.py  # MAD 解码器 [核心]
│   │   └── scale_adaptive_attn.py     # SAS 注意力 [新]
│   ├── tracking/
│   │   ├── mtuq.py                    # MTUQ 查询管理 [核心]
│   │   ├── motion_view.py             # 运动视图更新 [新]
│   │   ├── query_interaction.py       # QIM (from MOTR)
│   │   └── tala.py                    # TALA (adapted with NWD)
│   ├── loss/
│   │   ├── nwd_loss.py                # NWD 损失 [新]
│   │   ├── cmc_loss.py                # CMC 一致性+对比损失 [核心]
│   │   ├── matcher.py                 # NWD 匹配代价 [修改]
│   │   └── collective_loss.py         # CAL (from MOTR)
│   └── dsitt.py                       # 主模型 [核心]
├── datasets/
│   ├── rgbt_tiny.py                   # 双模态数据加载 [新]
│   └── transforms.py
├── engine/
│   ├── train.py
│   └── evaluate.py
├── tools/
│   ├── train_net.py
│   ├── eval_net.py
│   └── visualize.py
├── analysis/
│   ├── paper_analysis.md              # 初始分析
│   ├── implementation_roadmap.md      # V1 路线图
│   ├── second_round_review.md         # 二次审视
│   └── implementation_roadmap_v2.md   # V2 路线图（本文档）
└── README.md
```
