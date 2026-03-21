# DSITT 核心模块实现路线图

## 总览

基于分析报告中识别的四大创新点，本文档给出从零到完整系统的分阶段实现方案。

---

## 阶段零：环境与基线搭建

### Step 0.1 — 复现 Deformable DETR 基线
- 拉取 Deformable DETR 官方仓库（https://github.com/fundamentalvision/Deformable-DETR）
- 在 RGBT-Tiny 数据集上跑通单帧红外目标检测，确认环境可用
- 记录基线指标（AP、AP_small）

### Step 0.2 — 复现 MOTR 基线
- 拉取 MOTR 官方仓库（https://github.com/megvii-research/MOTR）
- 将 MOTR 适配到 RGBT-Tiny 数据集上（单模态红外输入）
- 跑通训练和推理流程，记录 HOTA / MOTA / IDF1 基线指标
- 这就是论文中的 baseline，所有改进都在此基础上叠加

### Step 0.3 — 数据准备
- 下载 RGBT-Tiny 数据集，理解标注格式（bbox + track_id + category）
- 编写 dataloader，确保能同时加载 RGB 帧和对应的红外帧（配对关系）
- 确认训练集 85 个序列、测试集 30 个序列的划分
- 编写评估脚本，集成 HOTA / MOTA / IDF1 / SAFit 指标计算

---

## 阶段一：NWD 增强损失（最先实现，难度最低，收益明显）

### Step 1.1 — 实现 NWD 距离计算
- 将边界框 `(cx, cy, w, h)` 建模为二维高斯分布 `N(μ, Σ)`
  - `μ = (cx, cy)`
  - `Σ = diag(w²/4, h²/4)`（将框的宽高映射为标准差）
- 实现两个高斯分布之间的二阶 Wasserstein 距离：
  ```
  W₂² = ||μ_p - μ_g||² + ||Σ_p^{1/2} - Σ_g^{1/2}||_F²
  ```
- 实现归一化：`NWD = exp(-W₂² / C²)`，C 为归一化常数

### Step 1.2 — 替换损失函数
- 在 MOTR 的损失计算中，将 GIoU Loss 替换为 NWD Loss：
  - `L_nwd = 1 - NWD(pred_box, gt_box)`
- 总损失变为：`L = λ_cls * FocalLoss + λ_nwd * L_nwd + λ_l1 * L1Loss`
- 调整权重系数 `λ_nwd`（建议初始值 2.0，与原 GIoU 权重一致）

### Step 1.3 — 替换匹配代价
- 在 TALA 的匈牙利匹配中，将 IoU 匹配代价替换为 NWD 匹配代价：
  - `C = λ_cls * C_cls + λ_nwd * (1 - NWD) + λ_l1 * C_l1`
- 确保训练时的匹配策略和损失函数使用一致的度量

### Step 1.4 — 验证
- 在 RGBT-Tiny 上训练，对比 baseline（GIoU Loss）和 NWD Loss 的指标
- 重点关注小目标（<8px, 8-16px）上的检测精度提升
- 预期：HOTA 和 MOTA 应有 1-3 个点的提升

---

## 阶段二：跨模态特征桥（核心创新，难度最高，区分度最大）

### Step 2.1 — 构建双流骨干网络
- 复制一份 ResNet-50 + FPN 作为 RGB 分支
- 原有的 ResNet-50 + FPN 作为 IR 分支
- 两个分支不共享权重（独立的参数）
- 数据流：`RGB image → ResNet_rgb → FPN_rgb → {P2_rgb, P3_rgb, P4_rgb, P5_rgb}`
- 数据流：`IR image → ResNet_ir → FPN_ir → {P2_ir, P3_ir, P4_ir, P5_ir}`

### Step 2.2 — 实现模态间可变形交叉注意力
- 对每个特征层级 `l`，实现双向交叉注意力：
  ```
  F_rgb_l' = DeformCrossAttn(query=F_rgb_l, key/value=F_ir_l)
  F_ir_l'  = DeformCrossAttn(query=F_ir_l, key/value=F_rgb_l)
  ```
- 使用 Deformable Attention 的形式（非全局注意力），控制计算量
- 每个查询位置采样 K=4 个点，从另一模态的对应区域聚合信息

### Step 2.3 — 实现自适应模态门控
- 对两个模态的全局特征做池化：
  ```
  f_rgb = GlobalAvgPool(F_rgb)  # [B, C]
  f_ir  = GlobalAvgPool(F_ir)   # [B, C]
  gate  = sigmoid(W_g · concat(f_rgb, f_ir))  # [B, 1]
  ```
- 融合方式：
  ```
  F_fused_l = gate * F_rgb_l' + (1 - gate) * F_ir_l'
  ```
- gate 是标量门控，决定更信任哪个模态（白天倾向 RGB，夜晚倾向 IR）

### Step 2.4 — 尺度感知融合策略
- 高分辨率层（P2, P3）：在融合前增加一层 3×3 卷积 + BN + ReLU 做局部细节增强
- 低分辨率层（P4, P5）：直接使用交叉注意力融合的结果
- 原因：小目标的特征主要存在于高分辨率层，需要更精细的处理

### Step 2.5 — 集成到主流程
- 融合后的多尺度特征 `{F_fused_l}` 送入 Deformable DETR 编码器
- 后续的编码器、解码器、查询机制保持不变
- 训练时两个骨干网络分别用 ImageNet 预训练权重初始化

### Step 2.6 — 验证
- 对比实验：单模态 IR vs 单模态 RGB vs 简单拼接 vs 简单加权平均 vs CMFB
- 分场景分析：白天/夜晚/低光照场景下各方法的性能差异
- 预期：CMFB 在夜晚场景持平、白天和低光照场景显著提升

---

## 阶段三：尺度敏感可变形查询

### Step 3.1 — 实现尺度约束的采样偏移
- 修改 Deformable DETR 解码器中的可变形注意力模块
- 原始：`offset = linear(query)`，偏移量无约束
- 改进：`offset = tanh(linear(query)) * scale_factor`
- `scale_factor` 是根据查询预测的参考点尺度计算的：
  ```
  ref_wh = sigmoid(linear_wh(query))  # 预测参考框的宽高
  scale_factor = ref_wh * base_scale   # base_scale 是超参数
  ```
- 效果：小目标查询的采样点被约束在小范围内，不会采样到目标外部

### Step 3.2 — 实现多粒度查询组
- 将检测查询分为三组（总数不变，如 300 = 100 + 100 + 100）：
  - `Q_tiny`：`base_scale = 0.02`，对应 <8px 目标
  - `Q_small`：`base_scale = 0.05`，对应 8-16px 目标
  - `Q_medium`：`base_scale = 0.10`，对应 16-32px 目标
- 三组查询使用不同的尺度约束，但共享解码器权重
- 预测时根据分类置信度选择最终结果

### Step 3.3 — 对比度引导的参考点初始化（可选增强）
- 对红外输入计算局部对比度图：
  ```
  contrast_map = |I - GaussianBlur(I, kernel=15)|
  ```
- 从对比度图中提取 top-K 位置作为参考点的初始值
- 替代原始的均匀初始化或纯学习初始化

### Step 3.4 — 验证
- 消融实验：标准偏移 vs 尺度约束偏移 vs 多粒度查询
- 按目标尺度分组统计检测精度
- 可视化采样点分布：展示小目标上的采样点是否集中在目标区域内
- 预期：极小目标（<8px）检测精度提升明显

---

## 阶段四：时序运动感知模块

### Step 4.1 — 构建轨迹记忆库
- 为每个活跃的跟踪查询维护一个固定长度 K 的记忆队列：
  ```
  memory_i = deque(maxlen=K)  # K=5 或 K=10
  每帧结束后：memory_i.append((query_feature_i, pred_box_i))
  ```
- 记忆库存储在 GPU 上，推理时随帧更新

### Step 4.2 — 实现运动模式编码器
- 输入：轨迹记忆库中最近 K 帧的预测框序列 `[(cx,cy,w,h)_t-K, ..., (cx,cy,w,h)_t-1]`
- 编码器结构：2 层 Transformer Encoder（轻量级，hidden_dim=256）
  ```
  position_sequence = concat([box_t-K, ..., box_t-1])  # [K, 4]
  position_embed = MLP(position_sequence)  # [K, 256]
  feature_sequence = stack([q_t-K, ..., q_t-1])  # [K, 256]
  input = feature_sequence + position_embed
  motion_token = TransformerEncoder(input)[-1]  # 取最后一个输出 [256]
  ```
- motion_token 编码了历史运动模式

### Step 4.3 — 运动引导的查询更新
- 在每帧开始时，更新跟踪查询：
  ```
  gate = sigmoid(W_gate · concat(q_track, motion_token))
  q_track_updated = q_track + gate * W_motion · motion_token
  ```
- gate 控制运动信息的注入量（对于匀速运动注入多，对于静止目标注入少）
- 更新后的 q_track_updated 送入 Deformable 解码器

### Step 4.4 — 跨帧特征聚合（可选增强）
- 维护前 K 帧的编码器特征图（内存允许的情况下 K=2-3）
- 在解码器的交叉注意力中，查询不仅 attend 当前帧特征，也 attend 前几帧特征
- 实现方式：将多帧特征图拼接在 key/value 序列维度上
  ```
  keys = concat([F_t-2, F_t-1, F_t], dim=sequence)
  values = concat([F_t-2, F_t-1, F_t], dim=sequence)
  output = DeformAttn(query=Q, key=keys, value=values)
  ```
- 采样点偏移基于运动预测进行调整

### Step 4.5 — 验证
- 消融实验：无时序增强 vs 仅记忆库 vs 记忆库+运动编码器 vs 完整 TMAM
- 关注 IDF1 和 IDS 指标（衡量跟踪连续性）
- 长序列跟踪的轨迹可视化对比
- 预期：IDF1 提升 2-5 个点，IDS 下降

---

## 阶段五：系统集成与调优

### Step 5.1 — 全模块集成
- 将四个模块按以下顺序集成到统一框架中：
  ```
  双流骨干 → CMFB → Deformable 编码器 → SSDQ 初始化 →
  TMAM 更新跟踪查询 → Deformable 解码器 → 预测头 → NWD Loss
  ```
- 确保梯度能正常反向传播到所有模块

### Step 5.2 — 训练策略
- **第一阶段（Epoch 1-50）**：冻结双流骨干，只训练 CMFB + 编码器 + 解码器
  - 视频片段长度=2，学习率=2e-4
- **第二阶段（Epoch 51-100）**：解冻骨干最后两个 stage，端到端微调
  - 视频片段长度=3，学习率=2e-5
- **第三阶段（Epoch 101-200）**：全参数训练
  - 视频片段长度逐步增加到 5
  - Epoch 150 时学习率降低 10 倍
- 跟踪查询的 drop（p_drop=0.1）和 insert（p_insert=0.1）策略保持

### Step 5.3 — 超参数搜索
- 重点调优的超参数：
  - NWD 归一化常数 C（建议搜索 {2, 4, 8}）
  - 损失权重 λ_nwd（建议搜索 {1.0, 2.0, 5.0}）
  - 记忆库长度 K（建议搜索 {3, 5, 10}）
  - 多粒度查询的分配比例
  - 门控融合的温度参数

### Step 5.4 — 最终评估
- 在 RGBT-Tiny 测试集上评估完整模型
- 运行完整的消融实验矩阵（见分析报告表格）
- 生成可视化结果（注意力热力图、轨迹图、门控权重图）
- 计算推理速度（FPS）和模型参数量

---

## 各阶段预计时间与依赖关系

```
时间线（预估）：

阶段零 [环境搭建]     ████████  （1-2 周）
  │
  ├── 阶段一 [NWD Loss]  ████  （3-5 天）
  │     │
  │     └── 阶段二 [CMFB]   ████████████  （2-3 周）
  │           │
  │           ├── 阶段三 [SSDQ]  ████████  （1-2 周）
  │           │
  │           └── 阶段四 [TMAM]  ████████  （1-2 周）
  │                 │
  └─────────────────┴── 阶段五 [集成调优]  ████████████  （2-3 周）

总计预估：8-12 周
```

**依赖关系**：
- 阶段一（NWD）独立于其他模块，可最先完成
- 阶段二（CMFB）是核心，需要优先保证质量
- 阶段三（SSDQ）和阶段四（TMAM）可并行开发
- 阶段五（集成）依赖前四个阶段全部完成

---

## 关键检查点

| 检查点 | 条件 | 若不满足的应对 |
|--------|------|--------------|
| 阶段零完成 | MOTR 在 RGBT-Tiny 上跑通，基线指标合理 | 检查数据加载和评估脚本 |
| 阶段一完成 | NWD Loss 带来小目标检测精度提升 | 检查 NWD 实现正确性，调整 C 和 λ |
| 阶段二完成 | 双模态融合优于任意单模态 | 检查门控机制是否正常工作，可视化门控值 |
| 阶段三完成 | 极小目标检测精度有提升 | 可视化采样点确认约束生效 |
| 阶段四完成 | IDF1 提升，IDS 下降 | 检查记忆库更新逻辑，可能需要增大 K |
| 阶段五完成 | 全模块集成指标优于所有单模块 | 若冲突则逐步排查模块间干扰 |

---

## 文件结构建议

```
DSITT/
├── configs/                    # 配置文件
│   ├── dsitt_base.yaml        # 基础配置
│   ├── dsitt_nwd.yaml         # NWD Loss 实验
│   ├── dsitt_cmfb.yaml        # CMFB 实验
│   └── dsitt_full.yaml        # 完整模型
├── models/
│   ├── backbone/
│   │   ├── resnet.py          # ResNet-50 骨干
│   │   └── fpn.py             # 特征金字塔
│   ├── encoder/
│   │   └── deformable_encoder.py
│   ├── decoder/
│   │   ├── deformable_decoder.py
│   │   └── scale_sensitive_attn.py  # SSDQ 模块
│   ├── fusion/
│   │   ├── cross_modal_bridge.py    # CMFB 模块
│   │   └── adaptive_gate.py         # 门控机制
│   ├── temporal/
│   │   ├── memory_bank.py           # 轨迹记忆库
│   │   ├── motion_encoder.py        # 运动模式编码器
│   │   └── tmam.py                  # TMAM 完整模块
│   ├── tracking/
│   │   ├── track_query.py           # 跟踪查询管理
│   │   ├── query_interaction.py     # QIM
│   │   └── tala.py                  # 轨迹感知标签分配
│   ├── loss/
│   │   ├── nwd_loss.py              # NWD 损失
│   │   ├── matcher.py               # NWD 匹配代价
│   │   └── collective_loss.py       # 集体平均损失
│   └── dsitt.py                     # 主模型入口
├── datasets/
│   ├── rgbt_tiny.py           # RGBT-Tiny 数据加载
│   └── transforms.py          # 数据增强
├── engine/
│   ├── train.py               # 训练逻辑
│   └── evaluate.py            # 评估逻辑
├── tools/
│   ├── train_net.py           # 训练入口
│   ├── eval_net.py            # 评估入口
│   └── visualize.py           # 可视化工具
├── analysis/                  # 分析文档
│   ├── paper_analysis.md
│   └── implementation_roadmap.md
└── README.md