# DSITT/QFTrack 项目技术文档 — 初期阶段报告

## 1 完成目标

### 1.1 任务概述

本项目旨在设计并实现一个面向 RGBT 微小目标多目标跟踪（MOT）的端到端深度学习框架 DSITT（论文名 QFTrack），核心创新为查询级跨模态融合机制。具体任务分解如下：

1. **框架设计与代码实现**：基于 Deformable DETR 架构，实现包含双流骨干网络、可变形编码器、模态感知解码器（MAD）、多视图跟踪查询管理器（MTUQ）等核心模块的完整检测-跟踪框架。
2. **损失函数体系构建**：实现 Focal Loss、L1 Loss、归一化 Wasserstein 距离（NWD）、跨模态一致性损失（CMC）、尺度自适应注意力多样性正则（SAS diversity）及辅助解码损失等多损失联合优化体系。
3. **数据集适配与预处理**：针对 RGBT-Tiny 数据集（115 序列，640×512 分辨率，7 类目标，COCO 格式标注），完成数据加载、双模态同步增强、train/test 划分验证等工作。
4. **训练与评估流水线**：实现支持 AMP 混合精度、梯度累积、学习率 warmup、checkpoint resume、best model 追踪的训练脚本，以及支持 HOTA/MOTA/IDF1 等指标的评估脚本。
5. **消融实验与论文撰写**：完成 v1 基线、v1+NWD、v2 完整模型、消融实验等多组对比实验，将结果填入 CVPR 2026 格式论文。

### 1.2 复杂工程问题

在项目实施过程中，遇到并解决了以下复杂工程问题：

1. **多模态特征对齐与融合**：RGB 和红外图像在特征空间中存在显著的模态差异（分辨率相同但信息互补），如何在查询层面而非特征层面进行有效融合，避免信息丢失，是核心技术难题。本项目通过 MTUQ 四视图查询机制（q_rgb, q_ir, q_motion, q_fused）和 MAD 解码器中的跨模态交换与门控融合实现。

2. **微小目标检测的损失函数设计**：RGBT-Tiny 数据集中目标最小仅 4×4 像素，传统 IoU 类损失对微小目标极不敏感。本项目引入归一化 Wasserstein 距离（NWD）替代 GIoU，将边界框建模为二维高斯分布计算分布距离，显著提升微小目标的回归精度。

3. **多尺度可变形注意力的显存优化**：4 尺度特征图（stride 4/8/16/32）在 640×512 输入下产生大量采样点，纯 PyTorch 实现的可变形注意力显存开销巨大。双流模型（81.7M 参数）在 AMP 模式下仍需约 24GB 显存，需要合理的 clip_length 控制和 GPU 资源分配策略。

4. **跟踪查询的时序一致性维护**：在多目标跟踪中，跟踪查询需要跨帧维护身份一致性。QIM（Query Interaction Module）的筛选过程会改变查询索引，导致 TALA（Track-Aware Label Assignment）的 track_assignment 映射错位。本项目通过 old_to_new 重映射机制解决。

5. **训练稳定性与收敛性**：多损失联合训练（7 种损失）容易出现梯度冲突和训练不稳定。通过 warmup 策略（1000 iter 线性预热）、梯度裁剪（max_norm=0.1）、backbone 低学习率（0.1×）等手段保障训练稳定收敛。

---

## 2 实施方案与可行性分析

### 2.1 实施方案

本项目采用渐进式开发策略，分为以下阶段：

**阶段一：v1 基线模型实现（单模态）**
- 实现 ResNet-50 + FPN 骨干网络，含 FrozenBatchNorm2d
- 实现 6 层可变形 Transformer 编码器和解码器
- 实现 TrackQueryManager（QIM + TALA）
- 使用 Focal + L1 + GIoU 损失训练单模态 IR 检测-跟踪基线

**阶段二：NWD 损失替换**
- 实现归一化 Wasserstein 距离损失模块
- 在 v1 基线上替换 GIoU 为 NWD，对比微小目标检测性能

**阶段三：v2 双模态完整模型**
- 实现双流骨干网络（DualStreamBackbone）+ 模态 Dropout
- 实现 MTUQ 查询管理器（四视图查询构造）
- 实现 MAD 解码器（跨模态交换 + 门控融合 + SAS 注意力）
- 实现运动视图更新器（轨迹记忆库 + 时序 Transformer）
- 实现 CMC 跨模态一致性损失（KL 散度 + InfoNCE 对比）

**阶段四：实验与论文**
- 在 RGBT-Tiny 数据集上完成 4 组主实验 + 4 组消融实验
- 将实验结果填入 CVPR 2026 格式论文

**技术栈选型**：
- 深度学习框架：PyTorch 2.11 + CUDA 13.0
- 骨干网络：ResNet-50（ImageNet 预训练）
- 注意力机制：纯 PyTorch 实现的多尺度可变形注意力（无需编译 CUDA 算子）
- 训练优化：AdamW + SequentialLR（LinearLR warmup + StepLR decay）+ AMP fp16
- 评估指标：HOTA、MOTA、IDF1、DetA、AssA（匈牙利匹配）

### 2.2 可行性分析

**技术可行性**：
- Deformable DETR 架构已在目标检测领域得到广泛验证，其多尺度可变形注意力机制能有效处理不同尺度的目标。本项目在此基础上扩展为双模态跟踪框架，技术路线成熟可靠。
- NWD 损失已在微小目标检测文献中被证明优于 IoU 类损失，将其引入 MOT 框架具有理论依据。
- 纯 PyTorch 实现的可变形注意力虽然效率低于 CUDA 算子版本，但避免了编译兼容性问题，在单 GPU 训练场景下可接受。

**硬件可行性**：
- 实验环境配备 4 块 GPU（2× RTX 4080 SUPER 32GB + RTX 4060 Ti 16GB + RTX 5060 Ti 16GB），可同时并行运行多组实验。
- v1 单模态模型（40M 参数）需约 12-19GB 显存，v2 双模态模型（81.7M 参数）需约 24GB 显存，均在 32GB GPU 的承载范围内。
- 单个实验 200 epoch 约需 3 天，通过并行训练可将总实验周期控制在 2 周内。

**数据可行性**：
- RGBT-Tiny 数据集包含 115 个序列（85 训练 + 30 测试），46,701 帧 RGB+IR 配对图像，431,201 个 COCO 格式标注（含 tracking_id），7 个目标类别。数据规模适中，标注完整，满足实验需求。

---

## 3 知识技能与开发环境

### 3.1 开发环境和工具

| 类别 | 工具/环境 | 版本/说明 |
|------|-----------|-----------|
| 操作系统 | Linux 6.8 (Ubuntu) | 服务器环境 |
| GPU | NVIDIA RTX 4080 SUPER × 2 | 32GB VRAM，主力训练 |
| Python 环境 | Miniconda + conda env "dsitt" | Python 3.10 |
| 深度学习框架 | PyTorch | 2.11.0+cu130 |
| 计算机视觉 | torchvision | 配套 PyTorch 版本 |
| 科学计算 | NumPy, SciPy | 矩阵运算、匈牙利匹配 |
| 数据处理 | Pillow, PyYAML | 图像读取、配置解析 |
| 可视化 | Matplotlib, TensorBoard | 训练曲线、检测结果可视化 |
| 版本控制 | Git | 代码版本管理 |
| IDE | VS Code (Remote SSH) | 远程开发 |
| 论文排版 | LaTeX (CVPR 2026 模板) | 论文撰写 |

### 3.2 预备知识

完成本项目需要以下预备知识：

- **深度学习基础**：卷积神经网络（CNN）、Transformer 架构、注意力机制、反向传播与梯度优化
- **目标检测**：DETR 系列（Detection Transformer）、Deformable DETR、多尺度特征金字塔（FPN）、Focal Loss
- **多目标跟踪（MOT）**：tracking-by-detection 范式、tracking-by-query 范式、匈牙利匹配、HOTA/MOTA/IDF1 评估指标
- **多模态学习**：RGB-IR 图像特征差异、跨模态特征融合策略（早期融合/晚期融合/查询级融合）
- **PyTorch 工程实践**：混合精度训练（AMP）、梯度累积、学习率调度、分布式训练基础

### 3.3 新知识点学习和掌握情况

在项目实施过程中，学习并掌握了以下新知识：

1. **可变形注意力机制（Deformable Attention）**：学习了多尺度可变形注意力的原理——通过可学习的采样偏移量替代全局注意力，将复杂度从 O(N²) 降至 O(NK)。掌握了纯 PyTorch 实现方式，包括双线性插值采样和多尺度特征索引。

2. **归一化 Wasserstein 距离（NWD）**：学习了将边界框建模为二维高斯分布，通过 Wasserstein 距离度量分布相似性的方法。相比 IoU，NWD 对微小目标更敏感，因为它基于分布距离而非面积重叠。

3. **跨模态一致性学习（CMC）**：学习了通过 KL 散度约束 RGB 和 IR 查询的分类分布一致性，以及通过 InfoNCE 对比损失拉近同一目标在不同模态下的特征表示。

4. **尺度自适应注意力（SAS）**：学习了为每个查询学习独立的尺度参数，约束可变形注意力的采样范围，使不同查询关注不同尺度的特征区域。

5. **训练工程优化**：掌握了 AMP 混合精度训练、梯度累积、FrozenBatchNorm2d、学习率 warmup + step decay 等工程技巧，以及 nohup 后台训练、多 GPU 并行实验管理等实践技能。

### 3.4 参考文献

[1] Zhu X, Su W, Lu L, et al. Deformable DETR: Deformable Transformers for End-to-End Object Detection[C]. ICLR, 2021.

[2] Meinhardt T, Kirillov A, Leal-Taixe L, et al. TrackFormer: Multi-Object Tracking with Transformers[C]. CVPR, 2022.

[3] Xu C, Wang J, Yang W, et al. NWD: A Normalized Wasserstein Distance for Tiny Object Detection[J]. AAAI, 2022.

[4] Zhang L, Zhu X, Chen X, et al. RGBT Tracking via Multi-Modal Mutual Prompt Learning[C]. CVPR, 2024.

[5] Lin T Y, Goyal P, Girshick R, et al. Focal Loss for Dense Object Detection[C]. ICCV, 2017.

[6] Luiten J, Osep A, Dendorfer P, et al. HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking[J]. IJCV, 2021.

---

## 4 任务完成度与后续计划

### 4.1 前期任务完成度（35%）

| 序号 | 任务项 | 状态 | 说明 |
|------|--------|------|------|
| 1 | v1 基线模型代码实现 | ✅ 已完成 | ResNet-50+FPN 骨干、可变形编码器/解码器、QIM+TALA 跟踪管理、Focal+L1+GIoU 损失 |
| 2 | v2 完整模型代码实现 | ✅ 已完成 | 双流骨干、MTUQ 查询管理、MAD 解码器、SAS 注意力、运动视图、CMC 损失 |
| 3 | NWD 损失模块实现 | ✅ 已完成 | 归一化 Wasserstein 距离，支持 GIoU/NWD 切换 |
| 4 | 数据集适配与验证 | ✅ 已完成 | RGBT-Tiny 数据解压、COCO 标注验证、train/test 划分确认 |
| 5 | 训练/评估流水线 | ✅ 已完成 | AMP、梯度累积、warmup、resume、best model 保存、HOTA/MOTA/IDF1 评估 |
| 6 | 代码 Bug 修复 | ✅ 已完成 | 修复 15 个初始缺陷 + 6 个新发现缺陷 + 4 个功能缺失 |
| 7 | 端到端冒烟测试 | ✅ 已完成 | 真实数据 2 epoch 无报错，损失正常收敛 |
| 8 | v1 基线训练（200 epoch） | 🔄 进行中 | dsitt_base.yaml，单模态 IR，cuda:0 运行中 |
| 9 | v2 完整模型训练（200 epoch） | 🔄 进行中 | dsitt_full.yaml，双模态，cuda:1 运行中 |
| 10 | v1+NWD 训练（200 epoch） | ⏳ 待开始 | 等 v1 基线完成后启动 |
| 11 | 消融实验（4 组） | ⏳ 待开始 | 依赖 v2 训练完成 |
| 12 | 论文实验数据填充 | ⏳ 待开始 | 依赖所有训练完成 |
| 13 | 论文图表生成 | ⏳ 待开始 | 架构图、可视化对比、消融曲线 |

已完成工作占比：**约 35%**（代码实现与调试全部完成，数据准备完成，训练流水线验证通过，主实验已启动）

### 4.2 后续实施计划

| 序号 | 工作内容 | 工作开始时间 | 工作结束时间 |
|------|----------|-------------|-------------|
| 1 | v1 基线训练完成，记录 HOTA/MOTA/IDF1 指标 | 2026-04-08 | 2026-04-11 |
| 2 | v2 完整模型训练完成，记录指标 | 2026-04-08 | 2026-04-12 |
| 3 | v1+NWD 训练（200 epoch），对比 GIoU vs NWD | 2026-04-11 | 2026-04-14 |
| 4 | 消融实验 Abl-1：去除 CMC 损失 | 2026-04-14 | 2026-04-16 |
| 5 | 消融实验 Abl-2：去除 SAS 注意力 | 2026-04-14 | 2026-04-16 |
| 6 | 消融实验 Abl-3：去除运动视图 | 2026-04-16 | 2026-04-18 |
| 7 | 消融实验 Abl-4：NWD vs GIoU 对比 | 2026-04-16 | 2026-04-18 |
| 8 | 实验结果汇总，填充论文 Table 1 和 Table 2 | 2026-04-18 | 2026-04-19 |
| 9 | 生成论文图表（架构图、可视化对比、消融曲线等 5 张） | 2026-04-19 | 2026-04-21 |
| 10 | 超参调优（NWD C 值、辅助损失权重） | 2026-04-21 | 2026-04-23 |
| 11 | 补充材料撰写（per-class 分析、失败案例） | 2026-04-23 | 2026-04-24 |
| 12 | 论文终稿审校与提交准备 | 2026-04-24 | 2026-04-26 |