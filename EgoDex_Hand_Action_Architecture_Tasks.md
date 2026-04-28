# EgoDex 手部动作预测工具：系统架构与任务文档

## 1. 项目目标

在 EgoDex 数据集上构建一个从第一视角图像或视频预测手部动作的工具链，并对不同方法进行训练、测试和对比分析。

本项目分为三条主线：

1. Baseline：实现 `image / video -> ViT -> MLP -> hand action` 的简单模型，并在 EgoDex 上训练测试。
2. Phantom：阅读、部署 Phantom，并在 EgoDex 上微调测试；由于 Phantom 输出 21 个手部关键点，而 EgoDex 目标为 25 个关键点，需要设计 retarget 流程。
3. Analysis：对比不同方法的预测误差与时序稳定性，包括 MSE、L1、相邻帧抖动等指标。

## 2. 架构原则

- 数据流清晰：原始数据、预处理数据、模型输入、模型输出、评估结果必须分层存储。
- 模型可替换：Baseline 与 Phantom 使用统一的数据读取、输出格式和评估接口。
- 评估可复现：训练配置、checkpoint、预测结果和指标报告需要绑定同一实验 ID。
- Retarget 独立化：21 点到 25 点的映射逻辑独立成模块，避免散落在 Phantom 训练和评估代码中。
- 先跑通闭环：优先完成最小可用训练-推理-评估链路，再扩展更复杂的视频建模和稳定性优化。

## 3. 系统模块划分

### 3.1 数据管理模块

职责：

- 管理 EgoDex 原始视频、帧图像、标注和数据划分。
- 提供统一的数据索引文件，例如 train/val/test split。
- 支持 image 模式和 video clip 模式两种读取方式。

输入：

- EgoDex 原始视频或抽帧结果。
- EgoDex 手部动作/关键点标注。

输出：

- 标准化样本索引。
- 预处理后的图像帧、视频 clip、标签张量。

建议数据结构：

```text
sample_id
video_id
frame_index / clip_start / clip_end
image_path 或 frame_paths
hand_action_label
keypoints_25
metadata
```

### 3.2 预处理模块

职责：

- 视频抽帧。
- 图像 resize、normalize、augmentation。
- 关键点归一化，例如按图像尺寸、手腕中心或相机坐标归一。
- 构建 image-level 与 clip-level 的模型输入。

关键决策：

- Baseline 早期可以先用单帧 image 输入，降低系统复杂度。
- video 模式建议封装为 clip sampler，后续可支持多帧平均、时序 Transformer 或 3D backbone。

### 3.3 Baseline 模型模块

职责：

- 实现简单基线：ViT backbone 提取视觉特征，MLP head 预测 hand action 或 25 点手部表示。
- 支持训练、验证、推理和 checkpoint 保存。

数据流：

```text
image 或 video clip
-> image transform / clip transform
-> ViT encoder
-> feature vector
-> MLP head
-> predicted hand action / keypoints_25
```

设计点：

- ViT 可先使用预训练模型作为 backbone。
- MLP head 的输出维度必须与 EgoDex 标签定义一致。
- 如果 hand action 是连续关键点，则输出为 `25 x D`；如果是动作类别，则输出为类别 logits。当前需求更像连续手部动作预测，应优先按关键点或 pose regression 设计。

### 3.4 Phantom 集成模块

职责：

- 阅读并部署 Phantom。
- 将 EgoDex 数据适配到 Phantom 的输入格式。
- 加载 Phantom 预训练权重并进行 EgoDex 微调。
- 将 Phantom 的 21 点输出 retarget 到 EgoDex 的 25 点格式。

数据流：

```text
EgoDex image / video frame
-> Phantom input adapter
-> Phantom model
-> keypoints_21
-> retarget module
-> keypoints_25
-> evaluator
```

关键风险：

- Phantom 与 EgoDex 的坐标系、手部定义、左右手约定可能不同。
- 21 点到 25 点不一定能纯粹一一映射，可能需要插值、补点或学习式映射。
- 微调时需要确认 Phantom license、依赖版本、输入分辨率和标注格式。

### 3.5 Retarget 模块

职责：

- 维护 Phantom 21 点与 EgoDex 25 点之间的拓扑映射。
- 输出统一的 `keypoints_25`，供评估和可视化使用。
- 支持规则映射与学习式映射两种策略。

建议阶段：

1. 规则映射：已有对应点直接复制，缺失点通过相邻骨骼插值或固定比例外推。
2. 标定映射：用 EgoDex 训练集拟合一个轻量 MLP，将 21 点映射到 25 点。
3. 消融对比：比较规则 retarget 与学习式 retarget 对 MSE/L1 和稳定性的影响。

### 3.6 训练与实验管理模块

职责：

- 管理训练配置、随机种子、数据集版本、模型版本。
- 支持 Baseline 与 Phantom 的统一训练入口。
- 保存 checkpoint、日志、预测结果和指标文件。

建议实验目录：

```text
experiments/
  baseline_vit_mlp/
    config.yaml
    checkpoints/
    predictions/
    metrics.json
  phantom_finetune/
    config.yaml
    checkpoints/
    predictions/
    metrics.json
```

### 3.7 推理模块

职责：

- 对单张图像、视频片段或完整视频生成预测。
- 输出统一预测格式，供评估和可视化复用。

统一预测格式：

```text
sample_id
video_id
frame_index
method
keypoints_25_pred
confidence 可选
runtime_ms 可选
```

### 3.8 评估分析模块

职责：

- 计算预测误差：MSE、L1。
- 计算稳定性：相邻帧关键点差分、速度变化、抖动分数。
- 输出方法对比表和可视化报告。

指标定义：

- MSE：预测关键点与 GT 关键点的均方误差。
- L1：预测关键点与 GT 关键点的平均绝对误差。
- Frame Jitter：相邻帧预测差分与 GT 差分的偏差。
- Temporal Smoothness：预测序列二阶差分的平均幅度，用于衡量不自然抖动。

数据流：

```text
ground truth keypoints_25
+ baseline predictions
+ phantom predictions
-> metric calculator
-> per-frame metrics
-> per-video metrics
-> aggregate report
```

### 3.9 可视化与报告模块

职责：

- 可视化图像/视频帧上的 GT 与预测关键点。
- 绘制不同方法的误差曲线和稳定性曲线。
- 生成最终实验报告，辅助 Part 3 对比分析。

输出：

- 对比表：Baseline vs Phantom。
- 误差曲线：MSE/L1 随训练 epoch 或视频帧变化。
- 稳定性曲线：相邻帧抖动随时间变化。
- 示例视频：GT 与预测骨架 overlay。

## 4. 系统数据流转路径

### 4.1 Baseline 训练路径

```text
EgoDex raw data
-> 数据索引构建
-> 图像/clip 预处理
-> DataLoader
-> ViT + MLP
-> loss 计算
-> checkpoint 保存
-> validation predictions
-> MSE / L1 / stability metrics
```

### 4.2 Phantom 微调路径

```text
EgoDex raw data
-> Phantom input adapter
-> Phantom model fine-tuning
-> keypoints_21 prediction
-> 21-to-25 retarget
-> checkpoint / predictions 保存
-> MSE / L1 / stability metrics
```

### 4.3 对比分析路径

```text
baseline predictions
+ phantom predictions
+ EgoDex ground truth
-> 统一 evaluator
-> 指标聚合
-> 时序稳定性分析
-> 可视化样例生成
-> 实验报告
```

## 5. 任务拆解

### Phase 0：需求澄清与数据验收

- 明确 EgoDex 标签定义：hand action 是类别、关键点、pose 参数，还是多任务标签。
- 确认关键点维度：25 点的拓扑结构、坐标系、单位和左右手定义。
- 确认数据划分策略：官方 split 或自定义 train/val/test。
- 明确 Phantom 的输入输出定义、许可证和依赖要求。

交付物：

- 数据字段说明文档。
- EgoDex 关键点拓扑图或 mapping 表。
- 初版实验配置规范。

### Phase 1：Baseline 闭环

- 建立 EgoDex dataset adapter。
- 建立基础预处理和 DataLoader。
- 实现 ViT + MLP 训练、验证、推理闭环。
- 输出 baseline checkpoint 和 predictions。
- 计算 MSE、L1 基础指标。

交付物：

- Baseline 实验结果。
- Baseline metrics.json。
- 若干预测可视化样例。

### Phase 2：Phantom 部署与适配

- 阅读 Phantom 项目结构与运行方式。
- 跑通 Phantom 官方 demo 或最小推理样例。
- 编写 EgoDex 到 Phantom 的输入适配设计。
- 完成 Phantom 21 点输出解析。

交付物：

- Phantom 部署记录。
- Phantom 输入输出格式说明。
- Phantom 在 EgoDex 样本上的初始预测结果。

### Phase 3：Retarget 与 Phantom 微调

- 建立 21 点到 25 点的规则 mapping。
- 对缺失关键点设计插值/外推策略。
- 在 EgoDex 上微调 Phantom。
- 输出 retarget 后的 `keypoints_25` 预测。

交付物：

- 21-to-25 mapping 表。
- Phantom fine-tune checkpoint。
- Phantom retarget predictions。

### Phase 4：稳定性指标与对比分析

- 实现相邻帧抖动指标。
- 实现二阶差分平滑性指标。
- 对 Baseline 与 Phantom 进行统一评估。
- 生成对比表、曲线和示例视频。

交付物：

- 方法对比报告。
- 误差指标表。
- 稳定性分析图。
- 失败案例分析。

### Phase 5：工程整理与复现实验

- 固化配置文件和实验目录规范。
- 补齐 README、运行说明和结果复现步骤。
- 整理最终实验报告。

交付物：

- 可复现实验文档。
- 最终结果报告。
- 后续优化建议。

## 6. 关键技术决策

### 6.1 标签建模方式

如果 EgoDex 的 hand action 是连续手部状态，建议将主任务定义为关键点回归：

```text
input: image / video clip
output: keypoints_25, shape = [25, D]
loss: MSE + L1
```

如果 EgoDex 同时有动作类别，可作为辅助任务：

```text
loss = regression_loss + classification_loss
```

### 6.2 Image 与 Video 的优先级

建议先做 image baseline，再扩展 video：

- image baseline 更容易验证数据、标签和评估是否正确。
- video 建模可作为提升稳定性的后续方案。
- 稳定性指标即使在单帧模型上也可以先计算，因为推理结果按视频帧排序即可。

### 6.3 Retarget 策略

短期建议使用规则映射，快速让 Phantom 进入统一评估流程。

中期建议用学习式 retarget 做补充，因为 21 到 25 点可能存在非严格拓扑对应，规则补点会影响误差上限。

### 6.4 稳定性指标

稳定性不能只看预测点相邻帧差分大小，否则真实快速动作会被误判为抖动。建议同时计算：

- 预测速度与 GT 速度的差异。
- 预测二阶差分幅度。
- 每个关键点的抖动分布，而不只看全局平均。

## 7. 风险与缓解

| 风险 | 影响 | 缓解策略 |
| --- | --- | --- |
| EgoDex 标签语义不清 | 模型输出设计可能错误 | Phase 0 先验收标签字段和样例 |
| Phantom 坐标系与 EgoDex 不一致 | 指标失真 | 建立坐标归一化和可视化检查 |
| 21 点到 25 点不可直接映射 | Phantom 评估吃亏 | 先规则映射，再探索学习式 retarget |
| 视频帧顺序或采样不一致 | 稳定性指标失真 | 数据索引保留 video_id 与 frame_index |
| Baseline 太弱 | 对比意义不足 | 先作为 sanity check，再加入 video/temporal variant |
| 训练资源不足 | 实验周期变长 | 先冻结 ViT 或用小模型跑通闭环 |

## 8. 建议里程碑

| 里程碑 | 目标 | 验收标准 |
| --- | --- | --- |
| M1 | 数据闭环 | 能读取 EgoDex 样本并显示 GT 关键点 |
| M2 | Baseline 闭环 | ViT + MLP 可训练、验证、保存预测 |
| M3 | Phantom 跑通 | Phantom 可对 EgoDex 样本输出 21 点预测 |
| M4 | Retarget 跑通 | Phantom 输出可转换为 EgoDex 25 点格式 |
| M5 | 统一评估 | Baseline 与 Phantom 共用同一 evaluator |
| M6 | 对比报告 | 产出误差、稳定性和可视化分析 |

## 9. 推荐工程目录

```text
egodex-hand-action/
  configs/
    baseline_vit_mlp.yaml
    phantom_finetune.yaml
  data/
    raw/
    processed/
    splits/
  docs/
    architecture_tasks.md
    keypoint_mapping.md
    experiment_protocol.md
  src/
    datasets/
    preprocessing/
    models/
      baseline/
      phantom_adapter/
    retarget/
    training/
    inference/
    evaluation/
    visualization/
  experiments/
  reports/
  README.md
```

## 10. 下一步讨论清单

1. EgoDex 的 hand action 标签到底是关键点、类别、轨迹，还是多种标签组合？
2. 25 个关键点的定义是否有官方拓扑图？
3. 是否需要同时预测左右手，还是单手？
4. 评估单位使用像素坐标、归一化坐标，还是 3D 坐标？
5. Baseline 是否允许使用 ImageNet 或手部数据预训练 ViT？
6. Phantom 微调是否只调 head，还是全模型 fine-tune？
7. 最终交付更偏实验报告，还是要做成可复用 CLI/工具包？
