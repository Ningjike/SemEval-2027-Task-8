# StereoQueerEval：实验记录

## 数据与划分

使用 EN、IT、NL 共 7,627 条样本，按语言、原始视频标题和描述构造上下文分组，再按约 **8∶2** 划分训练集与验证集。当前训练和评估均使用 `grouped_multilingual`。

| 数据 | 样本数 | 用途 |
| --- | ---: | --- |
| 训练集 | 6,106 | 训练 A/B；C 使用其中的仇恨样本 |
| 验证集 | 1,521 | 每轮验证、选择最佳 checkpoint，以及训练后的评估 |
| 验证集中的仇恨样本 | 790 | C 的对象识别评估 |
| FA 开发集 | 100 | 单独保留，未用于当前训练和结果汇报 |

**训练后的评估仍使用上述验证集，没有另外划分测试集。** 数据准备当前启用 `CLEAN_DESCRIPTION=True`，清理视频描述中的推广行；清洗后为空时保留原文。

## 模型与任务

编码器为 `FacebookAI/xlm-roberta-base`，输入按照 `[COMMENT] 评论 [TITLE] 标题 [DESCRIPTION] 描述` 拼接，截断至 512 tokens。编码器输出经均值池化、dropout 和任务分类头得到预测。A/B/C 分别训练模型。

| 任务 | 预测目标 | 实现 |
| --- | --- | --- |
| A | 刻板印象 | `no / yes` 二分类 |
| B | 仇恨言论 | `no / yes_implicit / yes_explicit` 三分类 |
| C | 仇恨对象 | `group / individual` 二分类 + 9 个身份的多标签预测 |

C 的身份标签为 `l, g, b, t, q, i, a, nb, lgbtqia+`，同一样本可有多个身份。身份预测阈值为 0.5；若均未达到阈值，选择概率最高的身份。无标签数据推理时先由同版本 B 模型判断是否为仇恨，非仇恨样本直接输出 `none`。

## 方法与训练配置

- **v1：批内 SCL**，在当前 micro-batch 内构造同标签正例。
- **v2：MoCo-SCL**，使用动量模型和带标签的历史队列扩展对比候选。

A/B 的监督损失为交叉熵，C 为范围交叉熵与身份二元交叉熵之和。总损失为 `L_supervised + 0.1 × L_SCL`。A/B 按类别标签匹配正例，C 按范围与完整身份组合匹配正例。

以下为当前代码配置，每个版本内 A/B/C 一致：

| 参数 | v1 | v2 |
| --- | ---: | ---: |
| epochs | 8 | 3 |
| batch size | 8 | 4 |
| gradient accumulation | 2 | 4 |
| 有效 batch size | 16 | 16 |
| eval batch size | 8 | 8 |
| learning rate | 2e-5 | 2e-5 |
| weight decay | 0.03 | 0.01 |
| SCL weight | 0.1 | 0.1 |
| temperature | 0.1 | 0.1 |
| queue size | — | 4096 |
| momentum | — | 0.999 |

共同设置：seed=42、max length=512、dropout=0.1、projection dim=128、AdamW、warmup ratio=0.1、线性学习率调度、CUDA 混合精度。A/B 按验证 Macro-F1 保存最佳 checkpoint，C 按验证 Target exact match 保存。

## 本地验证结果

以下为新版 v1 加载最佳 checkpoint 后的验证集评估结果，分数范围为 0–1。

| 任务 | 指标 | 分数 | Accuracy | 验证样本数 | 最佳 epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Macro-F1 | 0.721931 | 0.742932 | 1521 | 6 |
| B | Macro-F1 | 0.565439 | 0.598948 | 1521 | 5 |
| C | Target exact match | 0.630380 | — | 790 | 7 |

C 的范围 Macro-F1 为 **0.811704**，身份 Macro-F1 为 **0.348667**。C 在真实仇恨验证样本上直接评估对象识别，不经过 B 模型筛选。

## Codabench 测试状态

尚未进行 Codabench 测试；任务页面开放后提交并补充官方评测结果。

## 脚本入口

- [数据准备](experiments/prepared_data/01_data_preparation.ipynb)：修改 `PROJECT` 后运行；`CLEAN_DESCRIPTION` 位于最后一个代码单元。
- [v1 训练与评估](experiments/contrastive_v1/) · [v2 训练与评估](experiments/contrastive_v2/)：分别运行 `A/B/C.ipynb` 和对应的 `*_evaluation_prediction.ipynb`。
- 训练数据目录为 `/content/drive/MyDrive/prepared_data`；v1、v2 输出目录分别为 `/content/drive/MyDrive/contrastive_outputs`、`/content/drive/MyDrive/contrastive_outputs_v2`。
- [数据来源与使用协议](data/README.md)。仓库仅发布代码与汇总结果，数据集保留在本地。
