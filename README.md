# StereoQueerEval：实验方法与结果记录

本仓库记录 SemEval 2027 StereoQueerEval 的现有实现与实验结果，包含 **v1：批内监督对比学习（SCL）** 和 **v2：MoCo 队列监督对比学习**。每个版本分别训练 A、B、C 三个模型。

**当前阶段：停止继续调参，冻结已有方案，整理结果与复现材料。** 下列数值来自已保存的 Notebook 输出，本次整理没有重新训练或评估模型。整理日期：2026-10-24。

## 1. 任务与数据

输入为 YouTube 评论及对应视频的标题、描述，任务如下：

| 子任务 | 预测目标 | 实现 |
| --- | --- | --- |
| A | 是否包含针对酷儿群体的刻板印象 | `no / yes` 二分类 |
| B | 仇恨言论类型 | `no / yes_implicit / yes_explicit` 三分类 |
| C | 仇恨对象 | `group / individual` 范围分类 + 9 个身份多标签预测 |

C 的身份顺序固定为 `l, g, b, t, q, i, a, nb, lgbtqia+`。非仇恨样本的最终对象为 `none`。

| 语言 | 原始样本数 | 当前训练数 | 当前验证数 |
| --- | ---: | ---: | ---: |
| 英语 EN | 2,989 | 2,395 | 594 |
| 意大利语 IT | 2,400 | 1,912 | 488 |
| 荷兰语 NL | 2,238 | 1,799 | 439 |
| 合计 | 7,627 | 6,106 | 1,521 |
| 波斯语 FA | 100 | 不参与当前训练 | 独立开发集，当前未记录评估结果 |

当前实验使用 `grouped_multilingual` 划分。数据准备程序按“语言 + 原始标题 + 原始描述”的哈希分组，分别在 EN、IT、NL 内选择约 20% 的组作为验证集，再合并三种语言。组是上下文的代理标识，并非直接读取视频 ID。划分使用固定种子 42/43/44，在 2,000 个候选划分中平衡样本量及 A、B、A×B 标签分布，不依赖模型分数。

本地同时保存了单语言 `grouped_EN/IT/NL`、留一语言 `lolo_EN/IT/NL` 和 `persian_external` 划分；存在划分文件不代表完成了相应实验。当前数据准备配置为 `CLEAN_DESCRIPTION=False`，未启用推广行清洗。

数据来源及使用条款见 [data/README.md](data/README.md)。原始数据禁止再分发，本仓库的 `.gitignore` 排除原始数据、包含原文的处理后数据、逐样本划分 ID、生成统计文件及未审查图片。

## 2. 共同模型设计

- 编码器：`FacebookAI/xlm-roberta-base`，全参数微调。
- 输入：`[COMMENT] 评论 [TITLE] 标题 [DESCRIPTION] 描述`，直接拼接后截断到 512 tokens。数据准备 Notebook 中另有分字段预算的长度分析函数，但训练代码未采用该函数。
- 表征：按 attention mask 对最后一层 token 表征做均值池化，再使用 dropout 和线性分类头。
- 对比学习投影头：`Linear → GELU → Linear`，输出维度 128。
- A/B 监督损失为交叉熵；C 为范围交叉熵与身份 BCE-with-logits 之和。
- 总损失：`L = L_supervised + λ × L_SCL`。
- C 仅在真实仇恨样本上训练、验证；身份 sigmoid 阈值为 0.5，如果没有身份过阈值，则保留概率最高的一个。
- A/B 根据验证 Macro-F1 保存最佳模型；C 根据验证 target exact match 保存最佳模型。

## 3. 两个方案

### v1：批内 SCL

仅在当前 micro-batch 内构造对比关系，排除自身。A/B 中标签相同的样本为正例；C 要求范围和全部身份标签完全一致。没有正例的 anchor 不参与 SCL，整个批次没有有效 anchor 时返回零对比损失。

梯度累积扩大优化器的有效 batch，但不会扩大批内 SCL 的候选集合。因此，小批次和稀有 C 标签可能限制可用正例。该版本显式关闭对比损失区域的 autocast，以 FP32 计算相似度与 log-sum-exp。

### v2：MoCo 队列 SCL

保留相同的任务模型，增加动量 key 模型与带标签的 FIFO 队列。候选包括当前批次的 key 和历史队列；key 模型不反向传播，参数通过 EMA 更新。队列容量为 4,096，动量系数为 0.999。

query 与 key 使用同一文本，通过独立 dropout 构造两个视图；当前样本的配对 key 始终作为正例。历史队列中同一样本的旧视图被排除，其余正例仍使用与 v1 一致的标签匹配规则。推理和保存的模型权重只需要 query 模型，不需要队列。

该方案旨在增加小批次下的对比候选与正例覆盖；目前结果不支持“所有任务均受益”的结论。

## 4. 已保存配置

以下是当前 Notebook 源码中的配置。由于本地未包含运行时 checkpoint 配置，它们不能替代原始运行目录中的 `config.json`。

| 参数 | v1-A | v1-B / v1-C | v2-A / v2-B / v2-C |
| --- | ---: | ---: | ---: |
| epochs | 8 | 3 | 3 |
| micro-batch size | 8 | 4 | 4 |
| gradient accumulation | 2 | 4 | 4 |
| 名义有效 batch size | 16 | 16 | 16 |
| learning rate | 2e-5 | 2e-5 | 2e-5 |
| weight decay | 0.03 | 0.01 | 0.01 |
| SCL weight λ | 0.01 | 0.1 | 0.1 |
| temperature | 0.2 | 0.1 | 0.1 |
| queue size | — | — | 4096 |
| momentum | — | — | 0.999 |

共同配置：seed=42、max_length=512、dropout=0.1、projection_dim=128、eval_batch_size=8、warmup_ratio=0.1、AdamW、线性学习率调度、梯度裁剪 1.0、CUDA 混合精度。

**v1-A 与 v2-A 不仅对比学习机制不同，训练轮数、batch、正则化和对比损失参数也不同，因此不能把 A 的分数差异单独归因于 MoCo。**

## 5. 当前结果

所有分数均为 0–1 范围，来自各版本 `*_evaluation_prediction.ipynb` 的已保存输出，使用 `grouped_multilingual` 验证集。它们是本地验证指标，不是官方测试成绩。

| 任务 | 方案 | n | Macro-F1 | Accuracy | Target exact match | Scope Macro-F1 | Identity Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | v1 SCL | 1521 | 0.716902 | 0.743590 | — | — | — |
| A | v2 MoCo-SCL | 1521 | **0.719947** | **0.745562** | — | — | — |
| B | v1 SCL | 1521 | 0.512022 | **0.585141** | — | — | — |
| B | v2 MoCo-SCL | 1521 | **0.514733** | 0.577252 | — | — | — |
| C | v1 SCL | 790 | — | — | **0.605063** | **0.841163** | **0.261369** |
| C | v2 MoCo-SCL | 790 | — | — | 0.587342 | 0.832639 | 0.254444 |

C 的 790 条样本由**真实仇恨标签筛选**，上述结果衡量已知样本为仇恨时的对象识别能力，未计入 B 的误判。Identity Macro-F1 是 9 个身份各自正类 F1 的平均值；Target exact match 要求范围与身份组合整体正确。

### 训练日志中的最佳值

| 运行 | 最佳 epoch | 训练期间最佳验证指标 | 与独立评估输出一致 |
| --- | ---: | ---: | --- |
| v1-A | 3 / 8 | Macro-F1 0.719019 | **否：独立评估为 0.716902** |
| v1-B | 3 / 3 | Macro-F1 0.512022 | 是 |
| v1-C | 3 / 3 | Exact match 0.605063 | 是 |
| v2-A | 3 / 3 | Macro-F1 0.719947 | 是 |
| v2-B | 3 / 3 | Macro-F1 0.514733 | 是 |
| v2-C | 3 / 3 | Exact match 0.587342 | 是 |

v1-A 的差异原因无法仅凭现有文件确定；主表按独立评估输出记录，并保留训练日志值，不将二者合并为同一个已核实结果。本地未发现 `model.pt`、`history.json` 或独立 `metrics.json`，现有证据是 Notebook 中的历史输出，尚不能核对 checkpoint 与输出的一一对应关系。

### 当前结论与冻结选择

- A：按独立评估输出，v2 比 v1 高 0.003045 Macro-F1（0.3045 个百分点），可暂存 v2 为候选。
- B：v2 Macro-F1 高 0.002711（0.2711 个百分点），但 Accuracy 下降；按当前主指标 Macro-F1 暂存 v2 为候选。
- C：v1 的 exact match 高 0.017721（1.7721 个百分点），范围与身份 Macro-F1 也更高，保留 v1 为候选。
- 以上仅是单种子、单划分的观测差异，没有显著性或跨种子稳定性证据；没有无 SCL 基线结果，不能据此声称 SCL 优于普通微调。
- 停止继续搜索学习率、温度、队列长度等超参数，保留两套实现与现有记录。归档 checkpoint 和核对日志属于记录整理，不要求继续调参。

**A=v2、B=v2、C=v1 是按任务指标得到的候选组合，尚未完成端到端联合评估。** 当前 C 推理脚本自动加载同版本、同 seed 的 B 模型，所以直接运行 v1-C 时使用的是 v1-B；若以后采用跨版本组合，需要显式调整 B checkpoint 路径，并记录实际组合。

## 6. 目录与复现入口

```text
.
├── README.md
├── .gitignore
├── data/
│   └── README.md                 # 数据说明；原始数据本地存放、不提交
└── experiments/
    ├── prepared_data/
    │   ├── 01_data_preparation.ipynb
    │   ├── label_schema.json
    │   ├── statistics.json
    │   ├── description_cleaning.json
    │   ├── splits/              # 固定划分与统计
    │   └── processed/           # 含原文的 JSONL，不提交
    ├── contrastive_v1/          # A/B/C.ipynb + 对应评估/预测 Notebook
    └── contrastive_v2/          # A/B/C.ipynb + 对应评估/预测 Notebook
```

| 版本 | 训练入口 | 评估与预测入口 |
| --- | --- | --- |
| v1 | [A](experiments/contrastive_v1/A.ipynb) · [B](experiments/contrastive_v1/B.ipynb) · [C](experiments/contrastive_v1/C.ipynb) | [A](experiments/contrastive_v1/A_evaluation_prediction.ipynb) · [B](experiments/contrastive_v1/B_evaluation_prediction.ipynb) · [C](experiments/contrastive_v1/C_evaluation_prediction.ipynb) |
| v2 | [A](experiments/contrastive_v2/A.ipynb) · [B](experiments/contrastive_v2/B.ipynb) · [C](experiments/contrastive_v2/C.ipynb) | [A](experiments/contrastive_v2/A_evaluation_prediction.ipynb) · [B](experiments/contrastive_v2/B_evaluation_prediction.ipynb) · [C](experiments/contrastive_v2/C_evaluation_prediction.ipynb) |

复现流程（供需要时使用，不是新的训练计划）：

1. 按数据协议自行取得数据，放入 `data/StereoQueerEval_training/` 与 `data/StereoQueerEval_FA_development/`。
2. 数据已准备好时直接复用 `experiments/prepared_data/`；否则运行 [数据准备 Notebook](experiments/prepared_data/01_data_preparation.ipynb)，先修改其中的 `PROJECT` 路径。
3. 现有训练与评估 Notebook 面向 Google Colab，挂载 Google Drive；将准备数据放到 `/content/drive/MyDrive/prepared_data`。本地运行需要修改路径并跳过 Colab 挂载单元。
4. Notebook 安装 `transformers==4.52.4`、`sentencepiece>=0.2`、`safetensors>=0.4`，使用环境已有的 GPU PyTorch。完整环境版本尚未锁定。
5. v1 输出根目录为 `/content/drive/MyDrive/contrastive_outputs`，v2 为 `/content/drive/MyDrive/contrastive_outputs_v2`。运行名分别为 `{task}_scl_seed42` 和 `{task}_moco_scl_seed42`。
6. 若已有模型，直接在对应评估 Notebook 指定 `CHECKPOINT`，加载 `model_runs/<run>/best/`；不需要重新训练。评估会写出 `evaluations/<run>/<split>/metrics.json` 和 `predictions.tsv`。
7. 无标签预测默认 `RUN_PREDICTION=False`；需要时设置输入路径并启用。C 会先调用 B，把非仇恨预测置为 `none`，再预测其余样本的对象。

运行目录通常包含 `config.json`、`history.json`、`best_metrics.json`，以及 `best/` 下的模型、编码器配置和 tokenizer。后续归档应保留这些配置、汇总指标及版本对应关系；权重与逐条预测留在外部存储。发布版 Notebook 已清除执行输出和运行元数据；整理前的原始 Notebook 仅备份在本地 `.private-backup/pre-publication/`（不提交），README 保留本次汇总结果。

## 7. 仓库发布范围

远程仓库只发布方法说明、数据使用说明和清除输出后的 13 个 Notebook。`prepared_data` 下的 JSON、划分和处理后数据均不发布，由获准访问数据的使用者运行数据准备 Notebook 在本地生成。目录树展示本地运行布局，不代表这些生成文件已上传。原始数据、数据截图、权重、缓存和逐条预测不纳入版本管理。
