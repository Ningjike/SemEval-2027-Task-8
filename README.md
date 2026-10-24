# StereoQueerEval：实验记录
## 数据与方法
- 数据准备启用 `CLEAN_DESCRIPTION=True`，清理视频描述中的推广行；清洗后为空时保留原文。
使用 EN、IT、NL 共 7,627 条样本，`grouped_multilingual` 划分为训练 6,106 条、验证 1,521 条。按语言与原始视频标题、描述构造上下文分组，避免同组跨训练与验证集；该分组是视频上下文的代理标识。FA 的 100 条开发样本不参与当前训练，尚未记录评估结果。

编码器为 `FacebookAI/xlm-roberta-base`，输入为评论、标题与描述拼接后截断至 512 tokens，采用均值池化和任务分类头。A 为刻板印象二分类，B 为仇恨言论三分类，C 为对象范围分类与 9 个身份多标签预测。训练损失为监督损失加监督对比损失。

- **v1：批内 SCL**，在当前 micro-batch 内构造同标签正例。
- **v2：MoCo-SCL**，使用动量模型与带标签队列，queue size=4096、momentum=0.999；保留历史实现与结果。

共同设置：seed=42、learning rate=2e-5、dropout=0.1、projection dim=128、AdamW、warmup ratio=0.1。v2 仍为 3 epochs、batch size 4、梯度累积 4、weight decay 0.01。
- v1（批内 SCL）的 A/B/C 统一配置为 8 epochs、batch size 8、梯度累积 2、weight decay 0.03、SCL 权重 0.1、temperature 0.1。
- 更新 A/B/C 在验证集上单独加载 checkpoint 得到的独立评估结果。
## 本地结果

分数范围为 0–1，均为本地验证指标，**不是 Codabench 官方测试成绩**。

| 任务 | 版本 | 指标 | 分数 | Accuracy | 结果来源 |
| --- | --- | --- | ---: | ---: | --- |
| A | 新版 v1 | Macro-F1 | 0.721931 | 0.742932 | 独立评估 Notebook，n=1521 |
| B | 新版 v1 | Macro-F1 | 0.565439 | 0.598948 | 独立评估 Notebook，n=1521 |
| C | 新版 v1 | Target exact match | 0.630380 | — | 独立评估 Notebook，n=790 |

A/B/C 的独立评估主指标与可见训练日志最高值一致，分别出现在第 6/5/7 轮。C 的范围 Macro-F1 为 **0.811704**，身份 Macro-F1 为 **0.348667**。C 仅在 790 条真实仇恨验证样本上评估，不包含 B 的误判，不能视为端到端 B→C 的成绩。“独立评估”指单独加载 checkpoint 运行评估脚本，使用的仍是验证集。


## Codabench 测试状态

**尚未进行 Codabench 测试。** 根据 StereoQueerEval 组织者提供的通知，Codabench 平台当时发生故障（通知所附[问题跟踪页面](https://github.com/codalab/codabench/issues)），三个子任务的比赛页面将在平台恢复后上线。组织者还计划在后续几周发布 baselines 与官方评估脚本，并通过 Google Group 公告。

平台恢复且任务页面开放后，将进行 Codabench 提交测试，并补充官方评测结果。当前先保留本地验证结果。

## 脚本入口

- [数据准备](experiments/prepared_data/01_data_preparation.ipynb)：修改 `PROJECT` 后运行。
- [v1 训练与评估](experiments/contrastive_v1/) · [v2 训练与评估](experiments/contrastive_v2/)。
- [数据来源与使用协议](data/README.md)。
