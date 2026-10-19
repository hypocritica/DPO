# 已有实验的可复现配置

这些是实验定义和配置归档，不是当前作业状态或结果报告。来源以原实验冻结源码及 Git 历史为准。所有训练参数保留原值，仅替换机器路径。

| 模板名（configs/historical） | 状态构造与目标 | 原预算 | batch × accumulation × world |
|---|---|---:|---|
| dpo | 普通均匀加噪，A，K1 | 10000 | 32 × 4 × 2 |
| multiview-C | 普通均匀加噪，平均margin后loss，K2 | 10000 | 32 × 8 × 1 |
| multiview-D | 普通均匀加噪，C目标、随机单视图反传，K2 | 10000 | 32 × 8 × 1 |
| rollout-K1 | model扰动，单视图A | 2000 | 8 × 32 × 1 |
| model-K2-B | model扰动，平均loss，K2 | 2000 | 8 × 32 × 1 |
| model-K2-C | model扰动，平均margin后loss，K2 | 2000 | 8 × 32 × 1 |
| oracle-K1-C | oracle，单视图C | 2000 | 8 × 32 × 1 |
| oracle-K2-C | oracle，平均margin后loss，K2 | 2000 | 8 × 32 × 1 |
| 上述联合四臂的 -6000 | 同对应2000步配置，仅延长总预算 | 6000 | 8 × 32 × 1 |

共同设置为SD1.5 revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`，512px、BF16，LoRA rank=alpha=16、dropout=0，beta5000，AdamW lr1e−5、wd0.01、clip1，warmup200后恒定，seed17，canonical caption、balanced sampler。图片整图 LANCZOS resize，无随机crop/flip。使用原151418对训练清单才能保持历史数据曝光设计。

当前只允许B/C。上表中的普通A、单视图rollout A、D三个原配置已经移到`docs/legacy/`，参数没有改动，仅供追溯，不能用当前训练入口启动；原实验仍由原冻结代码复现。其余9份B/C配置仍在`configs/historical/`。

新实验默认使用`configs/model-K2-C.json`（或B模板），采用第200步及每500步监控、每200步滚动保留最新模型与完整续训状态、每1000步永久保留LoRA的新节奏。历史配置保留原稀疏计划，不能把新节奏误记成旧实验当时的设置。K1使用C时与原单视图目标相同。当前实现保留各视图计算图并对联合loss一次反传，旧冻结代码使用逐视图重算，额外增加策略评分前向；比较运行时间时需记录实现版本。

历史评估使用8个scene、每scene3种表达、seeds17/29/43，共72图；其中4个train、4个unseen。采样为512px、BF16、DDIM30、CFG7.5、空负面prompt，PickScore按原scene caption评分。原prompt及suite元数据属于复现输入，需单独带到新服务器。新脚本抽出的validation caption是一套新评估，不能冒称历史72图复现。

本包没有搬迁任何实验结果，也没有启动新的实验。原冻结实验和原环境继续负责已有作业与其恢复；重构后的source hash不同，不能拿本包继续写旧run目录。
