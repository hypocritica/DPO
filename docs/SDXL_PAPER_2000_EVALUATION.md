# SDXL 2000步主表与四个3000步Parti补充

完整部署与五组运行命令统一在[command.md首节](../command.md)。

用户确认的主表为Base0，以及SFT、MAPO、DPO、Global/Gaussian C K2、Stagewise DPO K6、
DMPO、Stagewise DMPO K6、DSPO、Stagewise DSPO K6的2000步，共10个版本。
仅DMPO、Stagewise DMPO K6、DSPO、Stagewise DSPO K6补充3000步Parti，共4个版本。
不加入SFT/MAPO1000、其他3000步、latest或未保存权重的1500/2500步。

`evaluation-sdxl-main.json`和`evaluation-sdxl-parti-supplement.json`从用户2026-11-13盘点提取，
固定完整run名称、run_id、永久checkpoint_id和模型revision。SFT使用v2，Stagewise DMPO K6使用b2-v2。
元数据不符、缺精确快照或方法语义不符时拒绝计划；生成前逐个验证权重文件checksum。
清单本身不是当前远端权重可加载的证明。

主表协议：1024×1024、EulerDiscreteScheduler、BF16、50 steps、CFG5；每GPU batch4、整模型常驻。
Parti/PickV2/HPSv2均seed17，GenEval使用17/29/43/71。
偏好面板保留既有PickScore、HPSv2、ImageReward、CLIP、Aesthetic五指标，同一PNG复用。
GenEval使用独立环境。此处HPSv2 scorer与SD3.5精简流程的HPSv3 scorer不同。

| 组（每组4×5090） | 内容 | 图数 |
|---|---|---:|
| A | 主表HPSv2 | 32000 |
| B | 主表Parti+PickV2，再四个3000步Parti | 27718 |
| C | 主表GenEval四seed | 22120 |
| 合计 | 主表75310 + 补充6528 | 81838 |

三组使用已有`/mnt/afs/L202500030/xietianyu/workspace/DPO-eval-v2`，
与双H100/四5090 SD3.5任务共享固定代码。保持同一资源表、batch和输出根目录，依赖同一AFS/AOSS挂载。
若已在其他checkout冻结了SDXL计划，继续使用该checkout恢复；配置绝对路径参与身份核验，不能直接迁移。
共同选择的短锁支持并行规划；独立group锁防止误启动同组两次。不能在运行中更新checkout。
主表根目录为`.../sdxl/paper-2000-5090-v1/main`，补充为同级`supplement`。
其中每个面板独立保存plan、prompts、PNG、manifest、scores、image-index、summary和paper。
补充计划显式不含Base，既有主表图片无需重复生成。

重复执行同一`run_sdxl_paper.sh A|B|C all`即可恢复。已完成图片和分数校验后跳过；部分批次
按原批边界重新计算，仅发布缺失PNG。输入、batch或seed变化拒绝接管旧输出。
生成失败不会进入评分/导出。评分/汇总可独立运行，不需要重新规划或读取训练权重。
需要更改batch时，使用新根目录并让三组统一该值；不会静默降batch。

```bash
bash scripts/run_sdxl_paper.sh B plan --dry-run
bash scripts/run_sdxl_paper.sh B generate
bash scripts/run_sdxl_paper.sh B score
bash scripts/run_sdxl_paper.sh B summarize
bash scripts/run_sdxl_paper.sh B paper
```

用时尚无真实5090测量。按每卡有效生成2–4秒/图计算，并考虑10个模型任务分给4卡的尾部空闲：
A约5.3–10.7小时，B约4.4–8.9小时，C约3.7–7.4小时，均不含评分、模型加载和I/O。
三个组同时运行可先预留8–18小时总窗口；这是条件性预算，不是实测承诺。
实际数据由每次生成后的report_generation_sessions和组日志的phase elapsed_seconds提供。
CPU检查不能证明batch4显存或CUDA环境可用，正式运行前可在已分配资源上做小规模验证。

论文应说明2000步是在查看监控曲线后确定的共同步数，四个3000步是单独的补充对照，
不把主表称为每个方法的最优checkpoint。监控采用30采样步，正式评测50步；监控分数不能
作为正式表格数据。相同步数/pair-view预算不等同于相同pair曝光或FLOPs。
PickV2完整487条面板可能和监控prompt重叠，未做实际去重核验前不能声明完全独立。
