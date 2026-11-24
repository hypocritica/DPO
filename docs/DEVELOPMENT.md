# Self Correct DPO 开发与实验约定

## 2026-11-24 parallel图库worker派发修复

回传日志确认parallel run误调用scripts/appendix_figures.py，5090完成生成后被原H100 worker的runtime检查拒绝。改为调用parallel脚本自己的worker。仅放行已发布旧launcher SHA的恢复，保留旧plan ID、历史输入、已生成PNG；重新prepare仅在其余冻结输入一致时复用原plan。没有修改src、原H100入口或flexible定量入口。

14项CPU检查通过：新增四卡worker派发与runtime传递、旧plan字节保留/未知launcher及输入变化拒绝、原worker render图片无新增生成恢复。AOSS只读核对Base/SFT四分片各50图的complete记录；未执行远程GPU修复任务。原目录续跑命令在command.md首节。

## 2026-11-24 四卡任务拆分

独立codex/mvpo-parallel-5090基于300c274，复用本机未提交flexible实现；src及原H100入口字节不变。新增parallel配置将原79个曲线任务精确拆成E1/E2/E3与SD35七方法，固定四分片，2/4/8卡调度。图库独立脚本复用原拼图格式，为保护冻结src采用脚本内适配；SD15/SDXL可5090，SD35/Z图库仍要求H100。图库索引跳过未完成外部render，允许恢复自己的部分图片。旧产物只读，独立输出，生成/评分分离，命令见根command.md。


## 2026-11-23 双H100生成、LossAvg匹配对照与浅层绘图包

同分支保留梯度修复cd80347，在当前生成代码扩展group-h100-a/b各2卡。batch=16/8/8/4且全部无offload。新增SD35 LossAvg K2/K4、Global/Stage K2主表与400步间隔曲线；22主表、92曲线点去重148160图，离线历史分数复用33152、新增115008。LossAvg校验formal_arm B、相同K目标/data/base及训练设置；checkpoint仅从指定run/step解析并冻结，不用latest。实际权重仍待服务器核验。

固定100艺术词、30原复杂词、12条Parti展示词。SD15九列、SDXL八列（用户确认无KTO）、SD35五列、Z三列；训练进程固定同prompt/seed，图库3740槽位，预计额外新增2756。历史PNG仅按SHA与原生协议/包版本/模型身份匹配复用；缺像素重新生成，缺权重显示空缺。所有候选导出，不按分数自动挑优。SDXL保留旧5090来源，本轮生成H100。

新增appendix_progress将原生生成/评分日志落盘，journal驱动完成/剩余/实测阶段ETA，子任务失败或信号终止整组worker。不改render.py字节，保留既有精确renderer SHA兼容约束。原生truncate策略记录超长prompt，UI不刷屏；真正错误仍可见。figures/index.html与同层CSV/JSONL/PNG/PDF并列，原图一层images；带GPU/step/seed/截断标记。导出按family与共享index加锁。

轻量CPU检查覆盖数量/去重、LossAvg语义、原生token截断、逐图复用/缺图生成/恢复/导出、组级派发/错误传播、已有多种子与方法目录检查。合成拼图和七曲线布局已检查，非实验结果；没有提交GPU。本轮完整命令置于command.md首节，桌面清单同步；输出新根mvpo-appendix-h100-v2。

## 2026-11-23 Native adapter训练batch记录兼容

用户回传DSPO step1000两份不可变元数据仅batch_size/gradient_accumulation_steps不同：2×96与16×12。native checkpoint允许这两个正整数字段重分配且乘积相同，其余训练字段和模型仍要求一致；DPO/Flow旧路径不放宽。checkpoint identity记录原始adapter_training_adjustments，不改训练文件。错误包含backend.json绝对路径和具体字段，便于区分DMPO后续checkpoint的未知差异。相同有效batch不代表浮点训练等价，本次仅允许对已保存权重做固定checkpoint测量。

28项CPU梯度检查通过，覆盖真实差异、保留原字典、非整数/不同乘积/beta/模型变化拒绝、默认DPO严格比较和既有探索模式。未在GPU加载实际权重。command.md提供同分支独立worktree启动4/5，避免影响正在运行的1–3；v2输出隔离旧源码身份，未修改既定checkpoint数量。

## 2026-11-23 用户指定跳过梯度数值核验

在原codex/mvpo-multiseed-generation分支新增显式MVPO_SKIP_NUMERICAL_CHECKS=1/--skip-numerical-checks探索模式，覆盖任务1–5。模式冻结于配置和job身份，probe不做初始化/repeat/batch检查，runner跳过pilot直接32对采集；续跑、collect和报告不再要求相应核验文件。保留默认旧严格流程和完整性检查，不把skip写成passed。scan.json/protocol/逐checkpoint及跨checkpoint CSV/所有图标明skipped；NaN/Inf不转换成有效结果。

新加--family可只采Flow；与--pilot-only互斥。探索仍使用原SD15确定性路径和原Flow后端，不合入独立flow-determinism改动。源码变化需新输出根，原失败目录不重写。生成/历史复用功能无修改。CPU检查覆盖任务1–5共用runner跳过pilot、两launcher环境开关、完整记录可汇总、缺项仍拒绝及未经核验标记导出；合成报告渲染验证不是研究数据，未运行SLAI GPU。完整拉取与启动在command.md顶部。

## 2026-11-23 同分支、跨GPU复用与三组实际提交

用户要求继续维护codex/mvpo-multiseed-generation及原远端checkout；将历史SHA兼容修复合入，不再要求切换独立分支。明确授权H100旧分数与5090新增图混用。prepare新增显式generation-gpu，将history_noise_runtime与新noise_runtime分别冻结，source保留真实runtime；其他身份和版本校验保留。报告逐槽位推导生成GPU，输出hardware-provenance.csv、配对GPU是否一致和图表说明，不把混合硬件seed波动当纯随机种子效应。

三个分组入口：5090-a四卡顺序SD1.5+SDXL九包，5090-b四卡顺序SD3.5-M两包，h100双卡Z一包。family分片4/4/4/2；batch4/4/2/4，Z与SD35均关闭offload（SD35由用户随后明确确认），SD35仍为每卡batch2、4×5090。历史manifest证明SDXL/5090与Z/H100的batch4已有完整生成；SD15/5090 batch4和SD35/5090 batch2尚待GPU实测。全量盘点吞吐汇总在桌面报告中，未将不同步数/精度/面板的吞吐比称为batch加速。

16项CPU检查通过：跨GPUprepare仍复用旧分数，旧源runtime不被改写，新runtime及prepare选项冻结，错GPU拒绝；分组四卡数量检查、串行派发和失败停止、报告配对硬件标志。合成报告图经过检查；没有提交远端GPU。command.md顶部为同分支pull、prepare四模型及三个独立括号启动命令，输出统一为mvpo-multiseed-mixedgpu-v1。旧冻结计划保持原样。

## 2026-11-23 真实历史复用核验

回传 inventory 的 8341 个文件通过大小/SHA 校验；801 个 render、930219 条旧分数的身份、manifest 关联与 summary 均核对。18 个主表均有全量单种子；SD1.5 实际 Euler50/seed0，另有 DSPO+MVPO10k 全量。原生 read_source/choose_slots 离线重放得到复用29888、新增84352，仍需远端资产/权重/环境匹配。旧128条 monitor 面板不抵扣新的512条 Parti 曲线。

Git e6ee941 到当前 renderer 的差异仅影响 Z-Image计算与会话元数据；将已存在的精确 SHA 等价检查由 SD3.5-M 扩至 SD1.5/SDXL，拒绝未知SHA及旧Z精度。15项CPU检查通过，包含真实格式旧SD1.5日志/分数复用、Z与未知版本拒绝。独立 codex/mvpo-history-reuse-audit 分支和新checkout/输出根，避免更改运行中的梯度源码身份。根command.md新增真实protocol-source、prompt路径、拉取/prepare/启动命令；SDXL沿用5090，其余主表历史生成使用H100。未执行远端prepare或GPU任务，未读取实际PNG/权重，论文协议差异仅记录未修改正文。

## 2026-11-23 自动评分器胜率与括号式运行命令

按用户命令风格在command.md顶部提供完整clone/pull、conda、safe.directory、环境变量、盘点、prepare、任务包和CPU汇总块。新生成入口识别SCD_EVAL_GPUS，默认两卡；旧SDXL峰值入口及其目录不改。每任务包成功后自动按family报告，失败停止；可显式关闭自动报告。并发报告用阻塞文件锁串行发布。

既有半平局win_rate补齐胜/平/负数量、严格胜率、分母与逐seed胜率，新增win-ratios.csv和每评分器的PNG/PDF百分比横条图，包含50%线和prompt-bootstrap区间。半平局两段互补，主表不同训练step标注，未完成比较显示N/A。当前仅PartiPrompts三生成种子，SD1.5/SDXL为HPSv2、Flow/Z为HPSv3；不等同人工偏好率，不新增Base/SFT或其他面板。

14项多种子CPU检查通过，新增胜平负/互补/全平局与真实Bash成功自动报告、失败停止、显式关闭分支检查。合成已有分数重新报告并检查胜率图；未执行GPU或访问实际评估分数。


## 2026-11-23 完整多种子生成与 evaluation 产物复用

独立 `codex/mvpo-multiseed-generation` 分支新增 appendix-multiseed.json、盘点/计划/分包执行/报告入口。18 个主表配置、60 个曲线点，重合点按唯一 method/step/seed/prompt 任务复用，共 105 个生成 job、114240 张目标图。SD1.5 DSPO+MVPO 主表采用已核对的6000步；10000保留曲线。SDXL DPO两者2000、DMPO10000/9000、DSPO两者3000。无 LossAvg 或新增训练。

历史盘点只导出 evaluation 元数据与逐图分数，不带图片/权重，不要求旧产物有 plan.json。准备按原生 run/manifest/score 身份逐项核验，可用部分已完成记录；已有完整分数即使图片缺失也复用。缺指标的有效旧图复制到新目录的可追溯评分子面板，旧数据不写入；缺图且缺指标时统一重生成/评分，不把不同像素的分数拼接。完整运行版本、scorer、协议、权重、prompt与seed冻结，历史源更新拒绝混用。模型/包级缺权重保持blocked，不自动更换checkpoint。

SD1.5 的正文 PNDM50/17 与代码 Euler50/0 尚需真实导出判定，prepare 强制指定实际主表 render/run.json；来源必须对应本次主表权重和完整 PartiPrompts。其他模型支持同样覆盖，默认沿用现有 evaluation-suite。三种子复用 s0/23/42，曲线使用s0，默认生图batch2（可显式4）。匹配历史 CUDA GPU/版本，换不兼容硬件必须另根目录显式fresh；原梯度入口不改动。根command.md包含拉取、只读盘点、CPU准备、12个任务包及汇总命令。

12项新增CPU检查与23项既有梯度回归检查通过。新增检查覆盖完整数量、同图跨任务去重、协议差异、无图有分数、部分评分、错误身份、导入图像完整性、native renderer/scorer调用、断点完成跳过、准备冻结与prompt统计。合成报告包含主表均值/分差、seed波动、相邻checkpoint比较和缺项曲线；图经渲染检查。没有访问SLAI、提交GPU或实测新任务吞吐/显存。桌面清单区分“代码就绪”和“历史盘点/远端准备待完成”。


## 2026-11-23 任务 4/5：原生 DSPO/DMPO 梯度扫描

新增独立分支 `codex/mvpo-objective-scan`、原生目标 collector/probe、任务 4/5 launcher 与冻结配置。任务 4 扫 DSPO 及 Stage-wise DSPO K6 的 1000/4000/6000/10000 步；任务 5 扫 DMPO 对应两种权重的 1000/4000/10000 步。6000 双方由用户确认加入，以覆盖正文 DSPO+MVPO 主表。各扫五个 scheduler index、32 对、4×48 source 与 4×64 audit，分别 573440/430080 pair-view 观测。DSPO 每 view 三项分量 VJP，不能与 DPO 按相同反向成本解释。

核验训练库存源码 SHA 与 git 9573c1e 的 objectives.py 一致。DMPO 按原 reverse-KL 及有限最优 logit；DSPO 按原 score-residual，利用 margin/winner MSE/prediction gap 的梯度与常量 reference MSE 重构完整共享 gate 梯度。初始化、重复与 batch 检查使用 native 目标；每权重独立 audit 的 native loss gradient 为方向参照。新增 schema v2 的原生 moment/Gram 记录、严格回放验证与完整热图；DPO v1 collector 和默认流程保持兼容。

资源集中登记；方法与 step 均进入目录身份，四方法共享 pair/latent/noise。新代码、输出与原任务 1–3 独立，原 3735e7f checkout 不更新。用户负责在 SLAI 提交两个 2×5090 任务；默认 batch2，可显式 batch4/独立输出。拉取、两段完整启动和恢复/汇总命令见 command.md 顶部。未访问训练机器、提交 GPU 或实测新的 native 5090 显存/速度。

23 项 CPU 梯度检查通过，覆盖原生损失链式梯度、batch 1/2/4 逐 view 一致、Gram 回放、checkpoint 身份和既有扫描回归。DSPO/DMPO 合成数据完成严格校验、单条件报告与跨 checkpoint 热图的端到端检查；热图已检查渲染，缺项保留 N/A。Shell 语法与 diff 检查通过；合成报告仅用于软件验证，不是实验结果。


## 2026-11-22 三任务 checkpoint/index 探索扫描

新增 `gradient_scan.py` 与 `run_gradient_scan_slai.sh`，基于 `codex/mvpo-gradient-walkthrough` 分支扩展为三个独立 2×5090 提交任务。每任务顺序处理 SD1.5/Flow 各一个 checkpoint、各五个固定 scheduler index；配置冻结 32 对、4×48 source、4×64 audit，30 条件共 430,080 次 pair-view 观测。默认 view batch=2，可显式设为 4；合批前向但逐 view 求原始梯度，不以 batch 均值梯度替代。保留原 SD1.5 严格确定性路径、旧入口与输出。

扫描通过已知 DPO checkpoint 锚定 training run，再冻结目标 step/完整 checkpoint/adapter 身份。跨 checkpoint 共用模型级 pair 面板、posterior latent seeds 和同 index 的噪声 bank；source/audit 分流。共享元数据用短时阻塞锁，任务执行锁拒绝重复启动。各 checkpoint 的独立工程 pilot 深验通过后进入正式采集；中断按 pair-condition 恢复，源码或 batch 漂移拒绝混写。支持新固定 index 的严格报告校验、同 K 配对差和完整热图，缺项及 audit split-half 诊断保留。第二轮确认条件尚待第一轮完成后冻结，三任务入口不会自动选择最佳条件或启动确认。

18 项 CPU 轻量检查通过，含 batch 1/2/4 逐 view 及 Gram 一致、跨 checkpoint RNG、Flow shift-3 native time 映射、30 条件数量、并发元数据、错误身份、缺 batch 检查拒绝、旧 workflow 和数值统计回归。为验证 autograd，仅在 `/tmp` 隔离测试环境安装 CPU PyTorch 2.8；未改用户原 conda 环境。新条件的合成报告端到端完成，并检查热图渲染。Shell/Python 语法与 diff 检查通过；尚未运行真实 SD1.5/SD3.5-M 的 5090 pilot，未访问训练机器、提交 GPU、改旧 D3 代码或旧结果。部署、三个启动命令、batch 4/恢复/汇总入口统一写入 `command.md` 顶部。


## 2026-11-22 MVPO 梯度逐项监督入口

新增gradient_experiment.py的plan/prepare/pilot/run/status/collect/report独立动作，SD1.5与Flow、pilot与formal分目录；正式128对、8×96 source、8×128 audit，按固定pair group再分GPU。正式worker要求对应pilot完整且两项检查通过，核验采集源码/checkpoint/数据身份。旧采集、后端、Gram格式、确定性SD1.5路径及默认入口未改；旧16对D先直接报告，不能由新prepare接管。报告新增同pair同K变化、audit分半诊断CSV和PNG/PDF，保留原始梯度、有限audit参照及两层区间；稀疏Gram收缩与原稠密公式等价。

既有ate环境的8项轻量检查中7项通过、原PyTorch collector检查1项因缺torch跳过。覆盖原始向量/去耦合、配对差、audit分半、Gram收缩等价、纯标准库plan、固定分组分片及缺pilot拒绝正式worker；toy报告图经渲染检查。未安装依赖、访问SLAI、下载模型、提交GPU或实测5090显存/性能。逐步命令在command.md顶部，解释见MVPO_GRADIENT_WALKTHROUGH.md。

## 2026-11-17 SDXL峰值补评

新增prepare_sdxl_peak.py与run_sdxl_peak.sh，固定DSPO两行3000、DMPO10000及Stagewise DMPO
最新永久LoRA（优先10000，否则例如9000），忽略latest.json、resume-*及9525滚动checkpoint。
核验所选run/永久checkpoint身份及权重哈希，首次catalog封存后不会随训练变化。
A组HPS，B组Parti/PickV2，各四卡、先生成再评分，沿用2k协议和五指标；兼容完成原图/分数
导入包含身份与协议核验及来源记录。17项CPU测试通过，覆盖永久9000固定、首选10000、
忽略滚动指针、损坏永久权重拒绝回退、图像/分数复用、两组顺序和失败停止及旧SDXL/导出回归。
未执行远端GPU加载、提交任务或下载模型。完整命令见SDXL_PEAK_SUPPLEMENT.md。

## 2026-11-15 运行目录统一为已有DPO-eval-v2

按用户要求，command.md当前五组命令统一使用已有远端DPO-eval-v2，移除新clone步骤。
只在启动所有worker前更新一次代码，运行阶段不再pull；评测输出、checkpoint、seed、batch均不改变。
同步SDXL与SD3.5 HPSv2说明，注明已经从其他checkout冻结的SDXL计划不能直接更换配置绝对路径。
本次仅文档调整，六段当前命令通过bash -n及diff检查，未操作远端文件或GPU。

## 2026-11-15 SDXL三组四卡：统一2000步与四个3000步Parti

固定用户盘点中的九个SDXL训练run及历史run/checkpoint ID，主表Base0+九方法2000，
只补DMPO/Stagewise DMPO K6/DSPO/Stagewise DSPO K6的3000步Parti，不补SFT/MAPO1000。
catalog可选expected_run_id/expected_checkpoint_ids在规划时核对；不匹配或缺快照拒绝回退。
prepare_sdxl_formal增加显式共享panel列表与methods过滤，支持GenEval及无Base补充，旧默认三面板不变。

run_sdxl_paper.sh提供A/B/C三组独立四卡入口，A为HPSv2 32000图，B为Parti/PickV2及补充27718图，
C为GenEval四seed 22120图；总81838图。batch4常驻、BF16/Euler50/CFG5，偏好面板seed17。
共享选择短锁与组级锁保护并行计划及重复启动；阶段失败停止，重复命令恢复图片/评分/导出。
每阶段打印用时，保留全部PNG、索引、分数和论文表。三组要求同一固定checkout/资源/协议路径。
双H100及四5090 SD3.5入口保持原实现；Z renderer及精确SHA兼容策略未改。

73项CPU检查通过：真实完整面板数量/种子/补充无Base、run/checkpoint身份拒绝、四卡UUID和环境路由、
重跑与失败停止、三进程并行规划、旧H100/SD3.5交接、Z BF16恢复、批次缺图恢复及论文导出。
六段新部署/运行命令与三个入口Bash语法、JSON和diff检查通过。未提交GPU作业或实测吞吐/显存。
command.md首节汇总五组完整命令和有条件时间估算。

## 2026-11-15 双H100：PickV2也改为单seed

新增H100 v3配置，在v2基础上将PickV2固定seed17；Parti仍seed17、GenEval四seed。
串行脚本显式使用H100 v3配置与compact-h100-v3新目录，batch8常驻；原Z曲线batch4/BF16续跑不变。
主表八版本三面板34,648图，四张5090 HPSv2任务仍用v2，旧v1/v2计划不改写。
21项相关CPU检查通过，覆盖v3实际图片量、各面板seed、batch/offload及串行脚本的配置选择和恢复。
Bash语法与diff检查通过；未运行远端GPU。完整命令已同步command.md首节。

## 2026-11-15 后续：Parti/HPSv2单seed与四张5090

新增compact-v2配置：Parti/HPSv2面板单seed17，SD3.5 PickV2仍17/29，Z-Image PickV2仍17，
GenEval四seed不变。新H100默认目录compact-h100-v2，前三面板38,544图；旧v1配置和计划保留。
新增run_hpsv2_5090入口：主表八版本、HPSv2完整3200prompt、25,600图，四个独立worker。
默认batch1/CPU offload=true并冻结在独立selection中，支持原图索引、生成吞吐、三核心指标和论文导出。
selected init增加显式cpu-offload选项，原入口默认false；两卡入口默认仍严格两卡，四卡入口显式声明四卡。
renderer及BF16精确SHA兼容策略未修改，不影响旧Z曲线续跑。

58项CPU检查通过，包括v1/v2实际面板图片数、单/双/四seed、5090 offload和H100常驻协议、
相同checkpoint身份、四卡UUID传递、完整shell阶段顺序和生成失败停止评分，以及旧BF16恢复/正式suite回归。
Bash语法与diff检查通过，未提交GPU作业或进行5090实测；模型、环境、显存和性能须远端核验。
最终命令与独立checkout并行说明见SINGLE_SEED_FLOW_EVALUATION.md及command.md首节。

## 2026-11-15 精简正式评测与双H100串行流程

新增独立evaluation-suite-compact配置：Parti+PickV2+GenEval；SD3.5偏好seed17/29，
Z-Image偏好seed17，GenEval保持四seed。核心偏好指标为PickScore/HPSv3/ImageReward，
UnifiedReward作为关键三方法Parti的可选补充。旧full配置与renderer不改。
selected plan支持main/all范围；compact默认仅主表八版本，交接DPO800仍保留可单独规划。
SD3.5主表三面板51,600图，Z-Image三方法12,993图，完整prompt集不抽样。

run_zimage_then_sd35_compact先恢复用户原batch4曲线与PickScore，再以batch8运行SD3.5；
任一阶段失败停止，已完成图片与导出复用。仅既有Z-Image流程使用上一版批准的BF16恢复策略，
未改renderer/common/padding或精确SHA白名单。独立compact输出不会接管旧全量计划。
55项CPU检查通过：主表/补充对照分离、完整面板图片量、核心表导出、Z-Image seed与BF16协议、
shell顺序与失败停止、重跑跳过init/已有导出、缺正式评分环境仍完成Z曲线、旧曲线/全量suite回归。
另通过Bash语法和diff空白检查。未运行远端H100生成/评分、未下载模型或安装环境；吞吐仍需远端日志。
最终命令、环境路径、输出和有条件时间预算见COMPACT_H100_EVALUATION.md及command.md首节。

## 2026-11-15 Z-Image评估默认BF16 autocast

仅Z-Image评估的Transformer forward默认启用BF16 autocast，在PEFT加载后包装，
保留FP32权重、padding兼容处理、调度器和FP32 VAE解码。有效sampling记录
transformer_autocast_dtype=bfloat16，可在独立配置中设none进行FP32对照；其他模型不启用。
不修改远端运行中的checkout或任何冻结实验产物，不扩大AOSS存储兼容白名单。
60项CPU检查通过，覆盖有效配置、实际BF16算子与FP32权重/VAE隔离、批处理、suite、
curve准备、索引、曲线汇总、存储兼容和精度切换续跑。1项真实tiny Z-Image因现有环境缺现代类跳过；
H800吞吐和图像质量尚未验证。

用户随后明确授权当前曲线保留旧FP32图片、缺失部分用BF16续跑并推送。
新增独立zimage_precision_compat策略，限定3148b5b的原renderer与本版目标renderer精确SHA；
其余冻结输入及common/padding源码仍严格核验。原run身份和PNG字节不变，恢复时写入
精度切换审计，新图manifest逐张记录精度与源码SHA，sessions记录有效sampling，
图片索引区分旧none与新bfloat16。完整点直接核验跳过，不重新生成。

## 2026-11-14 Z-Image默认双卡恢复

按用户已跑通的H100命令调整GPU默认值：SCD_Z_GPUS、SCD_EVAL_GPUS、CUDA_VISIBLE_DEVICES
依次取首个非空值，全部未设置或为空时默认0,1，启动命令不再要求额外配置GPU变量。
7项launcher CPU测试通过，包含无变量/空变量时all完整执行、已有编号/UUID继承、生成和评分
路由一致、失败/中断停止及非法显式配置拒绝。未改变冻结输入、采样或渲染身份；未跑GPU。

## 2026-11-14 Z-Image快捷任务与SSH准备分离

新增run_zimage_curves.sh包装既有evaluate_curves：Base一次、DPO和Stagewise K4的全部已保留
永久checkpoint，每点固定128图/seed17。all依次prepare/plan/generate/index/PickScore/summary，
前一阶段失败即停止；分阶段入口支持只生成、只评分、步数/方法筛选及可选HPSv3辅助曲线。
专用SCD_Z_GPUS支持1/2/4张已分配卡，避免复用SDXL的GPU配置；CPU盘点和资产下载在SSH执行。
准备与图片仍在原AOSS Z-Image路径，已冻结输入不自动纳入后续快照；原曲线身份和渲染代码不变。
22项CPU检查通过，覆盖shell顺序、失败/中断停止、GPU隔离和既有曲线冻结/恢复/导出。
没有执行SLAI GPU任务或下载；batch1、offload=false仍须远程验证。命令见command.md首节。

## 2026-11-14 SD3.5M 已确认主表与额外步数对照

evaluate_selected.py init --handoff 导入原交接JSON并核对实际曲线prepared：Base0及七方法1600
进入主表，额外DPO800标记development-selected，3200仅保留开发曲线。新SFT从唯一实际条目
解析，旧run/path/checkpoint身份及权重内容hash不匹配则拒绝；确认后所有步数不可手改。
计划只生成九个版本一次，export按冻结selection ID分别导出八版本主表与DPO800/1600对照，
复用DPO1600图片。导出只依赖冻结计划和评测产物，原训练目录移除后仍能读取报告。
没有handoff的旧选择表与Z-Image保留手填单步行为。renderer、scorer与SD1.5入口未修改。

50项CPU检查通过：49项选择/启动器/论文导出/Z-Image/SD1.5/catalog回归，以及1项首次plan前
禁止修改已确认步数的检查。包含身份不符、损坏/缺失adapter、SFT歧义、报告ID漂移、九版本
唯一输出和移除训练文件后的实际草稿导出。JSON与原交接逐字一致，Bash语法及diff检查通过。
未执行GPU推理、安装环境或下载模型。当前batch8、offload=false须远程验证。
新AOSS输出和完整命令见docs/SD35_MAIN_1600_EVALUATION.md及command.md首节。

## 2026-11-14 选定 checkpoint、Z-Image 曲线和 SDXL 分组评测

新增 evaluate_selected.py：SD3.5M 八方法与 Z-Image 三方法从冻结曲线导入精确 run、
永久快照与选点来源，selected_step 留空，人工填写后才能正式计划；首次发布选择封存，
后续各面板不能换步数。report_selection 随 plan 和论文导出保留训练数据身份、预算和
开发面板独立性限制，不把相同 pair-view 预算写成相同图片对曝光。

evaluate_curves.py 扩展 Z-Image Base/DPO/Stagewise K4 的固定128图开发曲线；默认PickScore，
HPSv3 可事先指定为选点指标或事后作为辅助评分，同一PNG复用、图表分目录。既有SD3.5
默认准备身份和PickScore输出保持兼容。renderer、评分器实现和旧H100入口未改动。

run_h800_selected.sh 分离生成、各环境评分、图片索引和论文导出；Z-Image 两正式面板可
用 export_zimage_table.py 核验后合为三行五指标。run_sdxl_formal.sh 支持三组各4×5090
分别生成Parti/PickV2/HPSv2，共享冻结catalog和协议，面板产物独立，稍后再评分。
所有运行产物在AOSS；不提交资源、不隐式安装下载、不自动选择checkpoint或降低batch。

88项CPU检查通过，包括新选择封存/输入漂移、Z曲线与辅助指标、真实三进程并发规划、
GPU token转发、生成/评分分离及既有SD35/SD15恢复、索引和论文导出回归。Python/Bash
语法与diff检查通过。未执行H800/5090推理、安装环境或下载权重；默认SD35 batch8、
ZImage batch1、SDXL batch4的显存及吞吐须远程验证。完整命令从command.md首节进入。

## 2026-11-13 SD1.5 batch试跑

确认SD1.5使用整套GPU常驻，未启用offload/attention或VAE slicing；保留原render源码与
采样身份。加入sd15-probe，仅需原eval，Base+DPO各两批，独立输出，不评分；结束报告
forward吞吐、总耗时与峰值显存。17项CPU回归通过，覆盖无Flow环境的两卡batch96试跑、
192条smoke来源、阶段停止，以及SD1.5计划与原批恢复。未执行真实H100推理。

## 2026-11-13 resident batch试跑

加入from-prepared复制冻结128prompt和checkpoint清单，允许新计划显式指定cpu_offload；
源目录和身份不修改。H100启动器新增probe，仅跑Base+DPO3200各128张；all复用后继续。
可跳过环境/资产及全点预检，冻结输入和checkpoint一致性核验保留。34项CPU回归通过，
覆盖源输入字节不变、无训练重扫、batch32/offload=false冻结与漂移拒绝、probe停止与
检查跳过，以及原曲线/计划流程。未运行H100或验证batch32显存/吞吐。

## 2026-11-13 首次计划与日志目录冲突修复

启动器在plan之前创建logs，触发非空输出保护。计划初始化现在仅允许预存普通logs目录，
保留所有日志；无plan的其他产物、logs文件或符号链接仍拒绝接管。无需删除或迁移输出。
38项CPU回归通过，包含真实SD1.5 plan预存日志、两卡batch64恢复、通用plan日志保留及
非日志产物拒绝、曲线准备和启动器；未执行GPU或修改采样/评分协议。

## 2026-11-13 双H100顺序评测

SD1.5入口支持2/4卡及plan显式generation batch，后续自动读取冻结批大小；
SD3.5曲线移除5090专用batch4上限，仍要求正整数并拒绝更改已有准备身份。
新run_h100_two_track.sh依次执行SD3.5生图/PickScore、SD1.5生图/五指标。
默认8/64仅为候选，分别先跑可复用DPO3200曲线点及独立Base+DPO整批smoke。
22项CPU检查通过，覆盖批大小冻结、恢复、两卡、真实shell编排顺序和smoke来源。
未运行H100或安装/下载资产；显存和吞吐以远程实际生图为准。

## 2026-11-12 SLAI两条评测工作流与批量生成

SD1.5专用入口冻结交接的11个版本、三套正式偏好面板、58509张图片和legacy五指标，
按Parti→PickV2→HPSv2生成，每卡batch32；计划发布前核对历史run/checkpoint身份及
adapter文件hash。主表固定使用Stage-C DSPO 6000，10000只进入独立步数对照。
SD3.5M曲线每卡batch4，每checkpoint128图，仅PickScore，四卡命令独立于SD1.5。

renderer使用prompt列表与逐图Generator；批边界固定，部分恢复重算整批，只提交缺失
PNG，不自动降batch。图片、manifest、日志、评分和曲线保存在AOSS分模型目录。
新增image-index.csv/json支持评分前按模型、方法、step、prompt与seed定位原PNG。
资产工具分离离线检查、metadata计划、显式下载和source准备；复用完整legacy手动缓存。
首次部署、独立环境和两轨命令见[SLAI_TWO_TRACK_EVALUATION.md](SLAI_TWO_TRACK_EVALUATION.md)。

本轮80项相关CPU检查通过（78项组合回归，另补批大小冻结及HPSv2源码目标路径测试）。
检查覆盖批量尾批与中断补图、精确checkpoint身份、主表/对照筛选、缓存复用、图片索引、
曲线与论文导出、原存储兼容；Python/Bash语法及diff检查通过。没有安装环境、下载模型
或执行GPU推理；5090显存、吞吐和真实评分仍须SLAI验证，不扩大旧源码兼容白名单。

## 2026-11-13 SD3.5M checkpoint曲线

`evaluate_curves.py`复用统一suite的计划、生成、评分与checkpoint级多GPU分片。
独立prepare冻结七方法全部永久快照及Base、128条validation文本、seed17和PickScore；
来源caption按NFKC/casefold/空白归一化去重，排除所有选中训练文本及已备齐的正式面板。
缺少排除面板明确记录独立性尚未建立，不假称完整held-out测试。输入产物校验后才能
规划，缺方法/重复候选/不足128条不回退。原训练目录只读。

曲线按同一prompt/采样/基座/评分协议验证；缺失和损坏保留状态，图中断线，不自动选
最佳step。标准误带明确非置信区间。生成/评分复用既有事务恢复，换2/4卡可重新分片。
PNG/PDF与表格在临时目录准备，最后发布status及hash清单。未执行SLAI真实GPU测试。
51项相关CPU测试通过，包括实际CLI准备/冻结/重跑、2/4个真实子进程分片、
partial到complete图表更新、损坏产物拒绝和launcher失败/中断处理。
命令见[曲线说明](SD35_CHECKPOINT_CURVES.md)。

## 2026-11-11 方法登记、SLAI部署和论文导出

在独立evaluation分支扩展`catalog`：实时导入多个runs根目录，使用训练配置验证方法，
保留精确永久checkpoint步数，拒绝歧义候选。Base按完整模型身份去重并独立于训练目录。
五面板计划冻结prompt的来源及full/smoke范围；没有selection来源的面板明确标记。
`export_paper.py`只消费已生成图片与评分，输出带协议分组和SHA256的CSV/JSON/LaTeX；
缺失不填0，不自动挑最佳step。gallery可复制原图用于论文并记录完整选择来源。
SLAI新增独立环境安装预览/执行、分profile预检和离线资产核验；详细入口见
[PAPER_EVALUATION_WORKFLOW.md](PAPER_EVALUATION_WORKFLOW.md)。

72项轻量CPU测试通过，覆盖catalog、checkpoint、阶段生命周期、论文/图片导出、部署、
资产、专项适配、原batch评估、prompt导入和存储兼容。实际CLI测试确认删除训练与原始
prompt来源后仍能导出，并保留smoke标记。另有4项renderer检查通过、1项Z-Image tiny因
现有环境缺现代类而跳过；7项SDXL tiny测试通过。真实PEFT检查覆盖缺失/额外LoRA权重
拒绝，Z-Image拒绝Turbo名称/来源且兼容本地inventory revision。未在SLAI安装完整环境、
下载模型或运行真实GPU生成/评分；环境ABI、显存和数值效果需远程小面板smoke验证。

## 2026-11-11 四模型分阶段评估

在独立 `self_correct_dpo_eval` checkout 新增 `evaluate_suite.py`，提供list/plan/generate/
score/summarize/gallery。保留原SD1.5/SDXL配置与五指标尺度，接入Flow的SD3.5 Medium /
Z-Image Base renderer、固定UnifiedReward 2.0 ACS、官方GenEval和DPG。评分只消费完整
render，不依赖训练目录、adapter或生成模型；输入/权重/代码变化拒绝混用，展示选择不改变
统计分母。可用多个独立Python环境和已分配GPU分片执行，禁止运行时自动下载。

DPG读取保留全部CSV行，修正固定上游跳过首题的行为，独立记录协议名和分项局限。
真实固定source准备核验553条GenEval、1065条DPG及14392题；官方函数签名/聚合用CPU
fixture检查。新增checkpoint/阶段生命周期/评分资产/专项适配测试，旧batch、storage、
配置及数据检查回归通过；Flow tiny测试使用已有现代环境。没有安装环境、下载模型、
提交GPU或运行真实完整评分器。SLAI命令及依赖候选集中在command.md第一节。
不扩大旧AOSS源码白名单，旧输出仍用原checkout恢复，不更新运行中的训练代码。

## 2026-11-01 独立 checkpoint 评估分支

`feat/checkpoint-benchmark-eval` 基于 `9730dab`，单独 checkout 维护 `evaluate_batch.py` 和 `requirements-eval.txt`。默认九个已选 SD1.5 run，5000/6000/7000/9000 步，五指标与三套可选提示词；旧训练入口、配置、checkpoint 和 AOSS 兼容白名单不变。新分支只用于独立评估，使用新输出目录，不承担旧训练/监控的源码迁移。

11 项批量评估 CPU 测试与 2 项提示词导入测试通过，覆盖精确步数与 checksum、缺失/损坏区分、资源覆盖、实际双 worker CLI、五评分进程序列、增量汇总、恢复与失败记录、输入变更拒绝。实际下载并核验固定版本的 487/3200/1632 条官方提示词；完整 CLI 计划生成九实验×四步数×三面板的108行。9 项 AOSS 存储回归通过；其中历史迁移测试显式使用批准的 original/fixed SHA fixture，并断言新的 render SHA 不获自动放行。

依赖检查包括官方 CUDA wheel 元数据、非 GPU 依赖解析和 ImageReward BLIP 微型 CPU 前向。未安装完整 eval GPU 环境、未运行五套真实评分模型或远程5090评估。SLAI安装、两图smoke、运行及恢复命令见 [command.md](../command.md) 和 [评估文档](CHECKPOINT_EVALUATION.md)。

这份约定适用于 `self_correct_dpo`，以用户的双服务器工作流为准。更新日期：2026-10-25。

## 两台服务器的职责

- 本机维护单独开发分支，当前为 `dev/self-correct-workflow`，主要编辑代码、整理配置和做轻量 CPU 检查；当前目录不作为正式实验运行目录。
- 远程服务器使用 GitHub `main`，主要在已分配的 2 / 4 / 8 张 RTX 5090 上快速验证 SD1.5、SDXL 算法；按需要使用 H100。
- 新算法先做最小实现和小规模验证。确认可运行或有收益后，再将对应代码提交合入 GitHub `main`，远程拉取指定 commit。
- 正在运行或准备续训的实验固定代码、依赖、资源路径和超参数。先结束当前实验再更新该 checkout；并行比较不同版本可使用独立 checkout/worktree。
- Git 只管理源码、配置、文档。模型、数据、日志、checkpoint、W&B 密钥不进入仓库。每次实验使用独立输出目录，不覆盖旧结果。

## 开发风格与测试

优先简洁的研究代码：入口明确、按功能分组、默认值容易找到、命令可覆盖。不要为一个算法引入服务层、插件框架或复杂配置系统。

只保留能验证实际风险的少量检查：配置能否解析、B/C/K 的目标和梯度、保存及续训是否一致。复用现有测试，避免大量 mock、机械地照抄实现的测试或为可逆的小改动新建测试体系。正常修改不必反复跑全部测试。

新的噪声算法后续按实验需求再整理。本次只整理现有实现的导入位置，保留 `algorithms/paired_drift.py` 及原有实验修改，不将其作为默认入口。

## 资源路径只有一个入口

统一维护 `configs/resources.json`。远程机器可用同目录的 `resources.local.json` 覆盖少数值；该文件不进入 Git。优先级为：环境变量 > local 配置 > 资源配置。也可通过 `SCD_RESOURCES` 或训练命令的 `--resources` 选择另一份资源表。

| 资源 | 配置键 | 当前约定 |
|---|---|---|
| SD1.5 | `SD15_MODEL` | 沿用先前下载脚本路径 `${MODELS_ROOT}/stable_diffusion/sd15` |
| SDXL / 可选 VAE | `SDXL_MODEL` / `SDXL_VAE` | 预留原规划路径；用户尚未下载 SDXL |
| SD3 | `SD3_MODEL` | 预留 `${MODELS_ROOT}/stable_diffusion/sd3`；用户尚未下载 |
| SD3.5 | `SD35_MODEL` | 保留现有下载工具的资源项，独立于 SD3 |
| Pick-a-Pic 原始文件 | `PICKAPIC_RAW` | `${DATASETS_ROOT}/pickapicv2/data` |
| Pick-a-Pic 图片和清单 | `PICKAPIC_IMAGES` / `PICKAPIC_SD15_PREPARED` / `PICKAPIC_SDXL_PREPARED` | 放在同一数据集目录下；这些是规划的处理输出位置，不代表已生成 |
| HPDv3 数据集 | `HPDV3_RAW` / `HPDV3_SD3_PREPARED` | 为后续 SD3 训练预留；用户尚未下载 HPDv3 |
| 评分模型 | `PICKSCORE_*` / `HPSV2_*` / `HPSV3_*` / `IMAGE_REWARD_*` 等 | 沿用先前下载脚本的分类目录 |
| 固定评估面板 | `BENCHMARKS_ROOT` / `MONITOR_PROMPTS_SD15` / `MONITOR_PROMPTS_SDXL` 等 | 按模型系列保存处理后的 JSONL |
| 训练输出 | `RUNS_ROOT` | `/mnt/aoss/xietianyu/self_correct/runs`，须确认属于远程持久存储 |

按用户 2026-10-23 最终确认的安排，资源父目录 `WORKSPACE` 为 `/mnt/aoss/xietianyu`，`MODELS_ROOT` 为其下的 `models`，`DATASETS_ROOT` 为其下的 `datasets`，已去掉此前重复的目录层级。模型和数据集各自保留原子目录结构，下载、数据准备、训练和评估均使用同一份资源表；远程代码目录为 `/mnt/afs/L202500030/xietianyu/workspace/DPO`。`RUNS_ROOT` 继续使用表中的独立输出地址。本机负责代码与轻量检查；远程实际文件由部署检查和 GPU 小跑核验，训练不会自动下载模型。

环境变量和 `resources.local.json` 的优先级高于资源表。迁移后同步清理或更新旧覆盖项；之前在 shell 中导出过资源变量时，使用新终端加载迁移后的配置。

商汤大装置首次部署按根目录 `command.md` 的顺序，先验证纯 DPO 的实际训练、暂停及恢复，再开始正式基线和多视图比较。`scripts/check_deployment.py` 默认只检查 CPU 元数据，只有 `--gpu` 才实际运行 CUDA；数据处理只需要 CPU。平台申请及容器挂载通过用户实际使用的商汤入口完成，不套用本机 Slurm 命令。

HPDv3 是偏好数据集，HPSv3 是评分模型，分别记录。当前训练入口支持 **SD1.5/SDXL**；预留 SD3 路径及下载选项不等于已实现 SD3 训练。SDXL 下载后从 `.scd-download.json` 自动读取 revision；手动下载时用 `--model-revision` 指明可追溯的版本。

原始 Parquet 目录不能直接用作 `--data_path`：训练需要处理后的 `pairs.jsonl`。大规模处理放在远程已分配资源上，命令见根目录 `command.md`。

训练输入默认轻量抽查：配置检查与清单加载并行，清单仍核对 schema、split/来源及 ready 身份；图片默认8线程抽查8张并覆盖每个 split 的首对，最多14张。已删除全图片 inventory、`.validation.json` 缓存协议、缺缓存自动完整审计及入口重复的训练行检查。实际训练读取时校验即将解码字节的 SHA256；缺 SHA 的旧清单默认报错，不静默读取全部图片，需要显式 full 或重新准备含 SHA 的清单。`scripts/check_training_data.py --mode full --pairs ...` 保留独立完整审计。ACP 训练启动器不再自动执行逐卡 CUDA 自测，`check_deployment.py --gpu` 仅首次部署或排错时手动调用。具体日志与参数见 `command.md`。

## 配置和超参数

配置按四层合成：`configs/defaults.json` → `configs/models/sd15.json` 或 `sdxl.json` → 方法入口 → 命令覆盖。实际加载时方法入口的 `extends` 先读公共和 SD1.5 默认值，`--family sdxl` 再应用 SDXL 默认值。

| 方法入口 | 含义 |
|---|---|
| `configs/dpo.json` | 纯 DPO，K=1，无 rollout |
| `configs/multiview-B.json` | 各个独立 rollout 视图计算 DPO loss 后平均 |
| `configs/multiview-C.json` | 独立 rollout 视图的 margin 平均后计算 DPO loss；默认入口 |
| `configs/multiview-B-gaussian.json` | 无 rollout，普通高斯视图的 DPO loss 平均；K 个视图合批 |
| `configs/multiview-C-gaussian.json` | 无 rollout，普通高斯视图的 margin 平均后计算 loss；K 个视图合批 |

B/C 默认 K=2，`--views 3`、`--views 4` 或其他正整数均可。增大 K 会增加显存和计算量，不自动改变学习率或 batch。在 rollout 入口中仅将 K 改为1仍保留 rollout；普通高斯入口 K=1 则退化为单视图目标。明确选择纯 DPO 使用 `dpo`。

普通高斯入口使用 `training.batch_views=true`，复用同一批 VAE latent 和文本条件；按视图顺序拼接 noisy/timestep/target 以及 SDXL 的嵌套条件，一次参考前向、一次策略前向，再还原 `[B,K,2]` 计算联合 loss。其余已有入口沿用原逐视图路径。`batch_views` 不与 rollout/paired_drift 组合。普通高斯可用 `--set training.batch_views=false` 做逐视图实现对照；两种计算顺序可能产生浮点舍入差异。

显存按每卡 `2×B×K` 张带噪图片衡量，B 是图片对数。双卡普通高斯启动器支持 K=2/4/6/8：SD1.5 K=2 保留 B=2、累积64，K=4/6/8 默认 B=1、累积128，均保持全局256对；SDXL 均从 B=1、累积64、全局128对起步。显存足够可用 `SCD_TRAIN_BATCH_SIZE=2` 在新实验中改 B，启动器自动调整累积并核对冻结配置的全局 batch。默认 balanced 下，每次更新的图片对集合不随 B 的这种拆分改变，但随机数分配和浮点累加可能变化。BF16 和梯度检查点沿用默认设置；参考复用禁用 LoRA 的基座，无第二份 UNet。DDP 不合并各卡显存，大 K 是否可容纳需要远程实测。B 已为1后，增加梯度累积不能继续缩小一次合批前向。启动器只读取准备完成的数据，避免并行任务重复预处理；命令及显存监控方式见 `command.md`。

默认 seed=17、learning_rate=1e-5、LoRA rank/alpha=16、BF16、2000 优化步、每 200 步完整保存、每 1000 步保留 LoRA。SD1.5 默认 beta=5000、分辨率512、全局256对；SDXL 默认 beta=500、分辨率1024、全局128对。

全局 batch = 每卡 batch × 梯度累积 × GPU 数。只修改卡数时自动保持默认全局 batch；显式传入梯度累积时以命令为准。使用 `--print-config` 查看最终配置；使用 `--set training.weight_decay=0.01` 等覆盖其他嵌套参数。实际配置、代码摘要、commit、依赖版本、GPU、随机种子和资源身份记录在每次运行的 `training.json`。

新增 Gaussian 对照入口 `acp_gaussian_fixed_pairs.sh` / `acp_gaussian_fixed_views.sh` 使用用户确认的 SD1.5=384、SDXL=192 基准，支持 K=2/4/6 和2/4/8卡，保持10000步；旧启动器默认值继续保留。前者固定图片对组数，后者固定图片对视图数并将图片对组数除以 K，两组每个模型共用一个匹配新预算的纯 DPO 基线。`make_config.py --budget-unit pairs|pair_views --budget-per-update N` 生成 `training.comparison_budget` 并精确推导累积，不能与 `--effective-batch` 或显式累积混用，不可整除即拒绝。新比较启动器拒绝 CLI 训练覆盖，只放行暂停/恢复；预算、K、卡数和模式写入独立命名及冻结检查。`validate_config` 也核对预算字段与实际配置一致。该入口只支持普通高斯 B/C 及 DPO，不将 rollout 的额外计算混入等视图预算。

## Accelerate 和日志

- `python train.py ... --world-size N` 运行一个协调进程，默认用 Accelerate 启动各段 DDP 训练，在训练进程退出、显存释放后串行评估固定面板。参数覆盖先写成独立配置快照，再传给子进程。
- 直接 `accelerate launch ... train.py ... --no-evaluation` 也可运行 DDP；卡数从启动环境读取。需要自动评估时使用上面的协调入口，避免嵌套分布式启动。只支持单机，每卡一个进程。
- Accelerate 负责启动进程，训练精度与完整 checkpoint 由本项目实现管理；没有引入 DeepSpeed/FSDP 或替换为 `Accelerator.save_state`。
- TensorBoard 默认开启。`--report_to wandb` 或 `all` 可启用 W&B；先安装可选依赖 `pip install '.[tracking]'`。W&B 默认 offline，在线上报需 `--wandb-mode online` 并在远程自行登录。
- 日志包含 loss、margin、梯度范数、学习率、数据曝光、每步耗时、显存峰值及监控 PickScore。JSONL 总会保留。B/C 的训练 loss 数值不宜直接作为效果优劣依据，应比较相同 prompt、seed、生成设置下的指标。
- TensorBoard 续训会清理 checkpoint 后的旧事件。W&B 在线使用稳定 run ID 和 `optimizer_step` 横轴；离线每次启动会产生独立会话，需要自行同步，可能含被回滚区间的记录。

启动参数依据项目固定版本的 [Accelerate 0.34.2 官方说明](https://huggingface.co/docs/accelerate/v0.34.2/en/package_reference/cli)。

## 被驱逐与续训

1. 默认每 **200 个优化步**原子保存完整状态：LoRA、optimizer、LR scheduler、步数、数据曝光统计，以及每个 rank 的 Python/NumPy/PyTorch CPU/CUDA RNG。
2. 完整恢复槽使用独立临时目录，写完并校验后提交。`latest.json` 指向最近完整槽；先提交新指针再清理旧槽。恢复时先处理保存事务，再读取 latest。
3. 每 1000 步和最终步额外保留 LoRA。LoRA 快照只有权重，完整续训必须使用 `--resume_from_checkpoint latest` 或完整恢复目录。
4. TERM/INT 到来时，在完成当前优化步后尽量保存。断电、SIGKILL、突然驱逐无法保证保存当前步；恢复到最近完整提交点，常规最多重算上次保存之后不足 200 步的工作。若信号在首个 checkpoint 前且没有保存成功，则从头开始。
5. 续训保持相同 GPU 数、配置、代码、依赖和资源路径/身份。2/4/8 卡用于分别启动实验；当前不支持同一实验中途改变卡数。不要拉取新算法后直接续旧实验。
6. 完整 checkpoint 和输出必须在远程持久挂载上；容器本地临时盘被回收后无法靠代码恢复。调度器重新申请到资源后重新执行原命令并加 `--resume_from_checkpoint latest`；本工具不自动申请机器或重建容器。
7. 调整训练预算使用 `--extend-from` + 新输出目录（仅增加 max_steps，constant LR）；比较新算法使用新实验，不借续训修改算法。

## 本集群边界

遵守账号级 AGENTS 约定：只操作本人文件、环境、进程和 Slurm 作业；登录节点只做编辑与轻量 CPU 检查。GPU 或大规模处理进入 Slurm，先核实资源并按用户授权提交。预计向 `/irip` 写入超过 1GB 前检查磁盘；下载前检查共享模型/数据缓存，使用独立环境，不覆盖实验产物。

## 本次整理的验证范围

已使用项目现有 CPU 环境检查 B/C 的 K=1/2/3/4 梯度、SDXL 配置和小模型往返、200步保存及续训；额外短时检查了 Accelerate 两进程 CPU 启动、指针提交/清理中断后的恢复、TensorBoard 事件读写，以及各方法的命令覆盖。不新增复杂测试框架。

这些检查不代表 5090/H100 的实际训练、显存或性能已经验证。本机未安装 W&B 可选依赖，尚未测试其真实在线上报。远程首次使用新配置时，先按 `command.md` 进行两步 GPU smoke，再启动正式实验。

2026-10-24 新增普通高斯配置、视图合批和双卡 K=2/4/6/8 启动器。本次本地仅使用标准库验证配置生成、方法切换、全局 batch、冻结配置保护及 Python/Bash 语法；没有安装依赖。`tests/test_joint.py` 已补充普通高斯合批与逐视图的误差、梯度、RNG 和 SDXL 嵌套条件对照，但因本机没有 PyTorch 尚未执行。远程 `de9eed3` 的并行预处理代码及四项 real-Parquet 测试原样保留，本次没有重跑其依赖 PyArrow/Pillow 的测试。

2026-10-25 训练前校验优化：本地10项真实小文件校验测试、6项配置测试通过，覆盖并行审计、准备记录/缓存复用、未完成准备拒绝、清单和来源变更、重复 ID / split 泄漏、缺失或损坏图片、训练读取时 SHA256 验证。没有安装 PyTorch/PyArrow/Pillow；实际 GPU 启动耗时和模型训练仍须在 SLAI 验证。改动涉及 `pairs.py`，未完成的旧预处理应保留原 checkout 完成或恢复；已完成的 `ready.json` 数据无需重新准备。

2026-10-25 统一 ACP 入口：`acp_prepare_data.sh <pickapic|hpdv3> <sd15|sdxl|unfiltered>` 独立完成数据准备，已完成输出复用；只有 HPDv3 支持数据导出的 unfiltered。训练入口只读已准备数据，纯 DPO 双卡和普通高斯 K=2/4/6/8 共用 `acp_common.sh`，正式预算统一10000步；SD1.5 纯 DPO 和 K=2 保留每卡B=2、累积64，K=4/6/8 默认B=1、累积128，均为全局256。脚本从自身位置确定 checkout，环境和资源表一致。8卡旧入口保留。本地验证不安装环境，实际 GPU、显存和数据耗时仍须在远程验证。

入口整理验证：7项配置测试、10项数据校验测试通过，Bash语法通过；临时 checkout 中实际生成9种 DPO/高斯双卡配置，并检查冻结配置冲突、缺失配置保护、续训参数、已完成数据复用及 HPDv3 自定义路径转发。此 shell 检查替代了 Conda 激活、部署检查、预处理和训练调用，不代表实际环境或 GPU 执行已经通过。

2026-10-25 SDXL 开发：新增 `acp_sdxl.sh`，将公共启动生命周期扩展到2/4/8卡和 rollout B/C，已有双卡入口仍固定2卡并保留原命名。SDXL 固定1024、每卡默认1对、全局128对，正式预算10000步；沿用每200步完整保存和既有监控/续训。论文依据、研究配置与原论文差异见 `docs/SDXL.md`，运行命令集中于根目录 `command.md`。新增验证聚焦五种方法的配置及真实 tiny SDXL 普通高斯合批/逐视图在梯度检查点下的梯度一致性；完整1024模型的5090/H100训练仍需远程验证。

本轮结果：8项配置测试和7项 SDXL 测试通过；26组启动配置及恢复/冻结配置保护通过，后者替代了 Conda 激活、GPU预检和实际训练。验证环境和测试边界记录在 `docs/SDXL.md`。

2026-10-25 两套预算对照验证：11项配置测试通过，其中新增48组预算组合及非法预算/冲突/篡改拒绝；3项现有 CPU 保存恢复测试通过，并新增视图元数据与恢复后累计数断言。临时 checkout 的60组新启动配置、26组原启动配置及默认值/自动恢复/模式冲突/CLI覆盖保护通过；这部分替代 Conda 激活、GPU部署检查和训练调用。所有 Bash 语法、变更 Python 语法和 diff 空白检查通过。未提交 GPU 作业或实测5090/H100性能。训练代码摘要已改变，旧实验续训继续使用原 checkout，新入口使用新实验目录。

2026-10-25 启动检查精简：删除自动逐卡 CUDA 自测、重复训练行检查和图片校验缓存，配置/数据并行检查，默认最多14张图片并行抽查；不再隐式全量审计。10项真实小文件测试、临时清单/模型元数据的实际 `train.py --check-only`、变更 Python/Bash 语法及 diff 检查通过。保留已有训练预算和续训约束；本轮仅轻量 CPU 验证，未提交 GPU 作业或测量远程启动耗时。

2026-10-26 SLAI 分组脚本：`launchers/sd15/`、`launchers/sdxl/` 各包含8个独立训练入口和1个双卡并行评估入口，预算分别384/192，直接复用已有配置生成和续训。补齐监控面板专用的独立评估配置，默认评估编号01/02的1000步保留快照，不轮询或提交平台任务。临时 checkout 的16组真实 shell/配置生成、自动续训、监控配置继承、两模型双评估进程并行、GPU mask分配和子进程失败返回检查通过；Conda和实际训练/评估调用被替代。19个新shell语法、2个JSON解析和diff检查通过，没有执行GPU任务。逐组命令统一见根目录 `command.md`。

2026-10-26 RTX5090依赖迁移：用户选择官方稳定组合PyTorch2.14.1/cu130、torchvision0.29.1/cu130、NCCL2.30.7，远程驱动为580.95.05；requirements和包元数据同步更新，模型相关API版本保留。远程操作保留已有dpo环境，不提供自动删除环境、隔离NCCL或LD_PRELOAD覆盖；升级前清理cu12/cu13的旧NCCL包，避免共享库路径互相覆盖。Python3.10下pip元数据解析通过，共89个发行包，采用dry-run、PEP658元数据和受限ZIP范围读取，禁止整包下载，未安装任何依赖。11项现有配置测试、依赖声明语法与diff检查通过。用户已有NCCL2.31.2通信通过记录不代表新2.30.7组合已验证；仍需远程版本核对和双卡DPO一步训练/保存，命令见command.md。

2026-10-27 AOSS存储恢复：图片发布与pending事务恢复在硬链接不支持时使用render输出锁保护的重命名，保留已刷新PNG、journal、manifest校验及目标不覆盖检查。训练、监控、评估、render的源码兼容仅放行从b766801基线到本修复的精确SHA配对；不重写旧training/monitor/evaluation身份或checkpoint，另存兼容审计记录。`acp_resume_storage_fix.sh`从training.json恢复原配置与latest，部署至独立checkout以保护正在运行的实验。9项新增CPU测试和3项既有保存/恢复测试通过，覆盖不支持硬链接、发布前后中断恢复、不覆盖图片、非兼容错误、旧版四层身份、200步恢复后的优化器/RNG一致性、1000步快照保留及原200/500评估计划。临时checkout中的真实恢复脚本保留原配置/latest/暂停参数并拒绝训练覆盖，测试替代了Conda激活和训练执行；Bash语法、Python语法和diff检查通过。实际SLAI GPU续训仍须远程执行。

2026-10-30 Stagewise SD1.5入口：新增独立 `launchers/sd15_stagewise/` 四组，固定10000步、192pair/384pair-view，双卡B=2、累积48；旧入口不变。临时checkout中执行12组真实配置生成（四目标×2/4/8卡），确认10000步和累积48/24/12、预算及128prompt基座监控；自动latest续训、阶段边界/评估采样篡改拒绝、已有输出缺冻结配置拒绝、训练CLI覆盖和非法卡数拒绝通过。检查替代了远程Conda激活与实际训练调用，未安装依赖或提交GPU作业；新增5个Bash脚本语法、help及diff空白检查通过。实际5090双卡训练、峰值显存、性能及正式两步保存/恢复仍须在SLAI已分配容器验证。

本轮算法在项目既有 `.venv`（Python3.10.19、PyTorch2.6.0）上通过5项新增CPU检查及22项原配置/联合梯度/时间采样回归。覆盖非中点边界的均匀边际、段内独立时间与噪声、pair共享、RNG重放、四目标与梯度、合批/逐视图一致性和旧Gaussian随机序列保留。四个10000步解析配置与原C-K2固定384视图预算的配置逐项比较，仅 `method` 和新增 `training.stagewise` 不同。

另通过12项既有暂停/恢复与AOSS存储兼容检查；历史版本迁移测试改用明确的original/fixed源码身份fixture，生产兼容白名单保持不变，未批准的训练源码变化仍拒绝续训。独立临时CPU小模型检查调用真实MultiViewBackend采样、误差计算和训练循环：四种目标的日志loss均与独立公式计算一致且产生不同更新；连续4步与第2步暂停后恢复的权重、optimizer、scheduler、阶段指标和各CPU RNG完全一致。这些检查未加载完整SD1.5模型，不代表GPU训练或显存验证。

2026-10-30 发布前复核：39项CPU测试通过（stagewise 5、joint 6、config 11、timestep 5、pause 3、storage 9）；12组stagewise启动配置及冻结/自动恢复保护复核通过。四组正式配置与旧C-K2固定384视图预算配置相比，仅method和training.stagewise不同；四组smoke及benchmark配置均可生成，分别保持既有2步/100步且关闭评估的语义。另对四种目标各执行真实1000步CPU小模型训练循环和监控协调器：确认基座/200/500/1000评估调用、200/400/500/600/800/1000完整状态保存、1000步独立LoRA保留，以及500步评分中断后从latest继续并补完评分。该检查替代模型组件、图像输入、评分资产及评分器，未运行真实生成、PickScore或GPU；环境为既有Python3.10.19/PyTorch2.6.0。检查记录在本地logs/stagewise-schedule-review-20261030.json。原SD1.5/SDXL启动脚本、公共ACP脚本和默认配置相对fcbfb37未修改；command.md补充独立smoke流程，并将旧AOSS续训部署固定到fcbfb37，防止误用含新算法的分支版本。

## 2026-11-17 recovered qualitative workflow

Recovered the seven qualitative source files from the irip archive unchanged.
The existing four CPU workflow tests and Bash syntax check passed. No GPU
generation was performed. Existing unrelated research and GenEval edits were
left in place and excluded from this recovery commit.
