# Flow 本轮验证记录（2026-11-06）

工作树：`/irip_16t_0/xietianyu_2024/SemancticDPO/LMDPO/self_correct_dpo-flow`，分支`feat/flow-offline-20261106`，基线`c2cdb78c0403cc84d46b62f855750cd64b5ab83f`。源checkout的14份既有未提交文件SHA256全部保持原值；原值见`flow-baseline.json`，相对该dirty基线的本轮文件列表见`flow-changes.json`。未提交GPU作业、未下载模型、未安装依赖、未覆盖旧实验。

使用现有CPU环境`/irip/xietianyu_2024/tools/miniconda/envs/self_correct_dpo_slai23_cpu/bin/python`：Python3.10.22、torch2.3.0+cpu、diffusers.33.1、transformers4.45.2、PEFT.10、Accelerate.34.2。设置`OMP_NUM_THREADS=1`，最终受影响模块复测同时设置`MKL_NUM_THREADS=1`。

共 **86项不同测试通过**（合并以下分批运行，不重复计数）：

| 模块 | 项数 |
|---|---:|
| test_config | 20 |
| test_objectives | 8 |
| test_baseline_training | 6 |
| test_joint | 6 |
| test_stagewise | 5 |
| test_timestep_sampling | 5 |
| test_stage_pause | 3 |
| test_storage_compat | 9 |
| test_validation | 11 |
| test_h100_launchers | 3 |
| test_flow | 8 |
| test_hpdv3_sub | 2 |
| 总计 | 86 |

首轮12模块84项中81通过，3项历史AOSS迁移fixture把当前renderer源码当历史fixed SHA导致安全拒绝。将fixture明确固定到已批准的历史SHA（生产白名单未改）后，连同新增公共objective双向切换测试，以下38项通过：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=tests \
/irip/xietianyu_2024/tools/miniconda/envs/self_correct_dpo_slai23_cpu/bin/python -m unittest \
  test_config test_storage_compat test_flow test_hpdv3_sub -q
```

此时test_flow为7项；随后新增Qwen真实fast-tokenizer截断/padding mask/negative prompt审计边界并单独通过，使test_flow为8项、不同测试总计86。其余47项旧回归在首轮均通过。首轮命令为上面同一Python、`OMP_NUM_THREADS=1 PYTHONPATH=tests`，模块列表为表中全部12模块。完整临时控制台记录为`/tmp/self-correct-flow-tests.log`与`/tmp/self-correct-flow-final-tests.log`，它们不是持久实验产物。

新flow测试包括实际tiny PEFT Conv2D LoRA、生产train/AdamW/checkpoint循环。单进程与两rank CPU/Gloo均比较连续4步和2步暂停后恢复，policy LoRA、EMA shadow、Adam moments及各rank generator/CPU RNG逐位一致；EMA updates=4，每rank独立RNG。另验证reference无梯度、functional_call前后已有policy图梯度/Parameter身份不变、严格EMA状态校验、K1退化、普通/阶段采样的pair共享与view独立、joint/sequential梯度及RNG一致、Linear detached梯度与margin符号。ZImage native forward测试核对list shape、time反转及速度负号，未加载完整模型。

HPDv3-sub fixture使用100对32×32小图通过完整prepare、ready/manifest哈希、path1优选标签、relative paths和prompt组件train/validation/test隔离；没有全量解码真实HPDv3，也没有声称真实子集数量等于210008。

主代理独立CLI抽查：单卡DPO smoke、四卡MVPO K2/stage/pair_views128累积16、LR/seed/no-eval覆盖、ZImage Linear eta=.01/EMA=.995、相同tag重用与冲突拒绝，以及公共DPO/Linear双向objective切换，均通过。Python compileall、flow launcher Bash语法、git diff空白检查通过。

依赖仅检查官方PyPI元数据：目标八包相互约束、Python3.10要求、pyproject基础范围及optional flow pins一致；锁与来源保存在`requirements-flow.txt`和`flow-package-metadata.json`。没有安装/实际import新flow锁，不将其称作完整环境已验证。原requirements.txt和已验证GPU runtime不变。

真实SD3.5 Medium/ZImage Base的CUDA训练、双卡NCCL、1024显存/吞吐、FP32/BF16精度效果、周期生成与PickScore/HPS、新依赖完整安装兼容性仍需计算资源smoke。具体步骤统一见根`command.md`；本轮不提交正式训练。

## 效果优先入口追加验证

用户改为直接用正式配方快速看效果后，新增paper-pilot预设、SLAI单实验入口、独立mvpo-flow环境bootstrap及跨run效果汇总。之前分布式smoke矩阵不再作为开跑前置条件；新默认方案见`FLOW_QUICKSTART.md`。

本次仅运行与这些改动相关的检查：paper-pilot预算/评估时间表测试及原2/4卡预算测试通过；CLI实际冻结配置验证四卡MVPO仍为global128 pairs、800暂停和3200续跑使用完全相同的冻结配置；合成但具有真实complete/summary/score哈希的评估产物验证汇总paired delta、显式pair-view计算量以及不同seed拒绝。bootstrap无写入dry-run、参数拒绝、Bash/Python语法与git diff空白检查通过。新增一项持久测试，未重复宣称整套GPU集成已经验证。

环境安装和真实训练需要用户在SLAI执行，因为当前会话无法连接该服务器；没有在本地CPU环境冒充SLAI安装，也没有修改旧dpo环境。

六组并行入口追加检查：实际执行01至06六个shell入口的 `--config-only`（模拟Conda激活，使用现有CPU Python），验证两骨干/三目标映射、K、global pairs128、beta500、Linear EMA=.995/eta=.01、64提示词面板，以及六份独立配置。每个入口再用 `--stop-after-step 3200 --config-only` 检查冻结配置不变；所有检查及Bash语法通过，未启动训练。

## 3200步默认日程与速度配置（v3）

以上800步自动暂停记录属于旧入口；v3已取消自动暂停。用户确认评估为base及200、400、800、1300、1800、2300、2800、3200，LoRA保留点与之对应。

本轮根代理5项针对性CPU测试通过：paper-pilot三硬件预算/日程、条件文本去重还原、pair/view共享与joint/sequential梯度、真实tiny PEFT EMA、Qwen mask/截断；子代理4项新增once/schedule测试和9项旧storage/render/evaluation兼容测试通过。新增测试验证首次预检只执行一次、后续直接读快照、完整/暂停恢复与原数学路径权重逐位相同，以及协调器准确遍历八个目标。未重复运行分布式测试矩阵。

实际执行3个四5090 shell命令与6个双H100 shell入口的config-only（模拟Conda激活），验证9份独立配置：SD35四5090 B1/accum32，SD35双H100 B4/accum16，ZImage双H100 B2/accum32；全部global128 pairs、3200步、模型常驻、preflight once。直接捕获正式入口启动命令，确认默认不带stop-after-step。Bash语法和git diff空白检查通过。

移除重复清单/资产扫描、消费图片SHA、逐pair曝光诊断、逐microbatch逐指标CPU同步及EMA逐参数检查；保留首检、配置一致、非有限梯度失败与原子checkpoint。文本条件按每个microbatch去重，不引入全数据缓存。日志新增每update秒数、pairs/s与峰值GiB。

SD35约15.3GiB静态权重来自本地safetensors header及目标dtype计算，排除了重复fp16变体；不是GPU峰值或吞吐实测。新默认B尚未在5090/H100上执行，独立环境安装仍由目标服务器完成。本轮未提交GPU任务，也未修改旧dpo环境。

## Winner-only SFT与效果比较（v4）

新增两骨干SFT配置及07/08双H100入口，四5090 SD35入口也支持sft。沿用独立flow环境、3200步、既定八点评估和首次预检复用。SFT从基座独立训练，仅监督winner的flow速度MSE，不使用loser/reference/EMA，不修改其他方法的目标。

根代理实际执行三个SFT shell入口的config-only（模拟Conda激活，真实run_flow配置生成），确认SD35双H100 B4/accum16、ZImage双H100 B2/accum32、SD35四5090 B1/accum32；均128张winner/update、K1、reference none、3200及八点评估。

新增2项汇总CPU测试通过：使用带真实完成记录/摘要/分数哈希的合成评估，验证SFT分类与winner预算、相对base和同一步DPO的配对差值/区间、参考步缺失留空、不同训练seed不混合。工具读取已有产物，不增加训练阶段检查。

Z-Image可学习性分析基于官方模型卡、Linear-DPO论文及本项目筛选代码，详见ZIMAGE_HPDV3_ASSESSMENT.md。未进行真实Z-Image生成评分、全量图片质量分析、SLAI安装或GPU训练；不能将CPU通过或论文在SD3-M上的结果称为Z-Image有效。

子代理4项SFT相关CPU检查通过（test_flow_sft的3项，加既有SD15/SDXL winner-only回归1项）：两个flow骨干的静态t/sigma、VAE scale/shift、FP32速度MSE及梯度/RNG正确；loser填NaN不影响结果且仅执行一次winner预测；实际tiny CPU训练读取winner PNG、不存在的loser路径不被读取，连续2步和暂停恢复的optimizer状态逐位相同，reference_state为空，policy/reference计数为2/0。SFT训练日志已明确winner图像数/吞吐，公开偏好pairs/pair-views计数置0，保留采样行数用于data_passes。Python/Bash语法及diff检查通过。

## 本地A800启动与固定完整checkpoint

新增full_checkpoint_every=400：每400步保留固定完整状态，latest每200步更新；固定点可直接充当latest避免双写，LoRA评估保留点不变。3项针对性CPU测试通过，包括400固定→600滚动→800固定事务中断恢复、旧journal兼容及原LoRA路径。环境元数据放宽到实际支持的torch>=2.3,<3，bootstrap安装完整项目依赖（含tensorboard）且精确约束源GPU runtime。

本地首次准备作业65760（CPU2/48GB）与两个依赖GPU作业65761/65762（各2A800/CPU8/96GB）已提交。该轮首次安装后来失败，详情见下一段；这里不将提交作业等同于训练成功。离线包来自本地源环境dry-run解析，原始模型/数据和code源环境保持不变。

65760实际退出1：Conda克隆已复制pip文件，但随后尝试联网获取Conda包，计算节点无法连接登录节点的localhost代理。65761/65762因afterok依赖失败自动取消，没有开始GPU训练。改为Conda与pip均显式离线，保留已复制文件，用源环境的110包explicit清单完成缺失的Conda安装；包缓存均存在。恢复准备65765（CPU2/48GB/4天）和训练65766/65767（各2A800/CPU8/96GB/4天）已重新提交，训练依赖afterok:65765。提交时准备作业为Priority等待；修复尚待实际计算节点运行验证。Bash语法与diff检查通过。旧日志和提交记录保留。

用户明确要求Conda安装在登录节点tmux执行。取消尚未运行的65765，启动tmux会话flow-env-a800；数据准备改为65768（原CPU资源），环境安装成功后自动释放。保留训练65766/65767并将依赖改为afterok:65768。prepare_a800.sbatch已移除环境安装；setup_a800_env.sh为独立tmux安装入口。Conda的CONDA_PKGS_DIRS实际使用逗号分隔（已通过conda config --show pkgs_dirs确认），不能使用冒号。

用户进一步授权数据处理在manager tmux执行。65768尚未运行即取消；训练65766/65767先hold并清空Slurm依赖。新增prepare_a800_data.sh，tmux flow-data-a800使用只读code Python、nice10、OMP/MKL2并行执行提取/完整筛选/面板冻结；数据成功后等待环境bootstrap status=complete，再release两组训练。当前环境Conda已过准备/校验，执行文件安装阶段；进程wchan为rpc_wait_bit_killable，等待NFS文件操作，尚未进入pip安装。数据任务已启动。

用户追加提交Z-Image LinearDPO。train_zimage_a800.sbatch放行linear_dpo参数，使用既有zimage-linear_dpo配置和paper-pilot覆盖（B2/accum32/global128，K1，EMA=.995，eta=.01，3200步、400固定完整/200 latest及八点评估）。作业65770：cluster-1、2A800、8 CPU、96GB、4天，依赖after:65766，DPO开始即可调度，无需等待其完成。这样复用已在tmux中的准备控制，不更改运行中的数据脚本。Bash语法与diff检查通过，GPU训练尚未开始。
