# 换容器后正式数据验证报告的文件信息不一致

错误位置：load_preflight → load_certified_pairs → data_receipt → check_binding，
`Validated file changed; regenerate its report: .../pairs.jsonl`。
这发生在模型构造之前，不能由此判断为CUDA/OOM或conda激活失败。
单卡探针用独立debug偏好清单，其通过不等于正式prepared-sub-flow-v2凭据已通过。

用户确认数据未修改，仅更换SLAI任务/容器；挂载的mtime显示差异是一种可能原因，
必须比较原报告与当前文件的实际SHA256才能确认内容身份。
本次只新增CPU诊断/文件信息绑定工具，不修改训练器、原数据、原报告或运行中的配置。

## 内容校验与新报告

在发生错误的SLAI容器中运行，复用mvpo-flow环境。
工具对pairs.jsonl、ready.json、audit.json计算SHA256，并检查哈希期间文件未变化。
三者必须与原passed报告完全一致，且ready的pairs摘要/source revision与报告一致，才允许创建新报告。
新报告保留原审计范围/数据身份，记录旧报告SHA256、原/新stat、容器hostname与时间；它不是重新做数据审计。
不扫描图片、读取图像、重下数据或加载模型，不覆盖原报告。

只读诊断：

```bash
python scripts/rebind_flow_data_receipt.py \
  --pairs /mnt/aoss/xietianyu/datasets/hpdv3/prepared-sub-flow-v2/pairs.jsonl
```

只有stat变化且SHA256全部相同，才按下面生成新的本容器报告并重启失败项。
同一内容与stat下重复执行会复用已存在且验证一致的新报告，不覆盖它；已有不相关或失效输出会拒绝。

```bash
source /mnt/afs/L202500030/xietianyu/miniconda3/etc/profile.d/conda.sh
conda activate mvpo-flow
cd /mnt/afs/L202500030/xietianyu/workspace/DPO-flow-trainonly
git pull --ff-only

# DPO任务用dpo；若另一个Stage K2任务也报相同错误，在其容器中改为stage-c-k2。
flow_case=dpo
flow_receipt="$PWD/reports/slai/data-${flow_case}-stat-v1.json"

python scripts/rebind_flow_data_receipt.py \
  --pairs /mnt/aoss/xietianyu/datasets/hpdv3/prepared-sub-flow-v2/pairs.jsonl \
  --out "$flow_receipt" &&
bash launchers/zimage/run_suite.sh "$flow_case" \
  --profile h100 --batch-size 4 \
  --data-report "$flow_receipt" --run-tag flow-b4-stat-v1
```

两个容器分别绑定各自所见stat，报告文件名按case区分；manifest内容身份仍是同一SHA256。
新data-report路径会改变冻结配置，故使用新tag，保留此前失败的flow-b4-v1配置和记录。
这次失败发生在preflight内，日志所示尝试尚未进入模型加载或优化更新。
已正常训练的任务保持原命令和报告，不需要切换。

如果输出`SHA256 differs`，说明当前读到的字节与原审计报告不一致，工具非零退出，后面的训练不会执行。
此时先核对容器挂载和准备产物，再决定是否对新数据运行正式audit；不直接改mtime、旧报告或关闭校验。
若准备产物仍在被写入，工具也会拒绝绑定。
