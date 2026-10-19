# SD1.5 与 SDXL

同一套训练、扰动、B/C损失、checkpoint和评估流程支持两种模型，通过 `model.family=sd15/sdxl` 选择。原 SD1.5 主模板保持不变，省略 `make_config.py --family` 仍选择 SD1.5；SDXL 使用独立模板。两者当前均只启用 B/C，默认 C，支持 K=1/2、model/oracle 扰动和移除 `independent_rollout` 后的普通多噪声视图。

## 组件与默认值

| 项目 | SD1.5 | SDXL |
|---|---|---|
| 基座 | Stable Diffusion v1.5 | `stabilityai/stable-diffusion-xl-base-1.0` 的完整本地 Diffusers pipeline |
| 训练模板 | `configs/model-K2-C.json` / B | `configs/sdxl/model-K2-C.json` / B |
| 默认分辨率 | 512 | 1024 |
| 文本条件 | 一个 CLIP，768维 | 两个 CLIP 的倒数第二层拼接为2048维；第二个编码器的1280维 pooled embedding |
| 附加条件 | 无 | `[H,W,0,0,H,W]` 六个 time_ids |
| VAE latent 缩放 | 来自原 VAE 配置 | 0.13025；训练使用 FP32 |
| 可训练部分 | UNet attention LoRA，rank=alpha=16 | 同左；双文本编码器、VAE和UNet基座冻结 |
| 训练 dtype | BF16 | BF16 |
| batch × accumulation | 8 × 32 = 256对 | 1 × 128 = 128对 |
| 学习率 / beta / warmup | 1e-5 / 5000 / 200 | 1e-5 / 500 / 0 |
| 步数 | 2000 | 2000 |
| 监控生成 | DDIM30、CFG7.5 | Euler30、CFG5.0 |
| 独立评估 | PNDM50、512px、CFG7.5 | Euler50、1024px、CFG5.0 |

SDXL 的双编码器、time_ids、可替换VAE，以及优化尺度参考 [Linear-DPO 的 SDXL 脚本](https://github.com/Whynot0101/Linear-DPO/blob/663179c7adbbbd2d77b97b5841534447eb291ebd/run_sdxl_pickapic_linear.sh) 和 [训练实现](https://github.com/Whynot0101/Linear-DPO/blob/663179c7adbbbd2d77b97b5841534447eb291ebd/train/train_sd_dpo.py)。上游脚本用16进程×累积8、FP16和完整UNet训练；这里默认单进程累积128，也支持2/4/8卡DDP并保持全局batch128，使用BF16、LoRA及本项目B/C目标，没有启用上游 Linear-DPO loss、在线重标偏好或EMA参考。有效batch相同不代表GPU成本或优化过程相同。

图像仍按现有数据契约缩放为正方形，无随机裁剪，故 time_ids 使用目标尺寸和零裁剪偏移。两个 tokenizer 都检查训练文本是否超过77 tokens，评估则按显式配置截断并记录。策略、冻结参考、扰动预测使用同一套完整条件；pooled embedding 和 time_ids 随 token embedding 一起停止梯度。参考通过禁用LoRA获得，不额外复制SDXL基座。

## 本地模型与可选 VAE

只支持标准 SDXL base 的双编码器、四latent通道、`text_time` UNet。CPU预检核对2048维 cross attention、2816维附加投影输入、双编码器配置、VAE和scheduler；refiner、inpainting和蒸馏模型没有对应训练/采样协议，不属于本模板支持范围。

完整pipeline需包含：

```text
sdxl-base/
  model_index.json
  unet/                 # config 与本地权重
  text_encoder/         # CLIPTextModel
  text_encoder_2/       # CLIPTextModelWithProjection
  tokenizer/
  tokenizer_2/
  vae/
  scheduler/
```

默认使用pipeline内置VAE。要对应Linear-DPO的 `madebyollin/sdxl-vae-fp16-fix`，通过 `--vae` 与 `--vae-revision` 显式指定已有本地 AutoencoderKL 目录。目录须有 `config.json` 和权重。这里的VAE编码始终FP32；生成按评估dtype与Diffusers原生decode逻辑执行。VAE资产纳入训练、生成和评估身份；adapter记录VAE版本，替换版本或从外置VAE切回内置VAE时会拒绝混用。

路径和revision格式：

```json
{
  "model": {
    "family": "sdxl",
    "path": "/your/models/sdxl-base",
    "revision": "verified-base-revision",
    "vae": {
      "path": "/your/models/sdxl-vae-fp16-fix",
      "revision": "verified-vae-revision"
    }
  }
}
```

使用内置VAE时省略 `model.vae`。SDXL模板的base revision是占位符，必须以实际本地模型的来源版本替换；不要将当前远程main版本当成已下载权重的版本。训练器、评估器均不自动下载模型。

## 配置与运行

在独立仓库根目录，使用已准备的模型、数据和评分权重：

```bash
export MODEL_PATH=/your/models/sdxl-base
export PAIRS_PATH=/your/data/pickapic/pairs.jsonl
export MONITOR_PROMPTS="$PWD/data/evaluation-v1/pickapic-monitor.jsonl"
export SCORERS_CONFIG="$PWD/configs/scorers.json"
# 继续按 EVALUATION.md 设置 PickScore 本地路径。

python scripts/make_config.py --family sdxl \
  --model "$MODEL_PATH" --model-revision '<verified-base-revision>' \
  --pairs "$PAIRS_PATH" --monitor-prompts "$MONITOR_PROMPTS" \
  --scorers-config "$SCORERS_CONFIG" --out configs/local-sdxl-C.json
python train.py --config configs/local-sdxl-C.json --out runs/sdxl-C-001 --check-only

# 已分配GPU后执行；不会创建Slurm、CCI或ACP资源。
bash launchers/train.sh configs/local-sdxl-C.json runs/sdxl-C-001
```

需要外置VAE时给 `make_config.py` 加上：

```bash
--vae /your/models/sdxl-vae-fp16-fix --vae-revision '<verified-vae-revision>'
```

B方法使用 `--arm B`；可用 `--batch-size` 与 `--accumulation` 调整显存并保留有效batch。`--resolution` 同时调整训练和监控的正方形尺寸；改变分辨率必须作为独立实验记录。`--smoke` 保留模型默认分辨率，改成batch1/累积1、两步、关闭监控，用于实际GPU集成检查：

多卡增加 `--world-size 2`、`4` 或 `8` 并使用 `launchers/train_ddp.sh`；完整流程与恢复条件见[DISTRIBUTED.md](DISTRIBUTED.md)。

```bash
python scripts/make_config.py --family sdxl \
  --model "$MODEL_PATH" --model-revision '<verified-base-revision>' \
  --pairs "$PAIRS_PATH" --smoke --out configs/local-sdxl-smoke.json
bash launchers/train.sh configs/local-sdxl-smoke.json runs/sdxl-smoke-001 --stop-after-step 1
bash launchers/train.sh configs/local-sdxl-smoke.json runs/sdxl-smoke-001 \
  --resume runs/sdxl-smoke-001/checkpoints/latest.json
```

监控仍在第0、200及每500步和最终步运行PickScore；每200步滚动保存最新LoRA与完整续训状态，每1000步及最终步独立永久保存LoRA和图像，其他点评分后保留分数和清单。200步滚动保存不额外生成图像或评分。SDXL的1024px成本更高，128条prompt仍需完整生成，不因模型切换改变监控子集。继续训练使用相同配置和 `--resume`。

## 任意保留步数的评估

```bash
export HPDV2_PROMPTS="$PWD/data/evaluation-v1/hpdv2-test.jsonl"
python evaluate.py --config configs/evaluate-sdxl.json \
  --run runs/sdxl-C-001 --step 1000 --out runs/sdxl-hpdv2-1000 --check-only
python evaluate.py --config configs/evaluate-sdxl.json \
  --run runs/sdxl-C-001 --step 1000 --out runs/sdxl-hpdv2-1000 \
  --execution standalone --run-label slai-sdxl-hpdv2
```

默认仍是HPDv2与四项指标，其他数据集和HPSv3/ImageReward仍通过 `--benchmarks`、`--metrics` 按需启用，见[EVALUATION.md](EVALUATION.md)。基座使用 `--step 0`。评估从训练记录恢复相同基座与VAE，SDXL配置的 `model_family` 可防止误用于SD1.5。生成器支持Euler与DDIM，默认Euler来自SDXL base的采样路径。

## 验证范围

已使用现有环境中的Diffusers0.33.1、Transformers4.53.3、PEFT0.10.0，在CPU上以随机初始化的小型双CLIP、UNet、VAE验证真实条件编码、普通/扰动前向、LoRA梯度、参考冻结、adapter保存加载和完整SDXL pipeline生成小图。没有下载权重、安装依赖、提交训练作业或修改旧实验；SD1.5主模板保持原内容。

本地通用模型目录未发现SDXL权重，因此未验证真实SDXL的1024px显存、BF16训练、GPU暂停恢复或真实奖励评分。部署后先在实际已分配GPU运行两步检查，再开正式训练。此版本源文件发生变化，旧代码版本的运行目录继续使用其冻结代码；新SDXL实验使用新输出目录。
