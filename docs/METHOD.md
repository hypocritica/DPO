# 方法

算法边界、历史证据与下一轮消融顺序见 [算法审查与逐轮改进计划](ALGORITHM_REVIEW_AND_PLAN.md)。

每个偏好对按 winner、loser 排列。两张图共享扩散时间与高斯噪声，使用各自的干净 latent；不同视图独立采样。每张图的 VAE posterior 只采样一次，在同一图像对的所有视图间复用。

## DPO 与 B/C

令每个视图的策略/参考误差为 latent 元素上的平均 MSE：

```text
m_k = (beta / 2) * [(e_ref_w - e_policy_w) - (e_ref_l - e_policy_l)]
B = mean_k softplus(-m_k)
C = softplus(-mean_k m_k)
```

B/C 支持任意正整数 K，默认 K=2，可通过 `--views 3`、`--views 4` 等设置。K=1 时 B/C 相同。SD1.5 默认 beta=5000，SDXL 默认 beta=500。普通 DPO 使用 K=1 且移除 `independent_rollout`；在 rollout 入口中仅将 K 改为1不会移除扰动，普通高斯入口则始终没有 rollout。

策略使用 UNet LoRA；基座、VAE、文本编码器冻结。参考通过禁用 LoRA 得到，策略和参考使用相同状态、条件与标签。当前训练保留所有视图的策略计算图，对联合 loss 反传一次；每个 microbatch 除以梯度累积次数，DDP 对 rank 梯度取平均。一个 step 是完成累积后的一次全局参数更新。

普通高斯多视图入口为 `multiview-B-gaussian` / `multiview-C-gaussian`，不含 `independent_rollout`。每个视图独立采样普通均匀时间与高斯噪声，pair 内仍共享时间/噪声。`batch_views=true` 将 K 个 `[B*2,C,H,W]` 输入合成 `[K*B*2,C,H,W]`，复用并复制匹配的文本条件，分别执行一次参考和策略前向，再还原 `[B,K,2]` 的误差；B/C 公式不变。算法支持任意正整数 K，双卡启动器提供 K=2/4/6/8，默认 rollout 入口仍保留原实现。

合批减少调用开销，UNet 的图片计算量仍随 K 增长。降低每卡图片对数 B、增加梯度累积可控制单次显存并保持全局 pair batch；不能把视图数 K 乘进图片对曝光计数。同一全局 pair batch 下，K=4 仍比 K=2 处理更多带噪图片，不能视为同算力预算。

## 按噪声阶段采样的 Gaussian C（四组消融）

`stagewise-C-gaussian` 独立于原有全域 iid-t 的 Gaussian C，默认 K=2、`batch_views=true`，无 rollout。每个 pair 先按区间长度概率选择阶段：低噪声 `[0,500)` 或高噪声 `[500,1000)`（各 1/2）。同一个 pair 的 K 个视图属于同一阶段，各视图在该阶段内独立采样 t 和 ε；winner/loser 内共享该视图的 t 和 ε。这不是 K 个视图共享一个精确 t。各视图的边缘 t 分布仍全域均匀，但视图的阶段相关性不同于旧全域 iid-t；因此第一组也是新的阶段采样对照。

令 `m = mean_k m_k`、`D(m) = softplus(-m)`、`H(m) = Huber_delta=1(m-1)`（均先聚合 margin 再计算 loss）。四组为：

| 入口 | objective | 低噪声 | 高噪声 |
|---|---|---|---|
| `01_stage_dpo.sh` | `dpo` | D(m) | D(m) |
| `02_stage_huber.sh` | `phase` | 0.5 H(m) | D(m) |
| `03_constant_mix.sh` | `constant` | 0.5 D(m) + 0.25 H(m) | 0.5 D(m) + 0.25 H(m) |
| `04_reverse.sh` | `reverse` | D(m) | 0.5 H(m) |

Huber 在 `|m-1| <= 1` 时为 `0.5*(m-1)^2`，否则为 `|m-1|-0.5`。`huber_weight=0.5`；constant 对 DPO 与已加权 Huber 各取一半，不额外叠加未加权 Huber。边界、目标 margin、Huber delta/weight 与 objective 写入 `training.stagewise` 并冻结。

四组均从同一 SD1.5 基座开始新训练，不从旧 DPO LoRA 续训；10000 个优化步、全局192 pair/384 pair-view，默认双卡每卡2 pair、累积48、seed17。其余优化与128 prompt监控沿用既有设置。配置和暂停/续训命令见根目录 `command.md`。

## 停止梯度的 DDIM 扰动

```text
x_u = alpha_u*x0 + sigma_u*epsilon
epsilon_hat = current_policy(x_u, u, caption)
x0_hat = (x_u - sigma_u*epsilon_hat) / alpha_u
z_s = alpha_s*x0_hat + sigma_s*epsilon_hat
x_r = (alpha_r/alpha_s)*z_s + (sigma_r-alpha_r/alpha_s*sigma_s)*epsilon
epsilon_star = (x_r - alpha_r*x0) / sigma_r
```

以上为 epsilon 参数化；v-prediction 使用等价转换及 `v_star = alpha_r*epsilon_star - sigma_r*x0`。状态和有效标签以 FP32 构造并停止梯度，rollout 仅支持 CFG=1。模型使用各张图自己的 x0，不交换正负标签。这里的完整梯度针对固定的 detached 状态，不包含状态生成分布对参数的导数。

`state_source=oracle` 用真实 epsilon/v 替代 rollout 预测，会退化为相同最终时间 r 上的普通前向加噪。

## 时间采样

`original`：u 均匀取 `[stride+min_anchor_t,T)`，s=u-stride，r 条件均匀取 `[s,T)`。默认 T=1000、stride=33、min_anchor_t=20；stride 是一次 DDIM 跨越的索引间隔。此时 r 明显偏高噪声，original-oracle 不是标准均匀时间 DPO。

`uniform_final`：r 均匀取 `[min_anchor_t,T)`，再按 `q(s|r) ∝ 1/(T-s)` 采样 s，u=s+stride。默认支持20…999，仍需与标准 DPO 的0…999区分。两种模式不增加额外 loss 权重。

## 比较口径

C 的 loss 可能因聚合方式而天然低于 B；DPO margin 改善也可能来自 loser 误差提高。比较效果使用相同固定 prompt 和采样设置的生成指标。相同步数、图片对曝光和全局 batch 不等于相同计算成本。

目标形式参考 [DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO)，SDXL 组件和优化尺度参考 [Linear-DPO](https://github.com/Whynot0101/Linear-DPO)。本项目没有实现 Linear-DPO loss。已有实验定义见 [EXPERIMENTS.md](EXPERIMENTS.md)。
