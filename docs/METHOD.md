# 方法

每个偏好对按 winner、loser 排列。两张图共享扩散时间与高斯噪声，使用各自的干净 latent；不同视图独立采样。每张图的 VAE posterior 只采样一次，在同一图像对的所有视图间复用。

## DPO 与 B/C

令每个视图的策略/参考误差为 latent 元素上的平均 MSE：

```text
m_k = (beta / 2) * [(e_ref_w - e_policy_w) - (e_ref_l - e_policy_l)]
B = mean_k softplus(-m_k)
C = softplus(-mean_k m_k)
```

K=1 时 B/C 相同。SD1.5 默认 beta=5000，SDXL 默认 beta=500。普通 DPO 使用 K=1 且移除 `independent_rollout`；仅将 K 改为1不会移除扰动。

策略使用 UNet LoRA；基座、VAE、文本编码器冻结。参考通过禁用 LoRA 得到，策略和参考使用相同状态、条件与标签。当前训练保留所有视图的策略计算图，对联合 loss 反传一次；每个 microbatch 除以梯度累积次数，DDP 对 rank 梯度取平均。一个 step 是完成累积后的一次全局参数更新。

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
