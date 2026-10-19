# 方法与实现边界

对每个视图，winner/loser 共享时间与原始高斯噪声，各自保留自己的干净 latent。策略与禁用 LoRA 的冻结参考使用完全相同的状态与监督标签。MSE 按 latent 元素平均，margin 为：

```text
m_k = (beta / 2) * [(e_ref_w - e_policy_w) - (e_ref_l - e_policy_l)]
ell(m) = softplus(-m)
B: mean_k ell(m_k)
C: ell(mean_k m_k)
```

SD1.5默认beta=5000，对应MSE差的系数2500；SDXL模板参考Linear-DPO脚本使用beta=500，对应系数250。当前训练入口支持SD1.5/SDXL，只保留B/C，默认C；K=1时两者均为单视图DPO。D已停用，定义只作为旧实验来源记录。B/C均使用所有视图的完整梯度。

B/C 当前保留所有视图的策略计算图，每个视图只做一次 loss 所需的策略前向，然后对联合 loss 做一次反传。C 先在同一图片对内平均 margin，再计算 softplus；B 先计算各视图 softplus，再平均。每个 microbatch 的 loss 除以梯度累积次数，累积完成后仅更新一次参数；K 不额外增加 step。

旧实现先无梯度评分确定 sigmoid 权重，再逐视图重算策略并立即反传。固定 detached 状态、标签并保持重算确定性时，两种实现的参数梯度等效；当前实现省去每个视图的一次无梯度策略评分前向，代价是同时保留多份计算图。各视图的 UNet 前向仍逐个执行，没有自动合并成更大的 batch，也不承诺 K=2 与 K=1 耗时相同。LoRA dropout 为0，参考禁用 adapter，不复制第二个基座。VAE 与文本每个图像池只编码一次，rollout 与参考仍不反传。

rollout 设总训练时间索引为1000，u均匀取[53,1000)，s=u−33，r条件均匀取[s,1000)。33是一次 DDIM 步跨越的索引间隔，非33次网络预测。

上述历史采样对应 `independent_rollout.timestep_sampling="original"`（默认，保持原随机流）。设置 `"uniform_final"` 时，先均匀采样最终训练时间 r，再按原有条件分布采样 s：

```text
r ~ Uniform{min_anchor_t, ..., T-1}
q(s | r) ∝ 1/(T-s), min_anchor_t <= s <= min(r, T-1-stride)
u = s + stride
```

它只均匀化 r 的边缘分布，保留给定 r 时原有的 rollout 状态构造；不添加 loss 权重。K2 的两个视图独立抽样，每个视图内部 winner/loser 共享时间与噪声。新模式记录 `[min_anchor_t,T)` 十个时间区间的采样比例。默认参数为 T=1000、min_anchor_t=20、stride=33，覆盖 r=20…999；oracle 在这个模式下退化为该支持集上的均匀前向加噪。

```text
x_u = alpha_u*x0 + sigma_u*epsilon
epsilon_hat = current_policy(x_u, u, caption)      # no_grad
x0_hat = (x_u - sigma_u*epsilon_hat) / alpha_u
z_s = alpha_s*x0_hat + sigma_s*epsilon_hat         # eta=0，不裁剪
x_r = (alpha_r/alpha_s)*z_s + (sigma_r-alpha_r/alpha_s*sigma_s)*epsilon
epsilon_star = (x_r - alpha_r*x0) / sigma_r
```

状态与标签用FP32构造并停止梯度；v prediction 使用等价转换。训练 rollout 只支持 CFG=1。oracle 用真实 epsilon/v 替代 rollout 预测，因此退化为相同 r 上的普通前向加噪。历史 original 模式下 r 的边缘分布不均匀，oracle 不能称为原普通 DPO；uniform_final 模式仍须注明其时间支持集为20…999。

模型扰动后各图以自己的原图为标签，没有跨正负图拼接，也没有额外 winner 重建损失。DPO 可通过提高 loser 误差改善 margin；不保证 winner 绝对重建误差下降。完整梯度仅针对固定 detached 状态，不包含状态采样分布对参数的导数。

固定 `dpo-diagnostic.jsonl` 使用普通均匀噪声分布，训练 metrics 使用对应方法分布；两者不可混作同一 loss 曲线。相同步数、相同图片对曝光量不等于相同 GPU 成本。

SDXL使用同一VP扰动/标签公式，额外传入两个CLIP的token embedding、第二个编码器的pooled embedding及六维time_ids；它们在状态构造、策略和参考中保持一致。VAE缩放为0.13025，训练编码FP32，文本/UNet按训练dtype执行。组件、默�