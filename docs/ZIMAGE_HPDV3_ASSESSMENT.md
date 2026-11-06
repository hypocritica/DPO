# Z-Image Base 能否从 HPDv3-sub 学到有效偏好

结论：值得做，优先把DPO作为是否存在可迁移偏好信号的基线，把SFT作为优选图模仿的对照，再检验MVPO是否优于DPO。当前证据支持开展实验，尚不能证明Z-Image生成质量会提升。这里没有真实Z-Image训练曲线或生成评分；源码/CPU验证不作为效果证据。

## 已有证据与本项目的边界

- [Z-Image官方模型卡](https://huggingface.co/Tongyi-MAI/Z-Image)把Base描述为非蒸馏、适合LoRA继续训练的模型；对照表中Base的RL标记为否。当前使用Base，不能将Turbo的对齐状态或表现直接套过来。
- 本项目`prepare_hpdv3_sub.py`要求两侧来源均在`flux/kolors/sd3/hunyuan/real_images`内，保留原始`path1`优选标签，剔除官方测试prompt；完整准备还隔离测试图片。这些来源不含Z-Image，因此是离线跨模型分布迁移。真实训练数量以prepared清单为准，不直接套论文210008。
- [HPDv3官方说明](https://github.com/MizzenAI/HPSv3#pairwise-training-data-format)规定`path1`为优选图。`confidence`是比较可靠性，不是图片绝对质量分，也不是Z-Image能否获益的概率。
- [Linear-DPO附录E.2表3](https://arxiv.org/html/2605.21123v1#A5.SS2)中，SD3-M在HPDv3-sub上的SFT使HPSv3下降，而DPO/Linear-DPO上升。这是“模仿优选图”和“学习偏好差异”可能产生不同结果的实证例子；它没有验证Z-Image、我们的LoRA设置或MVPO。

本地仅查看了原始JSON的字段及模型卡，没有解码全量图片、统计全量子集构成或比较Z-Image与数据优选图。元数据里的模型名不能替代图像质量比较。

## 为什么仍可能有效

Z-Image即便总体能力较强，仍可能存在构图、细节、美感、文本遵循等维度的偏好缺口。DPO使用同prompt两张图相对基座的误差差异，允许学习哪些特征应被鼓励、哪些缺陷应被抑制；并不要求每张优选图都优于Z-Image的所有输出。这是机制层面的可能性，不是收益保证。

SFT只最小化优选图的flow速度MSE，会直接拉近模型与这些图像的分布；优选图仍可能含有伪影、较弱细节或不适合Z-Image的风格。DPO也可能学到来源模型风格而非人类偏好：例如winner经常来自一种生成器，loser来自另一种生成器。强基座的剩余提升空间与数据偏差必须通过生成评估判断。

MVPO的多个噪声视图可以检验减少偏好估计噪声是否有用；不能靠增加K消除错误标签、来源偏差或优选图质量不足。当前MVPO K2在同128对预算下比DPO有更多计算，应单独报告。

## 最快的有效判断方式

四种方法均从同一个Z-Image Base开始，各自训练3200步；SFT不是其他方法的必需预训练阶段。默认lr5e-6、warmup200、LoRA16、1024及相同训练seed。SFT每步128张winner；DPO/Linear每步128对；MVPO每步128对×2视图。相同数据行数不等于相同FLOPs。

保持已有64条validation prompt、相同生成seed及50步/CFG4设置，评估base和200/400/800/1300/1800/2300/2800/3200。无需先增加新的检查任务，直接使用这些正常训练评估。

| 观察 | 可以支持的解释 | 不能据此断言 |
|---|---|---|
| DPO/Linear相对base持续改善，视觉质量也改善 | 当前配方存在可迁移的偏好信号 | HPDv3-sub对所有Z-Image任务都有效 |
| SFT退步，而偏好方法改善 | 相对比较比直接模仿winner更合适 | SFT在所有超参下都无效 |
| SFT改善，各偏好方法无明显增益 | winner分布有用，需检查偏好目标的尺度/设置 | 数据必然没有偏好信号 |
| MVPO优于DPO且配对差值稳定 | 该设置支持多视图收益 | 等计算量优势或多seed稳定性 |
| 全部退步，训练loss却下降 | 优化在拟合，生成分布可能偏移 | 一次实验足以证明数据永远不可学 |
| 区间跨0，方法曲线相近 | 64条面板的证据不足 | 没有效果或已证明等效 |

400/800步用于看方向，1300步以后看是否持续；默认训练仍跑满3200，不自动早停。不要反复挑选一个偶然最高点当最终效果。`delta_base`回答是否优于原模型；同一步的`delta_reference`回答是否优于DPO。

```bash
python scripts/summarize_flow_runs.py "$Z_SFT_RUN" "$Z_DPO_RUN" "$Z_MVPO_RUN" "$Z_LINEAR_RUN" \
  --reference-run "$Z_DPO_RUN" --out logs/zimage-four-methods-v1.csv
```

区间使用prompt配对bootstrap，包含同prompt多生成seed时先平均；这不是多个训练seed的不确定性，也不是人类胜率。如果DPO尚未评估到对应步数，参考比较留空，不能拿DPO的800步替代其他方法的1300步。工具只读取现有评估，不增加训练热循环检查。

## 正式结论需要补的证据

用validation选择checkpoint之后，只对基座、DPO及候选方法做一次独立heldout测试即可先确认方向；保留最终3200步结果，避免只报告挑选出的最好点。使用已隔离的test prompt，并检查外部benchmark与训练prompt的重叠。生成设置和随机种子保持一致。

至少同时看一个不同的评分器与固定prompt下的盲看结果，尤其检查文字、手部、对象数量和多样性。HPSv3与HPDv3有训练来源关联：在HPDv3验证prompt上更高的HPSv3不等于独立的人类偏好验证。PickScore也只是代理指标，不能单独宣布全面质量提升。当前64条单seed筛选不足以证明多样性维持不变。

如果到800/1300步四方法都没有向好迹象，先看已保存图片和训练日志，区分明显实现失败、过强更新与数据分布不合适；不要立刻同时搜索LR、beta、K和EMA。模型在早期出现过提升但后期退步时，优先研究较小学习率或较早checkpoint；若始终没有提升，再用少量相同prompt比较原模型输出与数据winner，决定是否需要更贴近Z-Image输出的偏好数据。任何新数据筛选须对所有对照一致应用，另开实验名，不能只为MVPO挑数据。
