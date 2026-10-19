# 来源

本目录由本地`joint_sd15`、`formal_sd15`、`independent_sd15`的已有代码整理。精确来源文件、整理时SHA256和实验配置映射在`provenance.json`。2026-10-18起训练器只启用B/C，默认C；A/D仅保留配置归档。包名和导入路径已独立化。

目录组织参考：

- [SalesforceAIResearch/DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO)
- [Whynot0101/Linear-DPO](https://github.com/Whynot0101/Linear-DPO)

没有直接复制这两个上游仓库的源码，也没有实现Linear-DPO目标。Diffusion-DPO的参考相对MSE目标与一步rollout/self-correction的研究来源应在论文中分别注明。本地研究说明将后者归于SOAR启发；本次整理不新增原创性结论。

尚未替项目选择公开许可证或虚构作者信息；正式公开发布时由代码权利人确定项目许可，并核对本地代码的完整来源。模型、数据集和评估权重各自遵循其原许可，不随代码包上传。

新增评估已核对Linear-DPO commit `663179c7adbbbd2d77b97b5841534447eb291ebd`，来源哈希见`evaluation_sources.json`；运行时适配本地权重及官方HPSv2的OpenCLIP源码。

SDXL组件与配置核对同一Linear-DPO版本的SDXL运行脚本、训练实现与验证入口，来源见`sdxl_sources.json`。使用现有Diffusers/PEFT后端实现双编码器、time_ids和本地VAE；新增B/C LoRA模板沿用本项目研究目标，与上游完整UNet/FP16/Linear-DPO训练存在明确差异，见[SDXL.md](SDXL.md)。
