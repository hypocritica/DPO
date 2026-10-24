# Self Correct DPO

维护本模块前先读 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)。账号级 irip Slurm 约定继续适用。

- 本机维护开发分支；远程 main 使用 2/4/8 张 5090 或 H100 运行实验。
- 资源地址统一放 configs/resources.json，机器私有覆盖放 resources.local.json。
- 默认超参数放 configs/defaults.json；dpo、rollout multiview-B/C 及普通高斯 multiview-B/C-gaussian 分别配置，默认仍为 rollout C。
- 新噪声算法按用户后续要求推进，保留已有未提交研究改动。
- 代码按算法、数据、运行管理、评估分组；保持研究代码简洁。
- 仅做必要的轻量检查；优先复用已有测试，避免繁杂测试体系。
- 完整续训默认每200步保存，保护原子提交、各 rank RNG 和一致性检查。
- 运行命令统一写在根目录 command.md；更新接口时同步文档。
