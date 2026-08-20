# 历史证据阻塞记录

更新时间：2026-08-20。本记录只说明发布验收中无法在当前工作区重放的历史证据，
不改变任何策略结论，不创建占位产物，也不降低测试门槛。

## 已核实的缺失项

本机和服务器 `/mnt/highway1/wang/ScholarNavigator-main` 的只读检查均未找到：

- `outputs/benchmark_runs/lexical_normalization_record160_813cf3a_r5/`；
- `outputs/benchmark_runs/lexical_normalization_record160_813cf3a_r6/`；
- 所需的冻结 Snapshot 输入；
- 历史 Git commit `a743c59c719dd10db742cbd526f8f09c5ba13839`；
- 历史 Git commit `d6d37eb1f0d9a7cff1e41fce69d1b6b4d9175548`。

因此，依赖这些输入的历史重放、部分发布候选可复现性测试和 Record160 复核不能在
当前仓库中通过。该结论不影响 P0/Faiss、rules、Dense、reranker 或新 LLM 实验的
独立运行与审计，但它阻止宣称“全历史发布门禁已清零”。

## 恢复条件

维护者需要从可核验的归档或远端恢复原始 commits 及对应的完整、未修改的冻结运行输入。
恢复后必须：

1. 校验 commit、运行配置、Snapshot manifest 和每个输入文件哈希；
2. 将原始运行目录回收到其记录的相对路径，不覆盖任何现有实验目录；
3. 使用原 manifest 连续重放两次，逐字节核验输出哈希；
4. 重新运行全仓测试和发布候选验收。

在上述证据齐全前，当前状态保持为明确的历史证据阻塞。不得以重跑不同代码、不同
配置或不同数据集的实验替代该历史重放，也不得删除、跳过或弱化相关测试。
