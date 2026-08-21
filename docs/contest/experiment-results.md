# 赛题三实验状态

更新时间：2026 年 8 月 18 日。

本页只汇总已落盘的真实运行，所有结果均来自 `outputs/benchmark_runs/`。除特别标注的小样本 sanity 外，结果均来自 1000 条公开 AutoScholarQuery 测试查询；它们仅用于本地工程验证和竞赛说明书，不代表隐藏测试集或比赛排名。

## 已验证的工程能力

- PaSa 官方标题库已转换为 569,432 篇本地 JSONL，并用 SQLite FTS5 建立低内存 BM25 索引。
- 旧的公开 arXiv 摘要数据标题匹配语料为 legacy 产物，原有 31,136 条和对应向量索引不得作为 P0 正式结果。
- P0 新语料已由官方 arXiv 元数据按规范化 arXiv ID 精确关联到 569,432 篇 PaSa 论文；不使用标题匹配、AutoScholarQuery gold 或 qrels。语料 SHA-256 为 `008ce46f15cc634f61bbadd4b960c6617c8d785c03dd2a307ed8d03fe9448d73`。
- BGE-small Faiss HNSW/IP 已完成服务器审计：569,432 个 384 维向量，`M=32`、`efConstruction=80`、`efSearch=64`，Recall@10 为 0.984；构建约 1764.6 秒、峰值 RSS 约 4.82 GB、ANN 延迟约 0.0239 秒、exact-flat 对照约 0.6153 秒。
- `local_hybrid` 已接入正式评测链路：本地 BM25 与摘要语义向量分别召回，使用 RRF 融合，再进入去重、规则判断、排序和结构化输出。
- Benchmark 输出包含逐 query 结果、F1/Precision/Recall/MRR/nDCG、阶段诊断、错误分析和资源账本。
- 后端 runtime config 和前端源选择已支持 `local_hybrid`，界面中显示为“语义混合”。

## 真实小样本结果

| 配置 | 运行目录 | F1@20 | Recall@20 | 平均 API | 平均延迟 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| local hybrid | `contest_smoke_local_hybrid_v1` | 0.054 | 0.440 | 0.0 | 2.508 s | 5 条链路 smoke，验证完整产物、摘要返回和账本写入。 |

## 候选参数实验

以下为前 100 条 AutoScholarQuery 测试查询上的候选实验，只用于筛选下一轮完整 1000 条配置，不得替代完整运行成绩。

| 配置 | 运行目录 | F1@20 | Recall@20 | MRR | 候选 Recall | Judgement 后 Recall | 平均延迟 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hybrid default | `contest_sample100_local_hybrid_default_v1` | 0.0127 | 0.1039 | 0.0378 | 0.2059 | 0.1159 | 0.830 s | 局部基线。 |
| hybrid deep | `contest_sample100_local_hybrid_deep_v1` | 0.0109 | 0.0906 | 0.0282 | 0.2359 | 0.1276 | 1.201 s | 加深候选池提高召回，但当前规则排序没有转化成 Top-20 收益。 |
| hybrid deep + RRF | `contest_sample100_local_hybrid_deep_rrf_v1` | 0.0136 | 0.1139 | 0.0402 | 0.2359 | 0.1276 | 1.213 s | 有效候选：更多候选配合检索融合排序，F1、Recall、MRR 均高于局部基线。 |
| hybrid deep + RRF + calibrated judgement | `contest_sample100_local_hybrid_deep_rrf_calibrated_v1` | 0.0136 | 0.1139 | 0.0426 | 0.2359 | 0.1276 | 1.221 s | MRR 略高，但 F1/Recall 未超过 `hybrid_deep_rrf`；暂不作为主候选。 |

旧的 `hybrid_deep_rrf` 只保留为历史诊断配置。P0 和 Faiss 变更后必须先进行固定 200 条资格比较，不能直接恢复旧的完整 1000 条运行。

## 历史完整运行（legacy）

| 配置 | 运行目录 | F1@20 | Recall@20 | Precision@20 | MRR | 平均 API | 平均延迟 | 成功率 | 资源账本 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| local BM25 | `contest_full_local_baseline_v3` | 0.0109 | 0.0620 | 0.0062 | 0.0407 | 0.0 | 0.762 s | 1.000 | passed |
| local hybrid | `contest_full_local_hybrid_v2` | 0.0147 | 0.0846 | 0.0085 | 0.0596 | 0.0 | 0.766 s | 1.000 | passed |

严格同代码版本对比报告：`outputs/benchmark_runs/contest_full_local_vs_hybrid_v2.md`。

## 阶段诊断

| 配置 | 初始候选 Recall | 最终返回 Recall@20 | Judgement FN 率 | 平均 gold rank | 瓶颈标签 |
| --- | ---: | ---: | ---: | ---: | --- |
| local BM25 | 0.131 | 0.062 | 0.467 | 7.723 | retrieval_recall_bottleneck, judgement_false_negative_bottleneck |
| local hybrid | 0.145 | 0.085 | 0.347 | 6.430 | retrieval_recall_bottleneck, judgement_false_negative_bottleneck |

以上两组结果只用于说明旧链路和资源账本曾经可运行；它们使用旧标题匹配语料和全矩阵向量实现，不是 P0/Faiss 正式成绩。

## P0/Faiss 正式运行状态

| 配置 | 运行目录 | 状态 | F1@20 | Recall@20 | 资源账本 | 说明 |
| --- | --- | --- | ---: | ---: | --- | --- |
| rules | `contest_full_rules_v1` | 已完成 | 0.01087 | 0.06195 | passed | MRR 0.04071，平均延迟 0.719 s。 |
| dense | `contest_full_dense_v1` | 已完成 | 0.02155 | 0.13508 | passed | MRR 0.09171，平均延迟 0.968 s。 |
| dense + reranker | `contest_full_dense_reranker_v4` | 已完成并审计通过 | 0.02442 | 0.15010 | passed | MRR 0.09406，平均延迟 3.909 s；零失败、零 fallback。 |
| dense + reranker + LLM | `contest_full_dense_reranker_llm_v14` | 未完成、不可审计诊断，不通过正式门禁 | 不得引用 | 不得引用 | 不适用 | 有 1000 条结果和后验诊断账本，但缺少 `RUN_COMPLETED`；另有 4 次 fallback，不能作为正式成绩。 |
| dense + reranker + LLM | `contest_qual200_dense_reranker_llm_v15` | 未完成诊断，不通过正式门禁 | 不得引用 | 不得引用 | 不适用 | 10 条成功、零 fallback，但结果 schema 丢失 HTTP transport 字段；停止后保留为诊断，不能恢复或引用。 |
| dense + reranker + LLM | `contest_qual200_dense_reranker_llm_v16` | 已完成诊断，不通过正式门禁 | 不得引用 | 不得引用 | passed | 200 条、零失败但有 1 次 temporary-overload fallback；LLM 审计失败，F1/Recall 的 paired-bootstrap 95% 区间均跨过零。 |
| dense + reranker + soft Judgement | `contest_qual200_dense_reranker_soft_v2` | 已完成并通过资格门禁 | qualification only | qualification only | passed | 200/200、零失败、零 fallback；GPU1 reranker 审计通过。 |
| dense + reranker + soft Judgement | `contest_full_dense_reranker_soft_v2` | 已完成并审计通过 | 0.02726 | 0.16817 | passed | MRR 0.09834，平均延迟 4.032 s；1000/1000、零失败、零 fallback，GPU1 reranker 审计通过。内部指标不等同赛事官方 scorer。 |

`contest_qual200_reranker_v4_gpu1` 已完成 200/200、零失败、零 fallback，并通过资源账本和配对 bootstrap 门禁；F1@20 增量为 +0.013747，Recall@20 增量为 +0.078419。完整 reranker 审计确认 1000 条、Qwen3 prompt v1、2048 最大长度、batch=8、候选上限=120、P50/P95 为 0.730/0.839 s、吞吐 156.13 candidates/s、峰值显存约 5.49 GiB。以上内部指标不等同于赛事官方 scorer。

Linux/Python 3.12 锁已在服务器 Python 3.12.3/x86_64 环境生成，覆盖 23 个包；由于服务器 pip wheelhouse 缺少兼容 wheel，离线安装资格为 `not_ready_missing_verified_version_or_artifact`，不能宣称完全离线可复现。

## 当前结论

## 运行可靠性与并发口径

LLM 传输层对每个逻辑调用采用固定的有限重试协议：HTTP `429/408/425/5xx`
属于可重试状态，最多 2 次 HTTP attempt，第一次失败后固定等待 1 秒；第二次仍失败
即交给上层 `current_rules` 回退，并在账本中记录失败原因。Schema、认证和其他不可重试
的客户端错误不会重复发送。兼容性参数降级也计入同一个 2 次总预算，不能额外扩大调用数。

`--max-workers` 是单条 query 内子查询检索的 worker 数，不是 1000 条 query 的并发数。
正式 `contest_full_dense_reranker_llm_v14` 运行目录的启动参数为 `--max-workers 1`，
因此不能宣称该运行使用 4 路并发；Windows/Linux runner 现在支持显式传入
`--max-workers 4`，但必须先以独立 smoke 和资源账本验证后才能用于新 RunId。当前 v14
不修改、不重启、不用新参数恢复。

1. 旧 `local_hybrid` 结果仅代表 legacy 标题匹配语料和全矩阵向量实现，不能作为 P0/Faiss 新方案的正式成绩。
2. 在同一 1000 条内部评测下，Dense 相对 rules 将 F1@20 从 0.01087 提升至 0.02155、Recall@20 从 0.06195 提升至 0.13508；完整 reranker 进一步达到 F1@20=0.02442、Recall@20=0.15010。资源账本和 reranker 审计均通过，但这些仍是内部指标，不等同于赛事官方 scorer。
3. Reranker 的完整运行使用真实 GPU 推理且零 fallback；代价是平均端到端延迟由 Dense 的 0.968 s 增至 3.909 s。提交材料应同时呈现质量增益和资源代价。
4. `contest_full_dense_reranker_llm_v14` 有 1000 条结果和后验诊断账本，但缺少 `RUN_COMPLETED`，故不能证明原子完成或作为可审计完整运行；同时记录 4 次 fallback。`contest_qual200_dense_reranker_llm_v15` 因结果 schema 丢失 HTTP transport 审计字段而停止；`contest_qual200_dense_reranker_llm_v16` 虽完成 200 条，但出现 1 次 temporary-overload fallback 且 paired-bootstrap 95% 区间未支持提升。三者都不得写成实测创新结果，且不得启动对应的 LLM 完整运行。
5. `local_bm25 + arXiv` 的旧完整运行及 `contest_full_local_hybrid_v1` 只保留为可靠性/中断诊断，不得写入正式质量对比或提交结果。
6. `contest_qual200_dense_reranker_soft_v2` 的 paired-bootstrap 支持质量提升；对应的 `contest_full_dense_reranker_soft_v2` 已完成独立 1000 条运行、完整性检查、资源账本和 reranker 审计。
7. `contest_full_dense_reranker_soft_v2` 已完成 1000 条并通过完整性、资源账本和 GPU reranker 审计；F1@20=0.02726023、Recall@20=0.16817491、MRR=0.09833638、平均延迟=4.032 s。以上仍为内部工程指标，不等同赛事官方 scorer。

## 后续正式实验口径

P0/Faiss 语料、索引和 Dense 完整运行已经完成。当前可引用完整成功且审计通过的 rules、Dense、reranker v4 和 soft Judgement v2；LLM 仍未通过正式门禁：

```powershell
 .\scripts\run_contest_benchmark.ps1 -Mode qualification -Configuration rules -RunId contest_qual200_bm25_v1
 .\scripts\run_contest_benchmark.ps1 -Mode qualification -Configuration dense -RunId contest_qual200_dense_v1
 .\scripts\run_contest_benchmark.ps1 -Mode qualification -Configuration reranker -RunId contest_qual200_reranker_v4_gpu1
```

只有门禁通过后，才运行对应完整组：

```powershell
 .\scripts\run_contest_benchmark.ps1 -Mode full -Configuration rules -RunId contest_full_rules_v1
 .\scripts\run_contest_benchmark.ps1 -Mode full -Configuration dense -RunId contest_full_dense_v1
 .\scripts\run_contest_benchmark.ps1 -Mode full -Configuration reranker -RunId contest_full_dense_reranker_v4
 .\scripts\run_contest_benchmark.ps1 -Mode full -Configuration dense_reranker_soft -RunId contest_full_dense_reranker_soft_v2
 .\scripts\run_contest_benchmark.ps1 -Mode smoke -Configuration dense_reranker_llm -RunId contest_smoke_dense_reranker_llm_v16
 .\scripts\run_contest_benchmark.ps1 -Mode qualification -Configuration dense_reranker_llm -RunId contest_qual200_dense_reranker_llm_v16
 .\scripts\run_contest_benchmark.ps1 -Mode full -Configuration dense_reranker_llm -RunId contest_full_dense_reranker_llm_v16
```

Linux/服务器同等命令：

```bash
./scripts/run_contest_benchmark.sh --mode qualification --configuration rules --run-id contest_qual200_bm25_v1
./scripts/run_contest_benchmark.sh --mode qualification --configuration dense --run-id contest_qual200_dense_v1
./scripts/run_contest_benchmark.sh --mode qualification --configuration reranker --run-id contest_qual200_reranker_v1
```

资源账本检查：

```powershell
.\.venv\Scripts\python.exe scripts\check_resource_accounting.py check --ledger outputs\benchmark_runs\contest_full_local_baseline_v3\resource_ledger.json
.\.venv\Scripts\python.exe scripts\check_resource_accounting.py check --ledger outputs\benchmark_runs\contest_full_local_hybrid_v2\resource_ledger.json
```

同代码版本对比：

```powershell
.\.venv\Scripts\python.exe scripts\compare_benchmark_runs.py `
  --run outputs\benchmark_runs\contest_full_local_baseline_v3 `
  --run outputs\benchmark_runs\contest_full_local_hybrid_v2 `
  --allow-source-difference `
  --output outputs\benchmark_runs\contest_full_local_vs_hybrid_v2.md
```

只有完整运行目录中的实际数值可以写入说明书；不得使用小样本、未完成目录或隐藏推测分数。
