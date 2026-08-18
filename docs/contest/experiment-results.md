# 赛题三实验状态

更新时间：2026 年 8 月 18 日。

本页只汇总已落盘的真实运行，所有结果均来自 `outputs/benchmark_runs/`。除特别标注的小样本 sanity 外，结果均来自 1000 条公开 AutoScholarQuery 测试查询；它们仅用于本地工程验证和竞赛说明书，不代表隐藏测试集或比赛排名。

## 已验证的工程能力

- PaSa 官方标题库已转换为 569,432 篇本地 JSONL，并用 SQLite FTS5 建立低内存 BM25 索引。
- 公开 arXiv 摘要数据已严格标题匹配到 31,136 篇 PaSa 论文，生成 `datasets/semantic/pasa_papers_with_abstracts.jsonl`。
- BGE-small-en-v1.5 向量索引已构建完成：`outputs/benchmark_cache/local_hybrid/embeddings.npy`，形状为 31,136 x 384。
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

当前下一轮完整候选配置为 `hybrid_deep_rrf`：`run_profile=high_recall`、`ranking_policy=rrf_fusion`、`max_candidate_papers=300`、BM25/semantic 内部候选各 120、API/LLM 仍为 0。

## 已完成的完整运行

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

## 当前结论

1. `local_hybrid` 相比 `local_bm25` 有稳定提升：F1@20 增加 0.0039，Recall@20 增加 0.0226，MRR 增加 0.0189，平均延迟只增加约 0.004 秒，API 与 LLM 调用仍为 0。
2. 这说明“摘要语义向量 + BM25 RRF”是有效优化，但提升幅度还不足以称为最终强方案。当前仍有 2,129 个 gold 属于 `not_retrieved`，核心瓶颈仍是初始召回。
3. 规则 Judgement 仍丢失较多已召回 gold：`local_hybrid` 的 Judgement FN 率为 0.347。下一步应优先调低标题/摘要证据的假阴性，或加入受控 LLM judgement 消融。
4. `local_bm25 + arXiv` 的旧完整运行未完成，公开 API 出现 429、读取超时和 TLS 握手超时。该目录只保留为可靠性诊断，不得写入质量对比或提交结果。
5. `contest_full_local_hybrid_v1` 是中断目录，不作为正式结果。Windows 下不要在 benchmark 运行中读取 `results.jsonl`，否则可能影响原子替换；需要观察进度时只检查进程，或中断后使用同一 RunId `-Resume`。

## 后续正式实验口径

当前可引用主实验命令：

```powershell
.\scripts\run_contest_benchmark.ps1 -Mode full -Configuration local -RunId contest_full_local_baseline_v3
.\scripts\run_contest_benchmark.ps1 -Mode full -Configuration hybrid -RunId contest_full_local_hybrid_v2
```

下一轮候选完整运行命令：

```powershell
.\scripts\run_contest_benchmark.ps1 -Mode full -Configuration hybrid_deep_rrf -RunId contest_full_hybrid_deep_rrf_v1
```

Linux/服务器同等命令：

```bash
./scripts/run_contest_benchmark.sh --mode full --configuration hybrid_deep_rrf --run-id contest_full_hybrid_deep_rrf_v1
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
