# 赛题三实验状态（以当前工作树为准）

更新时间：2026 年 8 月 25 日。

本页只记录当前 checkout 中可以直接读取、复核和重新运行的证据。服务器原始目录、旧聊天记录或未导出的 RunId 都不构成当前证据；用户导出的脱敏 v3 资产位于被忽略的 `outputs/server_assets/20260824_metadata_v3/`，并通过完成链与哈希复核。所有 F1、Recall 等数值只能作为内部离线工程指标，不能当作赛事官方成绩。

## 当前可确认的能力

本页随当前 Git 提交更新；可复核的代码/测试基线以 `git rev-parse HEAD` 和最新 clean-clone smoke 的 `source_commit` 为准。所有内部指标仍不是赛事官方 scorer 结果。

- `datasets/pasa/paper_database/id2paper.json` 包含 569,432 个稳定 arXiv ID 到标题的映射。
- `datasets/local_bm25/pasa_papers.jsonl` 包含 569,432 条唯一 arXiv ID 记录；当前 title 完整度为 1.0，但 abstract、authors、year、venue、doi 完整度均为 0。
- `datasets/semantic/pasa_papers_with_abstracts.jsonl` 包含 31,136 条唯一 arXiv ID 记录；title 与 abstract 完整度为 1.0，但 authors、year、venue、doi 完整度均为 0。该语料是 legacy 功能验证输入，不满足正式元数据质量目标。
- v3 脱敏语料 `outputs/server_assets/20260824_metadata_v3/pasa_arxiv_enriched_v3.jsonl` 包含 569,432 条唯一 arXiv ID 记录；title、abstract、authors、year 完整度为 1.0，venue 为 12.152%，DOI 为 17.897%。它用于可审计内部 Hybrid 诊断，但尚未满足 venue/DOI 全量正式门禁。
- 构建器现在要求稳定 arXiv ID，并可保留合法输入中的 authors、year、venue、doi；不允许通过标题模糊匹配或 AutoScholarQuery gold/qrels 生成正式语料。
- 本地代码包含 SQLite BM25、local_hybrid、Faiss/语义索引接口、规则判断、结构化导出、FastAPI 和前端检索源选择；是否带来质量提升必须用同一输入和成对实验重新验证。
- 质量面板现在额外展示作者/DOI 完整度与 arXiv–DOI 身份一致性；这些是独立诊断，不改变当前默认排序。没有独立撤稿或重复风险来源时仍显示 `unknown`。

## 当前可读取的历史运行

本机存在若干旧 `outputs/benchmark_runs/` 目录。其中较完整的 local baseline/hybrid 运行使用旧语料或旧实现，且代码指纹不同；比较工具拒绝将它们视为当前严格成对比较。v3 的三组完整脱敏运行已导入并完成哈希/完成链审计，相关数字只在下方 v3 小节引用。

### 历史可读取的 200 条 Hybrid 诊断配对

在历史 Git 提交 `05759c1309bcdc0b6c48209c58eaf6bdcabb7436`、相同 dirty 工作树差异（两组 `runtime_code_hash=5c018d9b…`）、同一前 200 条查询、相同 `high_recall`/300 候选预算/资源账本下，重新运行了：

| 配置 | RunId | Recall@20 | F1@20 | MRR | 平均延迟（秒） |
| --- | --- | ---: | ---: | ---: | ---: |
| BM25 baseline | `contest_qual200_local_clean_e7f2b72` | 0.0684 | 0.00982 | 0.0335 | 1.189 |
| BM25 + Dense RRF | `contest_qual200_hybrid_clean_e7f2b72_retry` | 0.1098 | 0.0167 | 0.0712 | 1.168 |

同一查询级别 bootstrap（5,000 次，seed `20260818`）得到 ΔRecall@20=`0.0413`，95% CI `[0.0140, 0.0718]`；ΔF1@20=`0.00687`，95% CI `[0.00301, 0.01111]`。两组成功率均为 1.0，运行代码指纹一致。完整 JSONL/资源账本仍只在本地 `outputs/benchmark_runs/`，不会提交 GitHub。

后续 qualification 命令已固定在两种脚本中统一使用 `high_recall` 和 300 候选预算；可用 `scripts/analyze_paired_benchmark_runs.py` 重新生成上述成对 JSON 报告。

### 2026-08-24：服务器 v3 Hybrid 成对审计

用户导出的脱敏服务器资产已复制到本地 `outputs/server_assets/20260824_metadata_v3/`（该目录被 Git 忽略，不进入发布包）。三组运行的最新 `.run_commits` generation 均包含 `RUN_COMPLETED` 与 `COMMITTED`，`run_manifest` 均为 200/200；每组 `results.jsonl` 为 200 行、`failures.jsonl` 为空。三组均绑定同一 v3 语料 SHA-256=`7a385c87250ff438f5748cc49ee683acf1edd01d2f12432d17fe60e83908a31a`、569,432 篇和索引指纹 `91302a92…e68f71`。

v3 语料严格审计结果：arXiv ID 唯一，title/abstract/authors/year 完整度均为 100%；venue 为 12.152%、DOI 为 17.897%。因此它足以支持当前 Hybrid 检索工程诊断，但仍不满足“venue 与 DOI 全量可核验”的正式元数据门禁，不能据此宣称 P1-01 或赛事资格完成。

baseline `contest_qual200_metadata_v3_hybrid_baseline_eb7d151` 与 candidate `contest_qual200_metadata_v3_hybrid_reranker_eb7d151` 已通过严格只差 Reranker 的审计：同一提交/查询顺序/预算/语料/索引；candidate 使用 CUDA、fallback=0、batch=8、候选=120，593 次成功推理，P50/P95 约 1.761/2.077 秒，峰值显存约 1.50 GB。5,000 次 query-level bootstrap 为：

| 指标 | baseline | candidate | Δ（candidate-baseline） | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Recall@20 | 0.12850 | 0.12797 | -0.00054 | [-0.00987, 0.00838] |
| F1@20 | 0.02108 | 0.02137 | +0.00029 | [-0.00219, 0.00273] |
| F1@10 | 0.02324 | 0.01849 | -0.00475 | [-0.00875, -0.00143] |

没有指标满足“均值为正且 95% CI 下界大于 0”的启用门槛，Reranker 保持默认关闭。候选数 60→120 的负向消融同样没有改善（ΔRecall@20=-0.00400，95% CI [-0.02083, 0.01133]），因此也不采用。以上数值是内部离线工程指标，不是赛事官方成绩。

### 2026-08-24：查询理解停用词清理成对复核

在同一 v3 Hybrid 语料/索引、同一前 200 条查询、`high_recall`/300 候选预算、`top_k=20`、单 worker、零网络/零 LLM 条件下，比较移除叙述性请求词后的当前 candidate 与 baseline。两组均完成 200/200，失败数为 0；5,000 次 query-level bootstrap 结果如下：

| 指标 | Δ（candidate-baseline） | 95% CI |
| --- | ---: | ---: |
| Recall@20 | 0.00000 | [0.00000, 0.00000] |
| F1@20 | 0.00000 | [0.00000, 0.00000] |
| F1@10 | -0.00071 | [-0.00214, 0.00000] |
| F1@5 | +0.00167 | [0.00000, 0.00500] |

该结果未达到 Recall@20 或 F1@20 的严格正向门禁，因此不把停用词清理宣称为检索质量提升，也不据此改变默认排序策略；代码保留为查询理解可解释性和演示稳定性修复。报告为 `outputs/benchmark_runs/querycleanup_paired_analysis.json`（本地 ignored），所有数值均为内部离线工程指标，不是赛事官方成绩。

### 2026-08-24：v3 检索策略消融汇总

为避免把单次小样本波动误写成优化收益，在同一 v3 语料/索引和前 200 条查询上继续检查了几个已有策略：

| 候选策略 | 结果 | 决策 |
| --- | --- | --- |
| `prf_v1` | ΔRecall@20=-0.00175，95% CI [-0.00659, 0.00200]；ΔF1@20=-0.00016，[-0.00172, 0.00134] | 不启用 |
| `controlled_relaxation` | 30 条筛选 Recall@20/F1@20 不变，F1@10 下降 | 不扩展 |
| `rrf_k=10` | 30 条筛选 Recall@20/F1@20 不变，F1@10 下降 | 不启用 |
| 候选数 60→200 | 30 条筛选 Recall@20=-0.01111，F1@20=-0.00047 | 不启用 |
| `calibrated_rules_v1` | ΔRecall@20=-0.00250，95% CI [-0.01000, 0.00500]；ΔF1@20=-0.00083，[-0.00261, 0.00091] | 不切换 |

这些数值均为内部离线工程指标，不是赛事官方成绩；实验失败数为 0，未使用 gold/qrels 参与在线检索。完整 JSON 产物保存在本地 ignored `outputs/benchmark_runs/`。

这组结果曾绑定干净提交且两组各有 200 条完整查询，但仍只是 legacy title+abstract 语料上的历史内部资格诊断，不是当前提交的运行结果或赛事官方成绩：作者、年份、期刊和 DOI 完整度为 0，故 P1-01 未完成；不能据此启动 1000 条正式运行或宣称已通过赛事资格。

候选预算消融（同一前 200 条查询）显示：Hybrid 的 120→200 将 Recall@20 从 `0.10975` 提升至 `0.11200`、F1@20 从 `0.01669` 提升至 `0.01750`；200→300 指标不再提升。因此 `hybrid_deep_rrf` 的诊断入口使用 200，生产默认仍保持不变。

负面消融：将语义语料中可按 arXiv ID 精确关联的 31,136 条摘要并入 BM25 标题库后，Recall@20=`0.06742`，低于标题基线 `0.06842`；配对区间跨 0，平均延迟也增加。该方案不启用、不进入发布语料，仅作为避免重复尝试的内部证据。

### 2026-08-25：词形归一化 30 条筛选

针对已召回但被规则判断为 weak/irrelevant 的英文词形差异，固定 v3 语料、Hybrid 索引、前 30 条查询、预算和代码，仅在判断阶段启用保守的 `lexical_normalization_v1`（不改变候选池、权重、阈值或在线 gold/qrels 访问）。两组均 30/30 成功，候选池保持一致。

| 指标 | baseline | candidate | Δ | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Recall@20 | 0.12500 | 0.14722 | +0.02222 | [-0.02222, 0.09444] |
| F1@20 | 0.02541 | 0.02345 | -0.00195 | [-0.01026, 0.00696] |
| F1@10 | 0.02581 | 0.02771 | +0.00189 | [-0.01250, 0.01818] |

筛选未达到“均值为正且 95% CI 下界大于 0”的进入门槛，且 F1@20 点估计下降，因此不启动 200 条正式配对、不切换生产默认策略。该结果仍只是内部工程指标，不是赛事官方成绩。脱敏包 SHA-256：baseline `66af50eee5d6e70e78caae0ca4cac0626fba33b6d75f6977939416556d4cd9b1`，candidate `0e3268643116dd924492e5356566d6755443a460d9a5d502837271d5871ccf01`。

## 正式实验门槛

1. 获取带稳定 arXiv ID、摘要、作者、年份、期刊和 DOI 的合法元数据源，并重建语料与索引。
2. 固定同一 200 条 query 顺序、数据哈希、模型/预算和资源约束，预绑定 baseline/candidate comparison plan。
3. 验证 candidate recall、F1@5/10/20、Recall@20、MRR、延迟、调用数、失败率、资源账本和 query-level 显著性区间。
4. 只有 200 条资格门槛通过后，才允许执行 1000 条完整运行；没有收益的 RRF、质量过滤、Query Evolution、Reranker 或 LLM 保持默认关闭。

## 已知阻塞

- 当前没有可读取的带稳定 arXiv ID 的完整元数据输入，P1-01 尚未完成。
- 历史 replay、network snapshot、预注册哈希和官方 scorer 输入部分缺失或漂移；严格门禁必须失败，不能通过改哈希或伪造产物解除。
- LLM Provider、GPU、开放全文许可和服务器实验结果均属外部条件；本地开发不读取 `.env`、服务器或 SSH 凭据。

## 复现入口

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\audit_corpus_metadata.py datasets\local_bm25\pasa_papers.jsonl
.\.venv\Scripts\python.exe scripts\audit_corpus_metadata.py datasets\semantic\pasa_papers_with_abstracts.jsonl
.\.venv\Scripts\python.exe -m pytest -q
```

只有新的、完整且可读取的运行目录中的实际数值，才可以写入参赛说明书；所有内部指标必须明确标注“非官方成绩”。
