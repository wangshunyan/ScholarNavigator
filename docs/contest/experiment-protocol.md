# 赛题三正式实验协议

## 目标

在 PaSa/AutoScholarQuery 的 1000 条公开测试查询上，比较零外部 API 的本地效率 baseline 与当前主候选的质量/成本权衡。

| 角色 | 检索源 | 目的 |
| --- | --- | --- |
| baseline | `local_bm25` | 验证本地索引效率、完整性和零 API 成本。 |
| candidate | `local_hybrid` | 验证本地标题 BM25、摘要 BGE 向量检索与 RRF 融合效果。 |

两组固定使用：`top_k=20`、`adaptive` query adapter、`current_rules` 查询规划、`current_rules` judgement、`current_rules` 排序、`evaluation` profile、诊断和资源账本。

## 执行

P0/Faiss 版本的正式主线先固定 200 条资格实验，旧 `local/hybrid` 结果仅为 legacy 工程基线：

```powershell
.\scripts\run_contest_benchmark.ps1 -Mode qualification -Configuration rules -RunId contest_qual200_bm25_v1
.\scripts\run_contest_benchmark.ps1 -Mode qualification -Configuration dense -RunId contest_qual200_dense_v1
.\scripts\run_contest_benchmark.ps1 -Mode qualification -Configuration reranker -RunId contest_qual200_reranker_v2
```

运行 `scripts/check_contest_qualification.py` 后，只有门禁通过的候选可进入完整四组。当前 P0/Faiss 主线使用 `contest_full_rules_v1`、`contest_full_dense_v1`、`contest_full_dense_reranker_v4` 和 `contest_full_dense_reranker_llm_v4`。每个运行需保存配置、commit、输入哈希、PID、命令、日志、资源账本和 committed generation。新 runner 在 `config.json` 写入 `code.commit` 与脱敏 `execution.process_id/launch_command/log_path`，并固定日志为 `outputs/run_logs/<RunId>.log`；这些会话字段不进入 resume 语义签名。

`dense_reranker_soft` 是一个独立的候选 policy，仅将 `partially_relevant_threshold` 从
`0.45` 降至 `0.35`，保留原有硬约束、检索、RRF 和 reranker 参数。它必须先使用
`contest_qual200_dense_reranker_soft_v1` 与 `contest_qual200_reranker_v4_gpu1` 做同一
200 条查询的配对资格比较。门禁会拒绝任何除该受审 delta 外的检索、数据、预算或
Judgement 配置漂移；通过前不得将其写入正式成绩或启动完整运行。
完整运行的阶段一离线瓶颈证据和该候选的边界记录在
[`stage1-bottleneck-analysis.md`](stage1-bottleneck-analysis.md)。

神经 reranker 资格运行还必须证明真实模型推理成功：结果中不得出现
`local_model_fallback_count`，且必须有正数 batch、候选数、推理成功数、模型指纹、设备、最大长度、固定 batch size=8、候选上限=120、延迟样本和 CUDA 峰值显存；汇总报告必须包含 P50/P95 延迟和候选吞吐。旧的
`contest_qual200_reranker_v1` 曾发生 CUDA 索引断言，只能作为失败诊断；修复后的资格运行使用新 RunId
`contest_qual200_reranker_v2`，不能用同一目录恢复旧配置。
若 v2 运行期间 GPU 被其他任务占用并产生 OOM，保留 v2 失败证据，不覆盖该目录；在同一代码、数据、索引和 reranker 参数下，可使用
`contest_qual200_reranker_v2_gpu1` 进行显式 GPU 隔离重试。该重试必须在 gate 中零 fallback、零失败且资源审计通过后才有资格进入完整实验。
若修复模型推理或固定最大输入长度，旧 v2/v2_gpu1 也不得恢复；使用新的 `contest_qual200_reranker_v3_gpu1`，并再次完整执行 200 条资格门禁。
v3 如因完整序列 logits 的 CPU 传输导致性能不满足实验资源边界，保留其 committed generation 作为性能诊断；仅优化最终时间步传输的 v4 使用 `contest_qual200_reranker_v4_gpu1`，仍需从前 200 条完整运行并通过同一门禁。

LLM 组必须等待完整 reranker 组核验通过后才启动。它固定使用 `llm_semantic`、当前 Prompt、`temperature=0`、严格 JSON Schema，且每个查询最多一次 LLM 调用、最多两条补充查询并始终保留原始查询。活动 Prompt `llm_query_planning@1.0.1` 为降低无效扩展风险固定生成一条不超过 12 词、逐字保留核心 topic 词的补充查询；runner 将 `--max-llm-calls=1` 作为每条 SearchBudget 的上限，并固定 `--max-search-rounds=3`。完整运行结束后必须执行：

```bash
python scripts/audit_contest_llm_run.py \
  --run outputs/benchmark_runs/contest_full_dense_reranker_llm_v4 \
  --output outputs/benchmark_runs/contest_full_dense_reranker_llm_v4/llm_audit.json
```

审计要求 1000 条结果、完成标记、每条调用数不超过 1、补充查询不超过 2、Prompt/模型元数据、Token、延迟、Schema 拒绝和 `current_rules` 回退记录齐全。Provider 不可用时只记录 LLM 组未完成，不伪造指标。

当外部 provider 未配置时，可在服务器项目目录内使用 `scripts/serve_local_llm_provider.py` 启动绑定
`127.0.0.1` 的 Transformers/Qwen3-4B 本地 OpenAI-compatible provider。模型必须下载至
`datasets/semantic/models/`，LLM Provider 与 Reranker 必须使用不同 GPU、`temperature=0`；
`llm_query_planning` 使用 JSON Schema 受约束解码，缺少该依赖或无法生成合规对象时 fail closed；
运行进程使用临时环境变量连接 loopback endpoint，不修改或读取 `.env`。本地模型服务只解决运行时可用性，
不改变当前 Prompt、数据、候选池、RunId 门禁或 LLM 完整审计要求。

`hybrid` 是零外部 API 的本地混合检索，首次运行前必须已用带 arXiv ID 的 Cornell/arXiv 元数据精确生成摘要语料和 Faiss 索引。中断后必须复用原 `run-id`，且配置、数据哈希和索引参数必须完全一致：

P0 构建器流式读取官方逐行 JSON 快照，只按规范化 arXiv ID 关联，不读取标题做匹配。官方快照同一 ID 的历史修订按 `update_date` 选择最新记录并写入冲突计数；缺少可区分更新时间的冲突记录会拒绝构建。

```powershell
.\scripts\run_contest_benchmark.ps1 `
  -Mode full `
  -Configuration hybrid `
  -RunId <原运行目录名> `
  -Resume
```

索引构建命令：

```powershell
.\.venv\Scripts\python.exe scripts\build_pasa_semantic_corpus.py --metadata datasets\semantic\arxiv_metadata.jsonl
.\.venv\Scripts\python.exe scripts\build_local_hybrid_index.py
.\.venv\Scripts\python.exe scripts\check_local_hybrid_search.py --limit 5
```

## 完整性检查

每个运行目录必须包含：

- `config.json`
- `metrics.json`
- `summary.md`
- `results.jsonl`
- `stage_metrics.json`
- `error_analysis.json`
- `resource_ledger.json`

资源账本检查：

```powershell
.\.venv\Scripts\python.exe scripts\check_resource_accounting.py `
  check `
  --ledger outputs\benchmark_runs\<run-id>\resource_ledger.json
```

来源消融报告：

```powershell
.\.venv\Scripts\python.exe scripts\compare_benchmark_runs.py `
  --run outputs\benchmark_runs\<local-run-id> `
  --run outputs\benchmark_runs\<hybrid-run-id> `
  --allow-source-difference `
  --output outputs\benchmark_runs\contest_full_source_ablation.md
```

## 报告规则

- 报告 F1@20、Recall@20、Precision@20、MRR、成功率、平均 API 调用、平均 Token、平均延迟和失败率。
- `OpenAlex`、`Semantic Scholar`、`PubMed` 只有在相同协议下完成真实运行且无明显可靠性回退时才可加入新候选组。
- LLM 只有在提供商、模型版本、Token、延迟、回退和独立消融结果齐全时才可列为实测创新点。
- 不使用 `AutoScholarQuery_test.jsonl` 的 gold 论文作为检索语料，不使用标题模糊匹配，不使用小样本数字代替完整结果。
- 先固定同一批 200 条查询比较 BM25、BM25+Dense、BM25+Dense+Reranker；只有 F1/Recall 有真实提升且资源账本通过，才运行完整 1000 条。
- 内部 F1/Recall 只用于项目工程比较，不得宣称与赛事官方 scorer 完全一致。
