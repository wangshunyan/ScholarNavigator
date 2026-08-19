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

运行 `scripts/check_contest_qualification.py` 后，只有门禁通过的候选可进入完整四组：`contest_full_rules_v1`、`contest_full_dense_v1`、`contest_full_dense_reranker_v1`、`contest_full_dense_reranker_llm_v1`。每个运行需保存配置、commit、输入哈希、PID、命令、日志、资源账本和 committed generation。

神经 reranker 资格运行还必须证明真实模型推理成功：结果中不得出现
`local_model_fallback_count`，且必须有正数 batch、候选数、推理成功数、模型指纹、设备、最大长度、固定 batch size=8、候选上限=120、延迟样本和 CUDA 峰值显存；汇总报告必须包含 P50/P95 延迟和候选吞吐。旧的
`contest_qual200_reranker_v1` 曾发生 CUDA 索引断言，只能作为失败诊断；修复后的资格运行使用新 RunId
`contest_qual200_reranker_v2`，不能用同一目录恢复旧配置。

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
