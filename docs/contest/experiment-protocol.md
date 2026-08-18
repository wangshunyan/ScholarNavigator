# 赛题三正式实验协议

## 目标

在 PaSa/AutoScholarQuery 的 1000 条公开测试查询上，比较零外部 API 的本地效率 baseline 与当前主候选的质量/成本权衡。

| 角色 | 检索源 | 目的 |
| --- | --- | --- |
| baseline | `local_bm25` | 验证本地索引效率、完整性和零 API 成本。 |
| candidate | `local_hybrid` | 验证本地标题 BM25、摘要 BGE 向量检索与 RRF 融合效果。 |

两组固定使用：`top_k=20`、`adaptive` query adapter、`current_rules` 查询规划、`current_rules` judgement、`current_rules` 排序、`evaluation` profile、诊断和资源账本。

## 执行

在 VSCode 中运行下列任务，或执行相应命令：

```powershell
.\scripts\run_contest_benchmark.ps1 -Mode full -Configuration local
.\scripts\run_contest_benchmark.ps1 -Mode full -Configuration hybrid
```

`hybrid` 是零外部 API 的本地混合检索，首次运行前必须已生成摘要语料和向量索引。中断后必须复用原 `run-id`：

```powershell
.\scripts\run_contest_benchmark.ps1 `
  -Mode full `
  -Configuration hybrid `
  -RunId <原运行目录名> `
  -Resume
```

索引构建命令：

```powershell
.\.venv\Scripts\python.exe scripts\build_pasa_semantic_corpus.py
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
- 不使用 `AutoScholarQuery_test.jsonl` 的 gold 论文作为检索语料，不使用小样本数字代替完整结果。
