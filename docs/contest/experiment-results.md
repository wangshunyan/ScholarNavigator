# 赛题三实验状态（以当前工作树为准）

更新时间：2026 年 8 月 23 日。

本页只记录当前 checkout 中可以直接读取、复核和重新运行的证据。服务器目录、旧聊天记录或不存在于当前 `outputs/benchmark_runs/` 的 RunId 都不构成当前证据。所有 F1、Recall 等数值只能作为内部离线工程指标，不能当作赛事官方成绩。

## 当前可确认的能力

- `datasets/pasa/paper_database/id2paper.json` 包含 569,432 个稳定 arXiv ID 到标题的映射。
- `datasets/local_bm25/pasa_papers.jsonl` 包含 569,432 条唯一 arXiv ID 记录；当前 title 完整度为 1.0，但 abstract、authors、year、venue、doi 完整度均为 0。
- `datasets/semantic/pasa_papers_with_abstracts.jsonl` 包含 31,136 条唯一 arXiv ID 记录；title 与 abstract 完整度为 1.0，但 authors、year、venue、doi 完整度均为 0。该语料是 legacy 功能验证输入，不满足正式元数据质量目标。
- 构建器现在要求稳定 arXiv ID，并可保留合法输入中的 authors、year、venue、doi；不允许通过标题模糊匹配或 AutoScholarQuery gold/qrels 生成正式语料。
- 本地代码包含 SQLite BM25、local_hybrid、Faiss/语义索引接口、规则判断、结构化导出、FastAPI 和前端检索源选择；是否带来质量提升必须用同一输入和成对实验重新验证。
- 质量面板现在额外展示作者/DOI 完整度与 arXiv–DOI 身份一致性；这些是独立诊断，不改变当前默认排序。没有独立撤稿或重复风险来源时仍显示 `unknown`。

## 当前可读取的历史运行

本机存在若干 `outputs/benchmark_runs/` 目录。其中较完整的 local baseline/hybrid 运行使用旧语料或旧实现，且代码指纹不同；比较工具拒绝将它们视为严格成对比较。因此它们只能证明历史链路曾运行，不能证明当前代码的候选收益。

文档曾提到的 Dense、Reranker 完整产物当前不在本机，相关数字暂不引用。服务器上的同名目录也必须先由操作者导出脱敏 manifest 和哈希后才能纳入审计。

### 当前提交的 200 条 Hybrid 诊断配对

在 Git HEAD `05759c1309bcdc0b6c48209c58eaf6bdcabb7436`、相同 dirty 工作树差异（两组 `runtime_code_hash=5c018d9b…`）、同一前 200 条查询、相同 `high_recall`/300 候选预算/资源账本下，重新运行了：

| 配置 | RunId | Recall@20 | F1@20 | MRR | 平均延迟（秒） |
| --- | --- | ---: | ---: | ---: | ---: |
| BM25 baseline | `contest_qual200_local_clean_e7f2b72` | 0.0684 | 0.00982 | 0.0335 | 1.189 |
| BM25 + Dense RRF | `contest_qual200_hybrid_clean_e7f2b72_retry` | 0.1098 | 0.0167 | 0.0712 | 1.168 |

同一查询级别 bootstrap（5,000 次，seed `20260818`）得到 ΔRecall@20=`0.0413`，95% CI `[0.0140, 0.0718]`；ΔF1@20=`0.00687`，95% CI `[0.00301, 0.01111]`。两组成功率均为 1.0，运行代码指纹一致。完整 JSONL/资源账本仍只在本地 `outputs/benchmark_runs/`，不会提交 GitHub。

后续 qualification 命令已固定在两种脚本中统一使用 `high_recall` 和 300 候选预算；可用 `scripts/analyze_paired_benchmark_runs.py` 重新生成上述成对 JSON 报告。

这组结果已绑定干净提交且两组各有 200 条完整查询，但仍只是 legacy title+abstract 语料上的内部资格诊断，不是赛事官方成绩：作者、年份、期刊和 DOI 完整度为 0，故 P1-01 未完成；不能据此启动 1000 条正式运行或宣称已通过赛事资格。

候选预算消融（同一前 200 条查询）显示：Hybrid 的 120→200 将 Recall@20 从 `0.10975` 提升至 `0.11200`、F1@20 从 `0.01669` 提升至 `0.01750`；200→300 指标不再提升。因此 `hybrid_deep_rrf` 的诊断入口使用 200，生产默认仍保持不变。

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
