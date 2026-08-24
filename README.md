# ScholarNavigator

面向复杂学术查询的论文搜索、排序与结构化归纳系统。

## 环境要求

- Python 3.11 或以上
- Node.js 20.19+（或 22.13+）

## 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 运行测试、clean-clone smoke 或开发审计时额外安装
pip install -r requirements-dev.txt

cd frontend
npm install
```

## 配置环境变量

根据 [`.env.example`](.env.example) 在项目根目录创建本地 `.env`；变量含义和默认值以该模板为准。
模板已指向仓库随附的标题型 `datasets/local_bm25/pasa_papers.jsonl`，复制后即可进行离线 BM25 演示；它不是元数据完整的正式竞赛语料。LLM、语义模型和凭据仍保持关闭/空值。

批量演示也可以显式使用本地来源；CLI 与生产检索器共享来源 schema：

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/run_search_batch.py `
  --input docs/contest/demo-queries.jsonl `
  --output outputs/demo-batch/results.jsonl `
  --manifest outputs/demo-batch/manifest.json `
  --sources local_bm25
```

如果本地语料或索引配置缺失，运行会记录结构化 connector warning/error，而不会静默改用在线来源。
`manifest.json` 会绑定输入/输出 SHA-256、代码 commit、成功/失败数量及网络/LLM 调用计数，便于队友复核运行是否来自同一份输入；它不包含 gold/qrels，也不是官方成绩。

评委现场可用一条命令验收 5 条 gold-blind 演示查询（需要先按 `.env.example` 配置本地 BM25）：

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\check_demo_reproducibility.py
```

该检查复用批量 CLI，要求 5/5 查询成功、每条至少有 1 条可见结果、网络/LLM 调用为 0 且未加载 gold/qrels；输出 manifest 只用于演示复现，不是 Recall/F1 或官方成绩。

## 启动后端

```bash
PYTHONPATH=src uvicorn scholar_agent.app.main:app --host 127.0.0.1 --port 8000
```

Windows PowerShell：

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m uvicorn scholar_agent.app.main:app --host 127.0.0.1 --port 8000
```

## 启动前端

```bash
cd frontend
npm run dev
```

## VSCode 一键运行

本仓库已提供 `.vscode/tasks.json`：

1. 在 VSCode 命令面板运行 `Tasks: Run Task`。
2. 选择 `ScholarNavigator: run app` 同时启动后端和前端。
3. 打开 `http://127.0.0.1:3000`。

## 竞赛数据集与本地语料

赛题三建议以 PaSa/AutoScholarQuery 作为主评测数据集。本地 BM25 语料转换、`.env` 配置和 benchmark 命令见 [docs/contest/local-bm25-pasa.md](docs/contest/local-bm25-pasa.md)。

可选的本地混合检索需要额外安装 `requirements-semantic.txt`。语义语料构建必须使用包含 arXiv ID、标题和摘要的 Cornell/arXiv 元数据；不含 ID 的旧摘要 CSV 会被脚本拒绝：

```powershell
.\.venv\Scripts\python.exe scripts\build_pasa_semantic_corpus.py `
  --metadata datasets\semantic\arxiv_metadata.jsonl
.\.venv\Scripts\python.exe scripts\build_local_hybrid_index.py
.\scripts\run_contest_benchmark.ps1 -Mode full -Configuration hybrid
```

该方案用 PaSa 标题库做 BM25 召回，用按 arXiv ID 精确关联的 title+abstract 语料做 BGE 向量召回，再用 RRF 融合。标题不参与语料关联；原始下载文件和索引属于本地资产，不应提交到源码仓库。
官方 `arxiv-metadata-oai-snapshot.json` 是逐行 JSON，构建器按流式方式读取。官方快照中同一 ID 的历史修订记录按 `update_date` 选择最新记录，并在报告中计数；没有可区分更新时间的冲突记录会直接失败。
索引构建默认使用可用的 CUDA；需要强制 CPU 时设置 `SCHOLARNAVIGATOR_LOCAL_HYBRID_DEVICE=cpu`，服务器实验应在资源账本中记录实际设备。

候选优化配置可运行：

```powershell
 .\scripts\run_contest_benchmark.ps1 -Mode full -Configuration hybrid_deep_rrf -RunId contest_full_hybrid_deep_rrf_v1
```

上述 `hybrid_deep_rrf` 命令是经过 120/200/300 候选池消融后保留的诊断入口（BM25/semantic 各 200）；它仍不是当前 P0/Faiss 正式运行入口，新主线必须先完成完整元数据输入和 200 条资格门禁。

Linux 服务器使用同等脚本：

```bash
./scripts/run_contest_benchmark.sh --mode full --configuration hybrid_deep_rrf --run-id contest_full_hybrid_deep_rrf_v1
```

## 赛题三固定资格与正式消融

P0 语料或 Faiss 索引变化后，不恢复旧 `local/hybrid` 结果。先在同一批前 200 条查询上运行：

对于服务器已构建的正式 v3 语料与索引，不要修改脚本或 `.env`。两个 runner 都接受
`Bm25Corpus`/`SemanticCorpus`/`SemanticIndexDir`/`SemanticModel`（Linux 分别为
`--bm25-corpus`/`--semantic-corpus`/`--semantic-index-dir`/`--semantic-model`），使两组
运行能够绑定同一份输入。可先使用 `-PlanOnly` 或 `--plan-only` 检查实际参数，不会加载
模型、访问网络或创建运行目录。

以下示例是 v3 Hybrid 与仅开启 Reranker 的受控 200 条对比；将尖括号替换为服务器上的
实际相对路径，并使用从未复用过的 RunId：

```bash
COMMON=(
  --mode qualification
  --bm25-corpus outputs/metadata/pasa_arxiv_enriched_v3.jsonl
  --bm25-cache-dir outputs/benchmark_cache/local_bm25_v3
  --semantic-corpus outputs/metadata/pasa_arxiv_enriched_v3.jsonl
  --semantic-index-dir outputs/benchmark_cache/local_hybrid_v3
  --semantic-model <bge-model-directory>
  --max-workers 1
)

./scripts/run_contest_benchmark.sh "${COMMON[@]}" \
  --configuration hybrid \
  --run-id contest_v3_qual200_hybrid_baseline_<date> \
  --plan-only

./scripts/run_contest_benchmark.sh "${COMMON[@]}" \
  --configuration reranker \
  --run-id contest_v3_qual200_hybrid_reranker_<date> \
  --reranker-model <qwen3-reranker-directory> \
  --reranker-device cuda:<index> \
  --plan-only
```

确认两份计划除 RunId 和 Reranker 相关参数外一致后，删除 `--plan-only` 顺序运行。
完成的运行必须保留结果、资源账本和完成标记，并通过
`scripts/package_server_evidence.py` 导出脱敏 bundle；历史 inventory 不能替代这一步。
本地导入 bundle 后使用 `scripts/analyze_paired_benchmark_runs.py` 比较 Recall@20/F1 与
bootstrap 区间。区间未严格支持提升、出现 fallback 或资源违规时，Reranker 保持关闭。
若比较的是单一受控规划/判断策略，必须显式加入 `--allow-strategy-difference`；该选项最多允许一个策略字段变化，数据集、查询顺序、预算、来源资产和其他共享输入仍会严格校验。
对于这组“仅开启 Reranker”的正式对比，必须加入 `--strict-reranker-only`：该门禁要求
两组都是干净代码的 200 条 `local_hybrid` 运行，固定相同语料/模型/索引/融合/预算，
并审计 candidate 的逐案例 GPU 推理、零 fallback、固定 batch=8/候选=120、延迟和显存记录。

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_paired_benchmark_runs.py \
  --baseline outputs/imported_server_evidence/contest_v3_qual200_hybrid_baseline_<date> \
  --candidate outputs/imported_server_evidence/contest_v3_qual200_hybrid_reranker_<date> \
  --strict-reranker-only \
  --output outputs/server_evidence/paired_v3_reranker_analysis.json
```

```powershell
.\scripts\run_contest_benchmark.ps1 -Mode qualification -Configuration rules -RunId contest_qual200_bm25_v1
.\scripts\run_contest_benchmark.ps1 -Mode qualification -Configuration dense -RunId contest_qual200_dense_v1
.\scripts\run_contest_benchmark.ps1 -Mode qualification -Configuration reranker -RunId contest_qual200_reranker_v4_gpu1
```

候选只有在配对 bootstrap 和资源账本门禁通过后才能运行完整 1000 条：

```powershell
.\.venv\Scripts\python.exe scripts\check_contest_qualification.py `
  --baseline outputs\benchmark_runs\contest_qual200_bm25_v1 `
  --candidate outputs\benchmark_runs\contest_qual200_dense_v1 `
  --output outputs\benchmark_runs\contest_qual200_dense_v1\qualification_gate.json
```

`contest_qual200_reranker_v1` 是已知 CUDA 失败诊断，v2/v2_gpu1 因 logits 位置兼容性导致回退，均不能作为资格或正式结果。v3 修复了正确性但暴露完整序列 logits 的 CPU 传输瓶颈，故不作正式结果。`contest_qual200_reranker_v4_gpu1` 固定 2048 token、120 候选和 batch=8，并只传输最终决策时间步；它必须同时满足真实 GPU 推理、零回退、P50/P95 延迟、吞吐和峰值显存审计，并通过配对 bootstrap 与资源账本门禁后，才可运行 `contest_full_dense_reranker_v4`。Qwen3 Reranker 只从本地模型目录加载；缺失或失败时回退并记录，不能写作神经重排成绩。内部 F1/Recall 不等同于赛事官方 scorer。

如需对已核验许可证的开放全文生成段落证据，可使用受限 CLI；没有许可证确认时会失败关闭：

```bash
PYTHONPATH=src python scripts/fetch_open_full_text.py \
  --url https://<verified-host>/<paper> \
  --license-id CC-BY-4.0 --license-verified \
  --allowed-host <verified-host> \
  --output outputs/demo-full-text-evidence.json
```

LLM 组必须在完整 reranker 核验通过后，以新的 RunId 先完成 smoke 和 200 条资格门禁。P0-01 的后检索反馈链使用 `dense_reranker_llm_feedback`，保留 `current_rules` 首轮规划；每条查询最多一次调用、最多一条反馈查询、`temperature=0`、严格 JSON Schema 且保留原始查询。运行完成后必须执行 `scripts/audit_contest_llm_run.py`。正常跳过反馈的 smoke 只验证链路，不能主张实测 LLM 效果；200 条资格必须至少有一次真实反馈调用。历史 LLM v5-v16 均为诊断，不能作为正式成绩；Provider 不可用、fallback 非零、审计失败或 bootstrap 不通过时仅记录该组未完成，不能伪造成绩。

`dense_reranker_soft` 是不改变默认 `current_rules` 的独立受控候选，仅降低已审计的 partial-relevance 阈值。自动 GPU 选择曾在常驻本地 Provider 时引发 fallback；资格运行必须显式隔离 reranker，例如：

```bash
bash scripts/run_contest_benchmark.sh \
  --mode qualification \
  --configuration dense_reranker_soft \
  --run-id <new-run-id> \
  --reranker-device cuda:1
```

只有 200 条完整、零失败、零 fallback、资源账本与 reranker 审计通过，并且 paired-bootstrap 95% 区间支持 F1@20 或 Recall@20 提升，才允许新的完整 1000 条 RunId。历史文档曾提到 `contest_full_dense_reranker_soft_v2`，但该运行目录和可核验产物不在当前 checkout，不能把其数字当作当前成绩；重新运行前所有指标均标记为待测。所有内部 F1/Recall 仅为工程比较，不等同赛事官方 scorer。

P3-00 另行预注册了候选召回/Judgement 验证，不复用 v2：

```powershell
.\scripts\run_contest_benchmark.ps1 -Mode qualification -Configuration dense_reranker_rrf_soft -RunId contest_qual200_dense_reranker_rrf_soft_v3 -RerankerDevice cuda:1
.\.venv\Scripts\python.exe scripts\check_contest_qualification.py `
  --baseline outputs\benchmark_runs\contest_qual200_reranker_v4_gpu1 `
  --candidate outputs\benchmark_runs\contest_qual200_dense_reranker_rrf_soft_v3 `
  --output outputs\benchmark_runs\contest_qual200_dense_reranker_rrf_soft_v3\qualification_gate.json
```

该门禁固定 BM25+Dense 候选各 60 条、RRF `k=60` 和 soft Judgement，且要求初始候选 Recall
不下降、Judgement false-negative rate 严格下降，且平均延迟不超过 baseline 的 1.10 倍。它只在检索结束后使用 gold/qrels 计算离线指标；
未通过时不运行 full 1000 条，也不能进入正式内部成绩。

参赛补齐步骤、优化优先级和提交材料清单见 [docs/contest/next-steps.md](docs/contest/next-steps.md)；演示查询可参考 [docs/contest/demo-queries.md](docs/contest/demo-queries.md)。

竞赛实验的真实小样本结果、正式 1000 条启动方式和说明书骨架见 [docs/contest/experiment-results.md](docs/contest/experiment-results.md)、[docs/contest/experiment-protocol.md](docs/contest/experiment-protocol.md) 和 [docs/contest/submission-report-template.md](docs/contest/submission-report-template.md)。
