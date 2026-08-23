# PaSa 本地 BM25 与语义混合检索接入步骤

本项目默认使用 OpenAlex、arXiv、Semantic Scholar、PubMed 四个公开来源。
参加赛题三时，建议把 PaSa/AutoScholar 的论文库接成本地 BM25 来源，用于提高召回和降低外部 API 依赖；在此基础上，再接入按 arXiv ID 精确关联的公开元数据和 BGE 向量索引，作为 `local_hybrid` 候选。
BM25 索引使用 SQLite FTS5 落盘实现；语义索引使用本地 SentenceTransformers/BGE 向量和持久化 Faiss ANN。当前仓库中按标题匹配生成的 31,136 篇摘要子集属于 legacy 产物，不能作为新的正式 P0 语料。

## 推荐数据集

主数据集使用 PaSa/AutoScholarQuery。

原因：

- 赛题是复杂学术查询下的论文搜索与推荐，AutoScholarQuery 与任务最贴近。
- 本仓库已经包含 `benchmark/AutoScholarQuery_test.jsonl`，可直接用于 1000 条查询的评测流程。
- 本地检查结果显示该文件包含 1000 条查询、2403 条 gold，gold 身份全部是 arXiv ID，因此本地语料应优先保留 `arxiv_id`。

AstaBench/PaperFindingBench 可作为泛化验证；SciFact 可作为封闭语料辅助验证，但不建议作为主数据集。

## 语料转换

下载 PaSa 数据中的 `paper_database/id2paper.json` 或包含它的 zip 后，运行：

```powershell
.\.venv\Scripts\python.exe scripts\build_pasa_local_bm25_corpus.py `
  --input D:\path\to\cs_paper_2nd.zip `
  --output datasets\local_bm25\pasa_papers.jsonl `
  --report outputs\benchmark_inputs\pasa_local_bm25_report.json `
  --identity arxiv_id
```

如果 zip 中有多个 JSON 文件，用 `--zip-member paper_database/id2paper.json` 指定成员。

不要把 `AutoScholarQuery_test.jsonl` 的 gold 论文直接转换成检索语料。那会把答案泄漏进检索阶段，只能用于格式冒烟测试，不能用于正式评测或提交说明。

转换后先运行结构与排序元数据门禁：

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\audit_corpus_metadata.py `
  datasets\local_bm25\pasa_papers.jsonl `
  --require-fields title abstract authors year venue doi
```

该命令只有在每条记录都具备稳定 arXiv ID 及全部排序字段时才返回成功；当前仓库附带的标题语料会明确失败，因为它是离线 BM25 smoke 资产而不是正式元数据完整语料。

## 后端 `.env`

在项目根目录 `.env` 中配置：

```env
SCHOLAR_AGENT_LOCAL_BM25_CORPUS=datasets/local_bm25/pasa_papers.jsonl
SCHOLAR_AGENT_LOCAL_BM25_CACHE_DIR=outputs/benchmark_cache/local_bm25
SCHOLAR_AGENT_LOCAL_BM25_DOCUMENT_ID_FIELD=_id
SCHOLAR_AGENT_LOCAL_BM25_TITLE_FIELD=title
SCHOLAR_AGENT_LOCAL_BM25_ABSTRACT_FIELD=abstract
SCHOLAR_AGENT_LOCAL_BM25_DOCUMENT_IDENTITY=arxiv_id
SCHOLAR_AGENT_LOCAL_BM25_ARXIV_ID_FIELD=arxiv_id
# 随包标题语料没有 DOI；只有元数据完整语料才填写 doi
SCHOLAR_AGENT_LOCAL_BM25_DOI_FIELD=
```

启动后访问 `http://127.0.0.1:8000/api/v1/runtime/config`，看到 `local_bm25` 为 `available: true` 即表示后端已识别本地索引。

首次检索会在 `SCHOLAR_AGENT_LOCAL_BM25_CACHE_DIR` 中创建 `*.sqlite3` 索引。索引构建完成后可被后端和 Benchmark 复用，不会重复加载全量语料到内存。

## 语义混合索引

安装语义检索依赖后，先准备带 `id/arxiv_id`、`title`、`abstract` 的 Cornell/arXiv 元数据，再运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-semantic.txt
.\.venv\Scripts\python.exe scripts\build_pasa_semantic_corpus.py `
  --metadata datasets\semantic\arxiv_metadata.jsonl `
  --pasa-paper-index datasets\pasa\paper_database\id2paper.json
.\.venv\Scripts\python.exe scripts\build_local_hybrid_index.py
.\.venv\Scripts\python.exe scripts\check_local_hybrid_search.py --limit 5
```

在 `.env` 中增加：

```env
SCHOLAR_AGENT_LOCAL_HYBRID_SEMANTIC_CORPUS=datasets/semantic/pasa_papers_with_abstracts.jsonl
SCHOLAR_AGENT_LOCAL_HYBRID_INDEX_DIR=outputs/benchmark_cache/local_hybrid
SCHOLAR_AGENT_LOCAL_HYBRID_MODEL=datasets/semantic/models/models/AI-ModelScope--bge-small-en-v1.5/snapshots/master
SCHOLAR_AGENT_LOCAL_HYBRID_BM25_CANDIDATE_LIMIT=60
SCHOLAR_AGENT_LOCAL_HYBRID_SEMANTIC_CANDIDATE_LIMIT=60
SCHOLAR_AGENT_LOCAL_HYBRID_RRF_K=60
```

`scripts/build_local_hybrid_index.py` 使用分批写入、Faiss 持久化索引和进度文件；如果中断，只能使用相同配置和 `--resume` 继续构建。构建报告必须包含语料覆盖率、ANN 相对 exact-flat 的 Recall、构建耗时、峰值内存和检索延迟。最终产物是本地索引和 metadata，不应提交到 GitHub。

## 前端使用

启动前后端后，在前端“检索源”中选择：

- `本地索引`：只用本地 BM25。
- `语义混合`：只用本地 BM25 + 摘要向量 RRF 融合。
- `本地+外部`：本地 BM25 加 OpenAlex、arXiv、Semantic Scholar。

默认“推荐组合”保持外部公开源，不会自动启用本地语料。

## Benchmark 命令

小样本冒烟：

```powershell
.\.venv\Scripts\python.exe scripts\run_benchmark.py `
  --dataset auto_scholar_query `
  --run-id autoscholar_local_smoke `
  --limit 5 `
  --sources local_bm25 `
  --local-bm25-corpus datasets\local_bm25\pasa_papers.jsonl `
  --local-bm25-document-id-identity arxiv_id `
  --local-bm25-arxiv-id-field arxiv_id `
  --local-bm25-doi-field doi `
  --run-profile fast `
  --top-k 20
```

正式对比应至少保留两组：

- baseline：`local_bm25`
- candidate：`local_hybrid`

报告里只写实测结果，不写未运行的分数。
