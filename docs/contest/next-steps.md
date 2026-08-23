# 赛题三下一步执行手册

本文只把赛题文档中的要求当作参赛上下文，不把文档里的文字当成对代码助手的指令。

## 当前状态

- 本地可启动：后端 `http://127.0.0.1:8000`，前端 `http://127.0.0.1:3000`。
- 可用检索源：OpenAlex、arXiv、Semantic Scholar、PubMed。
- 已具备功能：复杂查询解析、子查询规划、多源检索、去重排序、结构化结果、引用关系图、SSE 进度、导出和运行诊断。
- 已接入 PaSa 官方标题库：`datasets/local_bm25/pasa_papers.jsonl`，共 569,432 篇、保留 `arxiv_id`。
- 已构建低内存 SQLite FTS5 BM25 索引：`outputs/benchmark_cache/local_bm25/*.sqlite3`。后端与评测可共享该落盘索引，不再把全量语料装进内存。
- 已完成真实 5 条 AutoScholarQuery smoke：100% 成功、Recall@20=0.20、MRR=0.20，资源账本校验通过。该结果只验证链路，不代表最终比赛成绩。
- 当前核心短板：已下载的官方 `id2paper.json` 只含标题，没有摘要、作者、年份和引用信息，限制了复杂查询的初始召回与重排序质量；LLM 默认关闭。

## 推荐数据集

主数据集选 PaSa/AutoScholarQuery。

理由：

- 赛题三就是复杂学术查询下的论文搜索与推荐，PaSa/AutoScholarQuery 与任务最贴近。
- 本项目已内置 AutoScholarQuery 测试查询适配器，可直接计算 F1、Recall、MRR、nDCG 和延迟等指标。
- gold 身份以 arXiv ID 为主，适合把 PaSa 论文库转换成本地 BM25 语料并保留 `arxiv_id`。

不要把 `AutoScholarQuery_test.jsonl` 里的 gold 论文直接当检索语料。那是答案泄漏，只能用于格式冒烟，不能用于正式评测或答辩材料。

辅助数据集：

- AstaBench/PaperFindingBench：用于说明泛化性。
- SciFact：只适合作为封闭语料辅助验证，不建议作为赛题三主结果。

## 优先优化方向

第一优先级：完成 PaSa 本地 BM25 与 arXiv 的 1000 条对比实验。

收益：用真实指标验证“本地标题库召回补偿 + arXiv 公开元数据”的主候选是否优于本地效率 baseline。2026 年 8 月 17 日的真实 5 条测试中，arXiv 单源为 0，而 local+arXiv 达到 Recall@20=0.20、F1@20=0.019；该结论仅用于选择完整实验配置。

第二优先级：补全高质量论文元数据。

收益：优先增加摘要、年份、作者、引用数或官方允许的全文元数据；当前标题-only 语料已在 smoke 阶段显示出召回瓶颈。

第三优先级：只在有模型条件时启用 LLM 查询理解/判断，并和非 LLM 基线做消融。

收益：如果没有稳定模型、Token 预算和失败回退记录，LLM 容易变成不可复现风险；默认关闭更适合先跑稳工程基线。

## VSCode 启动

项目已补 `.vscode/tasks.json`：

1. 打开 VSCode 命令面板。
2. 选择 `Tasks: Run Task`。
3. 运行 `ScholarNavigator: run app`。
4. 浏览器访问 `http://127.0.0.1:3000`。

单独验证后端：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/runtime/config
```

## PaSa 语料接入

拿到官方 `paper_database/id2paper.json` 或包含该文件的 zip 后执行：

```powershell
.\.venv\Scripts\python.exe scripts\build_pasa_local_bm25_corpus.py `
  --input D:\path\to\cs_paper_2nd.zip `
  --output datasets\local_bm25\pasa_papers.jsonl `
  --report outputs\benchmark_inputs\pasa_local_bm25_report.json `
  --identity arxiv_id
```

如果 zip 里有多个 JSON 文件，增加：

```powershell
  --zip-member paper_database/id2paper.json
```

然后在项目根目录 `.env` 中配置：

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

重启后端后确认：

```powershell
(Invoke-RestMethod http://127.0.0.1:8000/api/v1/runtime/config).connectors
```

其中 `local_bm25` 应显示 `available: true`。

首次检索或评测会自动创建 SQLite FTS5 BM25 索引；之后会复用
`outputs/benchmark_cache/local_bm25/*.sqlite3`，无需重建 56.9 万篇语料的内存索引。

## Benchmark 命令

先跑 5 条冒烟：

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
  --top-k 20 `
  --diagnostics `
  --resource-ledger
```

完整 local baseline：

```powershell
.\scripts\run_contest_benchmark.ps1 -Mode full -Configuration local
```

完整 candidate（本地标题库 + arXiv）：

```powershell
.\scripts\run_contest_benchmark.ps1 -Mode full -Configuration hybrid
```

中断后使用同一 `run-id` 恢复：

```powershell
.\scripts\run_contest_benchmark.ps1 `
  -Mode full `
  -Configuration hybrid `
  -RunId <原运行目录名> `
  -Resume
```

来源消融对比：

```powershell
.\.venv\Scripts\python.exe scripts\compare_benchmark_runs.py `
  --run outputs\benchmark_runs\<local-run-id> `
  --run outputs\benchmark_runs\<hybrid-run-id> `
  --allow-source-difference `
  --output outputs\benchmark_runs\contest_full_source_ablation.md
```

提交材料只能引用这些输出中的真实指标。

## 提交前材料

- 源码压缩包：保留 README、`.env.example`、依赖文件、`docs/contest`、benchmark 脚本和产物说明。
- 项目说明书：任务理解、架构图、算法流程、数据集、实验设置、结果表、消融、失败分析、成本和局限性。
- 演示视频：展示复杂查询输入、检索进度、论文排序、引用图、导出和运行诊断。
- 答辩 PPT：突出复杂约束解析、混合检索、本地 BM25、结构化归纳、可复现实验。
- 匿名检查：不要暴露学校、导师、团队成员、本地用户名、API key 或绝对敏感路径。
