# ScholarNavigator

面向复杂学术查询的论文搜索、排序与结构化归纳系统。

## 环境要求

- Python 3.11 或以上
- Node.js 20 或以上

## 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm install
```

## 配置环境变量

根据 [`.env.example`](.env.example) 在项目根目录创建本地 `.env`；变量含义和默认值以该模板为准。

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

候选优化配置可运行：

```powershell
.\scripts\run_contest_benchmark.ps1 -Mode full -Configuration hybrid_deep_rrf -RunId contest_full_hybrid_deep_rrf_v1
```

Linux 服务器使用同等脚本：

```bash
./scripts/run_contest_benchmark.sh --mode full --configuration hybrid_deep_rrf --run-id contest_full_hybrid_deep_rrf_v1
```

参赛补齐步骤、优化优先级和提交材料清单见 [docs/contest/next-steps.md](docs/contest/next-steps.md)；演示查询可参考 [docs/contest/demo-queries.md](docs/contest/demo-queries.md)。

竞赛实验的真实小样本结果、正式 1000 条启动方式和说明书骨架见 [docs/contest/experiment-results.md](docs/contest/experiment-results.md)、[docs/contest/experiment-protocol.md](docs/contest/experiment-protocol.md) 和 [docs/contest/submission-report-template.md](docs/contest/submission-report-template.md)。
