# 服务器实验任务书

本机仓库是唯一代码源。服务器只在
`/mnt/highway1/wang/ScholarNavigator-main` 同步已验证 commit、构建索引和运行耗时实验；不得在服务器直接修改代码，不得在该目录之外创建环境、下载数据或写入文件。

## 开始前

1. 完整阅读 `AGENTS.md`。
2. 不读取、显示、提交 `.env`，不输出密码、API Key 或其他密钥。
3. 只读记录同步前后的 Git commit、Python/Node/GPU、`.venv`、语料、Faiss 索引、Benchmark 目录和每个目录最新 committed generation 的 cursor/status。
4. 仅同步本机已测试并明确指定的 commit；优先使用 `git pull --ff-only`。同步失败或服务器存在未确认改动时停止并报告，不在服务器修代码。

## P0 与 ANN

- P0 仅使用带 `id/arxiv_id` 的 Cornell/arXiv 元数据与 PaSa 按规范化 arXiv ID 精确关联。
- 禁止使用 AutoScholarQuery test gold、qrels 或标题模糊关联构建语料。
- 先运行 `scripts/build_pasa_semantic_corpus.py --metadata <已审计元数据路径>`，保存语料和报告哈希。
- 再运行 `scripts/build_local_hybrid_index.py`。中断后只可使用同一配置加 `--resume`；配置、语料哈希、模型或 HNSW 参数变化时必须新建索引运行。
- 索引报告必须包含语料覆盖率、ANN 相对 exact-flat 的 Recall、构建耗时、峰值内存和查询延迟。

## 资格实验

按同一固定 200 条查询、同一数据/索引哈希和同一评测配置执行：

- `contest_qual200_bm25_v1`
- `contest_qual200_dense_v1`
- `contest_qual200_reranker_v1`

每个阶段保存 `config.json`、Git commit、数据哈希、PID、启动命令、日志路径、资源账本和 committed-generation 状态。中断后只能使用同一 RunId、相同配置和 `--resume`。

只有候选组的 F1 或 Recall 有真实提升且资源账本通过，才允许排期完整 1000 条运行；小样本、失败或未完成目录不得写入正式结果。

## 完整消融

资格门禁通过后才可运行：

- `contest_full_rules_v1`
- `contest_full_dense_v1`
- `contest_full_dense_reranker_v1`
- `contest_full_dense_reranker_llm_v1`

`llm_semantic` 保持现有 Prompt：每条查询最多一次调用、最多两条补充查询、temperature=0、严格 JSON Schema、始终保留原始查询。记录模型、Prompt 版本、Prompt/Completion Token、调用次数、延迟、失败和规则回退。RefChain 仅是可选第五组。

所有报告必须声明内部 F1/Recall 是工程内部指标，不能宣称与赛事官方 scorer 完全一致。
