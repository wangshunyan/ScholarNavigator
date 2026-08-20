# 赛题三提交前清单

## 当前已完成

- FastAPI 后端和 Next.js 前端可在 Windows 本地启动。
- 真实公开学术源已接入：OpenAlex、arXiv、Semantic Scholar、PubMed。
- 查询理解、子查询规划、检索、去重、相关性判断、重排序、结构化结果和 SSE 进度已具备。
- AutoScholarQuery 本地评测文件可读取：1000 条查询、2403 条 arXiv gold。
- Windows 下 benchmark CLI、崩溃恢复存储和 local BM25 缓存已修复并有测试覆盖。
- PaSa 官方标题库已转换为 `datasets/local_bm25/pasa_papers.jsonl`，共 569,432 篇并保留 `arxiv_id`。
- 已构建 SQLite FTS5 本地 BM25 索引；`.env` 已配置本地语料，运行时配置可识别 `local_bm25`。
- 旧的 31,136 条公开 arXiv 摘要标题匹配语料和对应向量索引已标记为 legacy，不作为新方案正式依据。
- P0 精确 arXiv ID 语料、Faiss ANN、索引 Recall、构建耗时和峰值内存报告已完成并冻结；具体证据见 `docs/contest/experiment-results.md`。
- 已接入旧版 `local_hybrid`：BM25 与摘要向量各取候选，RRF 融合后进入原有去重、判断、排序和结构化输出链路；P0/Faiss 版本需要重新验证。
- 本地 BM25 已增加自然语言查询填充词过滤，避免礼貌语和泛化词主导标题检索；相关测试已通过。
- 已提供 `scripts/check_local_hybrid_search.py`，可重复检查索引加载、摘要返回、BM25/semantic 来源和前 5 条 gold 命中。
- 已完成真实 5 条来源消融：`arXiv`、`local_bm25 + arXiv`、`local_bm25 + arXiv + OpenAlex`，并保留配置、指标、逐条结果、失败分析和资源账本。
- 已提供 `scripts/run_contest_benchmark.ps1` 及 VSCode 任务，可启动可恢复的 local baseline 和 local_hybrid candidate。
- 已提供 `scripts/run_contest_benchmark.sh`，用于 Linux 服务器在独立项目目录内运行相同 Benchmark。
- 已保存当前代码版本的完整 1000 条历史对照：`contest_full_local_baseline_v3` 与 `contest_full_local_hybrid_v2`，两组资源账本均通过；由于使用旧标题匹配语料和旧全矩阵实现，只能作 legacy 工程基线，不是 P0/Faiss 正式成绩。
- 已保存 100 条历史候选参数实验：`hybrid_deep_rrf` 只用于诊断，P0/Faiss 变更后不得直接恢复。
- 已准备 20 条演示查询，覆盖时间、方法、数据集、排除条件、中英文混合和结构化导出。

## 提交前必须完成

1. P0 精确语料和 Faiss 资源报告完成并冻结；旧 `contest_full_local_*` 仅作为 legacy 对照。
2. P0/Faiss 变更后的 200 条资格门禁已完成；`contest_qual200_reranker_v4_gpu1` 通过零 fallback 和资源审计。
3. `contest_full_rules_v1`、`contest_full_dense_v1` 和 `contest_full_dense_reranker_v4` 已完成。旧 LLM 全量运行 `contest_full_dense_reranker_llm_v14` 正在以本地 loopback Provider 执行，但已出现 fallback，只能保留为诊断；新的 `contest_qual200_dense_reranker_llm_v15` 必须先通过 smoke、200 条审计和资格门禁，之后才可运行 `contest_full_dense_reranker_llm_v15`，完成前不作为成绩依据。
4. 只使用完整成功运行目录中真实生成的 `config.json`、`metrics.json`、`summary.md`、`results.jsonl`、`stage_metrics.json`、`error_analysis.json` 和 `resource_ledger.json` 写实验结果。
5. 当前提交明确“LLM 接口已实现，但尚无零 fallback、完整审计通过的 1000 条正式 LLM 运行”。不得把诊断、smoke 或未完成的 LLM 功能写成实测创新结果。
6. 将新主线的 `local_bm25` 与 `local_hybrid` 对比、阶段诊断中的初始候选 Recall、Judgement FN 和平均延迟写入说明书。
7. 填写团队、学校、指导教师、成员分工、软件著作权/开源许可证等竞赛表单字段，并完成匿名要求检查。
8. 录制 3 到 5 条交互式演示，确认视频没有 API Key、本地绝对路径或无关个人信息。

## 提交材料

- 项目说明书：使用 `docs/contest/submission-report-template.md`，填写问题、场景、系统架构、算法流程、数据集、评测协议、消融实验、真实结果、成本和局限性。
- 项目源码：完整 README、`.env.example`、依赖锁定、Windows 启动命令、Benchmark 命令和产物目录说明。
- `legacy/spar_original/` 当前没有随附可核验的许可证文件，提交包必须排除；审查记录见 [third-party-review.md](third-party-review.md)。
- 发布压缩包必须由受跟踪文件构建，执行 `python scripts/build_contest_release_package.py --repository-root . --output <package.zip>`。构建器明确排除 `.env`、`outputs/`、`datasets/semantic/` 和 `legacy/spar_original/`；正式提交前仍须人工检查压缩包中没有凭据、运行产物或本地绝对路径。
- 演示视频：从复杂查询输入到结构化论文结果、引用关系图、成本/延迟面板和异常降级。
- 答辩材料：突出复杂约束解析、检索迭代、本地+外部混合检索、可复现实验和 F1/效率权衡。
- 匿名检查：提交文档、视频、代码截图和压缩包不要暴露学校、导师、成员身份或本地敏感路径。

## 当前风险

- 旧 `local_hybrid` 的 F1@20 为 0.0147，但它使用 legacy 标题匹配语料，不能作为新主线成绩；P0/Faiss 结果已由新运行目录提供。
- `hybrid_deep_rrf` 目前只有 100 条候选实验，不能作为最终成绩；必须跑完 1000 条并通过资源账本后才能引用为主结果。
- LLM 当前默认 disabled；若要把 LLM 规划作为创新点，必须配置模型并记录版本、Token、延迟和回退。
- PaSa `id2paper.json` 是论文 ID 与标题库；正式语义摘要必须来自带 arXiv ID 的 Cornell/arXiv 元数据精确关联，不能使用标题匹配或 AutoScholarQuery gold。
- OpenAlex 在 2026 年 8 月 17 日的小样本中出现 HTTP 429 且未带来新增 gold，当前不作为默认主候选来源。
- 标题库专用阈值配置在前 5 条有改善，但在第 6 至第 10 条没有 gold 被召回，不能作为泛化证据或默认配置。
- SciFact 数据包不在本地；它只能作为辅助验证，不应代替 AutoScholarQuery 主评测。
- 历史 `record160` 冻结输入和两个所需 Git commit 在本机与服务器均缺失；相关全历史发布验收保持阻塞，详见 [historical-evidence-blockers.md](historical-evidence-blockers.md)。
