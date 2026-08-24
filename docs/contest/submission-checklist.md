# 赛题三提交前清单

## 当前已完成

- 当前证据版说明书底稿见 [`submission-report-current.md`](submission-report-current.md)；它只引用可读取的 v3 脱敏 200 条内部工程证据，并把官方 scorer、完整元数据和 1000 条正式成绩明确留作提交前事项。

- FastAPI 后端和 Next.js 前端可在 Windows 本地启动。
- 真实公开学术源已接入：OpenAlex、arXiv、Semantic Scholar、PubMed。
- 查询理解、子查询规划、检索、去重、相关性判断、重排序、结构化结果和 SSE 进度已具备。
- AutoScholarQuery 本地评测文件可读取：1000 条查询、2403 条 arXiv gold。
- Windows 下 benchmark CLI、崩溃恢复存储和 local BM25 缓存已修复并有测试覆盖。
- PaSa 官方标题库已转换为 `datasets/local_bm25/pasa_papers.jsonl`，共 569,432 篇并保留 `arxiv_id`。
- 已构建 SQLite FTS5 本地 BM25 索引；`.env` 已配置本地语料，运行时配置可识别 `local_bm25`。
- 旧的 31,136 条公开 arXiv 摘要标题匹配语料和对应向量索引已标记为 legacy，不作为新方案正式依据。
- P0 精确 arXiv ID 语料契约和 Faiss 构建器已实现；当前 checkout 已重建本地语义索引并通过 ANN Recall@10=`0.998` 的索引构建报告。干净提交上的 200 条 BM25/Hybrid 配对诊断已完成，但正式元数据字段仍缺失，不能视为赛事资格通过。
- 已接入旧版 `local_hybrid`：BM25 与摘要向量各取候选，RRF 融合后进入原有去重、判断、排序和结构化输出链路；P0/Faiss 版本需要重新验证。
- 本地 BM25 已增加自然语言查询填充词过滤，避免礼貌语和泛化词主导标题检索；相关测试已通过。
- 已提供 `scripts/check_local_hybrid_search.py`，可重复检查索引加载、摘要返回、BM25/semantic 来源和前 5 条 gold 命中。
- 旧来源消融目录只作为历史线索；当前可引用结果必须来自本 checkout 可读取、代码指纹一致的完整运行。
- 已提供 `scripts/run_contest_benchmark.ps1` 及 VSCode 任务，可启动可恢复的 local baseline 和 local_hybrid candidate。
- 已提供 `scripts/run_contest_benchmark.sh`，用于 Linux 服务器在独立项目目录内运行相同 Benchmark。
- 仓库仍保留 `contest_full_local_baseline_v3` 与 `contest_full_local_hybrid_v2` 的历史对照线索；它们使用旧标题匹配语料和旧全矩阵实现，不能当作当前代码版本、P0/Faiss 正式成绩或赛事结果。
- 已保存 100 条历史候选参数实验：`hybrid_deep_rrf` 只用于诊断，P0/Faiss 变更后不得直接恢复。
- 已准备 20 条演示查询，覆盖时间、方法、数据集、排除条件、中英文混合和结构化导出。

## 提交前必须完成

1. P0 精确语料和 Faiss 资源报告完成并冻结；旧 `contest_full_local_*` 仅作为 legacy 对照。
2. P0/Faiss 变更后的 200 条资格诊断已在 `e7f2b72` 干净工作树完成成对产物；正式资格仍必须使用完整元数据输入并通过门禁，完成前不得启动或宣称 1000 条候选成绩。
3. 文档曾列出的 `contest_full_rules_v1`、`contest_full_dense_v1`、`contest_full_dense_reranker_v4` 等完整 RunId 当前不在本机可读取证据链；旧 LLM 目录也不能恢复或引用。后续必须用新 RunId，从 smoke/200 条资格重新开始。
4. 只使用完整成功运行目录中真实生成的 `config.json`、`metrics.json`、`summary.md`、`results.jsonl`、`stage_metrics.json`、`error_analysis.json` 和 `resource_ledger.json` 写实验结果。
5. 当前提交明确“LLM 接口已实现，但尚无零 fallback、完整审计通过的 1000 条正式 LLM 运行”。不得把诊断、smoke 或未完成的 LLM 功能写成实测创新结果。
6. `contest_qual200_dense_reranker_rrf_soft_v3` 的历史声明当前无法由本机 RunId 和产物核验，不能视为已通过资格门禁。
7. `contest_full_dense_reranker_rrf_soft_v3` 的历史声明当前无法由本机 RunId 和产物核验，内部数字暂不引用；即使恢复，也不能等同赛事官方 scorer。
8. 将新主线的 `local_bm25` 与 `local_hybrid` 对比、阶段诊断中的初始候选 Recall、Judgement FN 和平均延迟写入说明书。
9. 填写团队、学校、指导教师、成员分工、软件著作权/开源许可证等竞赛表单字段，并完成匿名要求检查。
10. 录制 3 到 5 条交互式演示，确认视频没有 API Key、本地绝对路径或无关个人信息。

## 提交材料

- 项目说明书：使用 `docs/contest/submission-report-template.md`，填写问题、场景、系统架构、算法流程、数据集、评测协议、消融实验、真实结果、成本和局限性。
- 项目源码：完整 README、`.env.example`、依赖锁定、Windows 启动命令、Benchmark 命令和产物目录说明。
- `legacy/spar_original/` 当前没有随附可核验的许可证文件，提交包必须排除；审查记录见 [third-party-review.md](third-party-review.md)。
- 发布压缩包必须由干净 Git 工作树的受跟踪文件构建，执行 `python scripts/build_contest_release_package.py --repository-root . --output <package.zip>`。构建器会拒绝 dirty tree、超过官方 200 MB 的 ZIP，并在包内写入 `release-manifest.json`（源 commit、成员 SHA-256 和排除边界）；明确排除 `.env`、`outputs/`、`datasets/semantic/` 和 `legacy/spar_original/`。正式提交前仍须人工检查压缩包中没有凭据、运行产物或本地绝对路径。
- 交付前再执行 `python scripts/verify_contest_release_package.py --package <package.zip> --expected-commit <commit>`，确认包未被传输或打包过程篡改。
- 演示视频：从复杂查询输入到结构化论文结果、引用关系图、成本/延迟面板和异常降级。
- 答辩材料：突出复杂约束解析、检索迭代、本地+外部混合检索、可复现实验和 F1/效率权衡。
- 匿名检查：提交文档、视频、代码截图和压缩包不要暴露学校、导师、成员身份或本地敏感路径。

## 当前风险

- 旧 `local_hybrid` 数字来自 legacy 标题匹配语料和不一致代码指纹，不能作为新主线成绩；当前新索引诊断仍绑定 dirty 工作树，正式资格证据尚未冻结。
- `hybrid_deep_rrf` 目前只有 100 条候选实验，不能作为最终成绩；必须跑完 1000 条并通过资源账本后才能引用为主结果。
- LLM 当前默认 disabled；若要把 LLM 规划作为创新点，必须配置模型并记录版本、Token、延迟和回退。
- PaSa `id2paper.json` 是论文 ID 与标题库；正式语义摘要必须来自带 arXiv ID 的 Cornell/arXiv 元数据精确关联，不能使用标题匹配或 AutoScholarQuery gold。
- OpenAlex 在 2026 年 8 月 17 日的小样本中出现 HTTP 429 且未带来新增 gold，当前不作为默认主候选来源。
- 标题库专用阈值配置在前 5 条有改善，但在第 6 至第 10 条没有 gold 被召回，不能作为泛化证据或默认配置。
- SciFact 数据包不在本地；它只能作为辅助验证，不应代替 AutoScholarQuery 主评测。
- 历史 `record160` 冻结输入和两个所需 Git commit 在本机与服务器均缺失；相关全历史发布验收保持阻塞，详见 [historical-evidence-blockers.md](historical-evidence-blockers.md)。
