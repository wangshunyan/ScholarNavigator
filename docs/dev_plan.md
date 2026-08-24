# ScholarNavigator 开发主计划

本文件是后续开发的唯一主计划。状态以当前工作树、代码、测试和可读取的运行产物为准；旧报告只作为线索，不能替代验证。

执行循环：读取本计划 → 选择最高优先级且可执行的未完成项 → 修改代码/测试 → 运行针对性验证 → 自我审查 → 更新本计划。任何检索、排序、Query Evolution、Reranker、LLM 或全文改动都必须有可复现的成对实验；没有收益不得默认启用。gold/qrels 只能在检索完成后的离线 evaluator 中使用，不能进入在线查询、索引、Prompt 或 connector。

## 当前权威快照（2026-08-24，优先于下方历史增量记录）

- 当前代码提交：以 `git rev-parse HEAD` 为准；本地 `main` 与 `origin/main` 一致，工作树干净。不要从历史快照复制提交号；发布 smoke 的 `source_commit` 是权威绑定。
- 当前本地已导入的 4 个服务器 ZIP 均为 `server_legacy_inventory_v1`，其中 200 条 BM25 与 Reranker 运行分别使用 `local_bm25/fast/200` 与 `local_hybrid/high_recall/300`，且均无完成标记；配对分析器按 `run_profile,budgets` 漂移拒绝比较。因此它们只能作为历史线索，不能完成 P1-02 或证明 Reranker 收益。
- 已修复正式实验入口的可复现性缺口：`scripts/run_contest_benchmark.ps1/.sh` 现在允许显式传入 BM25 语料/缓存、语义语料/索引和 BGE 模型路径，并提供无副作用的 `-PlanOnly/--plan-only` 参数审计。以同一 v3 资产生成的 Hybrid baseline 与 `reranker` candidate 计划已验证：除 RunId 和 Reranker 参数外，53 项 baseline 与 61 项 candidate 参数一致。专项回归 `52 passed`；Linux shell 因当前 Windows 无 WSL/Bash 运行时只能静态复核，服务器实际运行仍待脱敏 bundle 审计。
- 已修复 runner 到服务器证据包的完成标记断点：新 runner 的权威 `RUN_COMPLETED` 位于 `.run_commits` generation 链，打包器现会先验证该链及顶层兼容视图的 config/metrics/results/resource-ledger 字节一致性，再允许导出 bundle；manifest 只记录 generation、记录数与完成标记哈希。篡改顶层结果或伪造缺少完成契约的 manifest 都会失败关闭。专项 runner/commit/evidence 回归 `70 passed`。
- 已将“只开启 Reranker”的因果比较做成显式门禁：`analyze_paired_benchmark_runs.py --strict-reranker-only` 要求两个干净代码的完整 200 条 `local_hybrid` 运行，严格匹配全部非运行时 config、语料/模型/索引/融合指纹，并要求 candidate 实际 GPU 推理、零 fallback、固定 batch=8/候选=120、延迟和显存诊断通过。历史 BM25/Reranker inventory 因 profile、预算和源漂移仍会被拒绝。专项回归 `20 passed`。
- 最近验证：查询清理与跨代码版本成对分析提交后的后端全量回归 `2316 passed, 184 skipped, 2 warnings`（约 13 分 07 秒）；前端 `npm run lint` 与 `npm run build` 通过；使用项目 `.venv` 的 clean-clone smoke 在项目根目录运行并以自身的 `source_commit` 绑定，health/config=200、离线 BM25 5 条、网络请求 0、LLM disabled；系统 Python 未安装 FastAPI 时 smoke 会明确返回 `not_ready`，应使用项目锁定环境。
- 当前公开语料仍不满足正式元数据门禁：BM25 569,432 条仅 title 完整；semantic 31,136 条 title/abstract 完整，authors/year/venue/doi 缺失。两者均结构有效但 `required_fields_complete=false`，不得启动正式资格评测。
- 当前未导入服务器的**完整** evidence bundle；已导入的四个 inventory 只用于确认历史运行边界。Git 同步不能证明服务器实验数据与本地一致。服务器只能通过 `scripts/package_server_evidence.py` 导出脱敏证据后再审计，原始运行、模型、索引和凭据不上传 GitHub。
- **当前复核（2026-08-24）**：重新检查当前工作树、`outputs/imported_server_evidence/` 与 `outputs/server_evidence/` 后，仍只有四个 `server_legacy_inventory_v1` 清单（`contest_full_rules_v1`、`contest_full_dense_reranker_rrf_soft_v3`、两个 200 条历史运行）。它们的 `completion_status=unverified_legacy_inventory`、完成标记为空且未导出 `results.jsonl`；其中规则组与 Hybrid/Reranker 组的 profile、来源或预算不同，不能解除 P1-02，也不能证明服务器与本地数据一致。
- 本轮补充 `scripts/import_server_evidence.py`：本地导入端只接受 `server_evidence_bundle_v1`，校验成员、导出大小/SHA-256、必要文件和敏感文本后写入被忽略的 `outputs/imported_server_evidence/`；专项导入/篡改回归 `4 passed`。当前仍没有实际用户服务器 bundle，因此服务器实验一致性尚未得到证据证明。
- 本轮只读审计 `datasets/semantic/` 中的本地压缩包：`arxiv_data.csv.zip`（SHA-256=`8169c3ad…cff640a`）仅含 `titles/summaries/terms`，没有稳定 ID 或正式排序元数据；`arxiv_paper_abstracts.zip`（SHA-256=`d906a415…384157`）缺少 ZIP 中央目录，属于不完整副本。审计记录见 `docs/contest/local-data-audit-20260823.md`；两者均不进入正式语料或 GitHub 发布包。
- 本轮从当前真实 demo 批量运行发现中文查询“2021 年以后关于扩散模型用于医学图像分割”未解析中文年份边界和关键术语，导致标题型语料演示退化为泛化数据集结果。已在规则 Query Understanding 中增加“以后/之后/以来/起”年份解析及扩散模型、医学图像、图像分割、评价指标的保守中英映射；`tests/test_query_understanding.py` `42 passed`，同一 5 条 demo 批量运行仍为零网络/零 LLM，demo_03 从 0 条可见结果恢复为 5 条。该改动只改善可解释的查询理解，不改变默认 LLM 或线上来源策略。
- 已补做该查询理解改动的修复前/后成对诊断：固定同一 `docs/contest/demo-queries.jsonl`（SHA-256=`cf491fa6…880efa7`）、同一标题型 BM25 语料、同一 `balanced/top_k=5/current_year=2026` 和零网络/零 LLM；旧提交 `41ef325` 与当前提交 `07d2978` 的 5 条 case 均成功。demo_03 可见结果从 `0` 增至 `5`，其余 case 结果数保持 `5/5/4/1` 不变，中文时间范围从缺失变为 `2022+`。产物 `outputs/demo_query_understanding_pair_20260823.json` 仅作内部工程证据，未加载 gold/qrels，不是官方成绩。
- 当前 demo 的第二个真实缺口是英文“exclude review papers”未进入硬排除约束；已增加保守的中英文显式排除解析（仅在 `exclude/without/排除/不含` 等表达出现时触发），并新增 `45 passed` Query Understanding 回归。固定同一 5 条 demo、语料和零网络/零 LLM 的成对运行显示 demo_02 的 `excluded_terms` 从空变为 `review/survey/literature review`，其余 case 可见结果数不变；证据见 `outputs/demo_query_exclusion_pair_20260823.json`，不含 gold/qrels，不是官方成绩。
- 又补齐了命名方法/数据集的结构化抽取：`retrieval-augmented generation`、`graph neural network` 进入 `methods`，`HotpotQA`、`MoleculeNet` 进入 `datasets`。成对运行固定同一输入/语料/预算且零网络、零 LLM，demo_01/demo_02 的可见结果均保持 5 条，只增加结构化约束字段；回归为 `47 passed`，证据见 `outputs/demo_structured_facets_pair_20260823.json`，不含 gold/qrels，不是官方成绩。
- 中文“自动学术论文检索智能体，涉及查询改写、迭代搜索或相关性判断”现在将 `agent/query rewriting/iterative search/relevance judgment` 及其受控同义项进入 `methods`；专项回归为 `48 passed`。固定 5 条 demo 的成对运行保持零网络/零 LLM，前四条可见结果不变，demo_05 从 1 条增至 2 条与查询改写直接相关的候选；证据见 `outputs/demo_chinese_workflow_facets_pair_20260823.json`，不含 gold/qrels，不是官方成绩。
- 全量回归在 `2166 passed, 160 skipped` 处发现快照规划测试失败：第二轮 Query Evolution 生成的新检索键被错误地当作 RefChain 的未满足前置依赖，导致引用计划永远为空；该失败在旧提交也可复现，与中文解析改动无关。已修复 `SnapshotRuntime`：第一轮仍 fail-closed，后续轮次仅在上一轮收集完成且动态键的依赖快照全部存在时允许继续规划 RefChain；快照专项现 `24 passed`。
- 在提交 `4f72991` 上完成全量回归：`2312 passed, 184 skipped, 2 warnings`；本次修复未引入新的产品测试失败。跳过项仍代表已登记的外部历史证据、模型/GPU、官方 scorer、全文评测或平台条件缺失，不等同赛事资格通过。
- 本轮补齐前端查询理解可视化：结果页现在展示已解析的时间、方法、数据集、必含/排除、领域、论文类型和 venue 约束，并可展开查看规划 facets、来源、置信度和必需标记。该改动只消费现有 API `query_analysis`/`query_planning` 字段，不改变检索、排序或默认策略；后端 API/映射/查询理解专项 `92 passed, 1 warning`，前端 lint/build 通过。该项提升评委可见的可解释性，但不替代正式 Recall/F1 评测。
- 本轮补齐论文卡片的证据边界：每条结果会明确展示标题、摘要或许可全文证据层级，并列出当前缺失的摘要、年份、作者和发表场所。标题型离线语料不再只以“年份未知/摘要缺失”分散呈现，而会明确说明不能据此核验相应结论；该改动不改变检索、排序或质量策略，前端 lint/build 通过。
- 本轮修复查询理解真实缺口：英文关系词 `that/use/report/result/evaluate/exclude` 不再进入 `must_have_terms`，关键词末尾标点会被规范化；固定同一 5 条 demo、标题型 BM25 语料、`balanced/top_k=5/current_year=2026`、零网络/零 LLM 的成对运行中，demo_01/02/04 的约束更干净，demo_04 可见结果由 4 条增至 5 条，demo_01/02/03/05 分别保持 5/5/5/2。证据 `outputs/demo_query_cleanup_pair_20260824.json` 仅是内部无 gold 诊断，不是 Recall/F1 或官方成绩；查询理解专项 `50 passed`。
- 对公开 `AutoScholarQuery_test` 的 1,000 条查询只读取 `question` 做结构审计：问句/引导词在 `must_have_terms` 中的命中从大量存在降为 0；保留模型、方法、任务、数据集等实体词。随后在临时旧提交 `4753983` 与当前候选提交 `c579e8e` 上，以同一前 200 条顺序、同一 BM25 语料 SHA-256、身份映射、`high_recall`/300 候选、`top_k=20`、单 worker 和 `current_year=2026` 完成无网络/无 LLM 成对运行。结果：ΔRecall@20、ΔF1@10、ΔF1@20 均为 `0`，ΔF1@5=`-0.00143`（95% CI `[-0.00429,0]`）；无正向收益，故该改动只保留为查询理解/演示可解释性修复，不作为排序提升宣传。`scripts/analyze_paired_benchmark_runs.py` 现将 `runtime_code_hash` 报告为允许差异、仍严格校验数据/查询/预算等共享输入；专项 `4 passed`。配对产物位于 ignored `outputs/query_cleanup_pair_20260824_v2/`，仅为 legacy title+abstract 语料的内部工程指标，非官方成绩。
- **本轮查询规划修复（2026-08-24）**：固定 `docs/contest/demo-queries.jsonl` 前 5 条、同一标题型 BM25 语料、`balanced/top_k=5/current_year=2026`、单 worker、零网络/零 LLM 做前后成对运行。对包含命名数据集且同时包含方法维度的 demo_01/02，规则规划在三条查询预算内由 `original + generic intent + method` 调整为 `original + method + dataset`，`dataset_coverage` 从 `0` 提升至 `1`；两组前 5 条结果、可见结果数和其余三条 demo 均保持不变。专项 Query Understanding/规划/API 回归 `161 passed, 1 skipped`（1 warning）；证据位于 ignored `outputs/query_planning_priority_pair_baseline/` 与 `outputs/query_planning_priority_pair_candidate/`，只作内部演示诊断，不含 gold/qrels，不是官方成绩。
- **本轮策略成对资格诊断（2026-08-24）**：在当前 checkout、相同前 200 条 AutoScholarQuery、同一 title-only BM25 语料 SHA-256、arXiv 身份映射、`high_recall`/300 候选/`top_k=20`/单 worker/零网络零 LLM 下，`current_rules` 与 `facet_balanced` 的 `Recall@20/F1@5/F1@10/F1@20` 均 Δ=`0`，bootstrap 95% CI 均为 `[0,0]`；`facet_balanced` 平均延迟约 `1.10s` 对 `1.24s`，但没有质量收益，因此不切换默认策略。为使该比较可审计，`analyze_paired_benchmark_runs.py` 新增 `--allow-strategy-difference`：最多允许一个规划/演化/判断策略字段变化，仍严格校验所有数据、查询顺序、预算和资产输入；专项回归 `59 passed`。报告 `outputs/benchmark_runs/contest_qual200_facet_balanced_7514ade/paired_strategy_analysis_v2.json` 仅为内部工程指标，不是官方成绩。
- 本轮发布复核：`scripts/check_sync_state.py` 为 `status=ready`、`github_in_sync=true`；项目 `.venv` 下 clean-clone smoke 在干净树上为 `status=ready`，发布包 1,024 个文件、约 36.96 MB，manifest/成员哈希通过，health/config=200，离线 BM25 5 条，网络请求 0，LLM disabled。smoke 输出会绑定执行时的真实 `source_commit`，不以历史文档中的提交号代替。
- **本轮 CLI 复核（2026-08-24）**：发现直接执行 `scripts/check_contest_qualification.py --help` 时无法导入同目录模块；已让脚本显式加入仓库根目录，并新增子进程回归。资格门禁逻辑未改变，专项 `18 passed`，直接脚本入口现可用。
- **本轮发布验证 CLI 复核（2026-08-24）**：同样修复 `scripts/verify_contest_release_package.py --help` 的直接脚本导入路径，并新增子进程回归；发布安全校验逻辑未改变，发布验证专项 `5 passed`，直接入口现可用。
- **本轮最终发布复核（2026-08-24，提交 `1785789`）**：同步检查 `status=ready`、`github_in_sync=true`、工作树干净；clean-clone smoke 为 `status=ready`，source-only 包 1,024 个文件、36,970,565 bytes，manifest/成员哈希通过，源提交绑定 `1785789ab835162266c1128df650511f2e4b932e`，health/config=200，离线 BM25=5 条，网络请求=0，LLM disabled。该证据证明发布与可复现链路，不替代 P1 正式评测或官方成绩。
- **本轮参赛材料一致性复核（2026-08-24）**：说明书模板已移除无法由当前 checkout 核验的“soft Judgement 已完成内部 1000 条验证”表述，并将旧 1000 条对照明确标为历史线索；新增文档回归，防止历史运行被写成当前正式成绩。
- **本轮文档变更后发布复核（2026-08-24，提交 `7ad1df7`）**：文档一致性专项 `9 passed`；clean-clone smoke 重新通过，source-only 包 1,025 个文件、36,971,502 bytes，manifest/成员哈希通过，源提交绑定 `7ad1df74bca5d6505d3bff55f82145e2363dda1f`，health/config=200，离线 BM25=5 条，网络请求=0，LLM disabled。
- **本轮演示入口改进（2026-08-24）**：前端新增 5 条预置复杂查询下拉入口，覆盖时间/方法/数据集/排除条件/中文工作流；只填充输入框、不自动提交、不改变检索源，并明确不读取 gold/qrels。`npm run lint`、`npm run build` 和后端契约专项 `10 passed` 通过；该改动只改善可复现演示，不改变检索或排序策略。
- **本轮演示入口发布复核（2026-08-24，提交 `e34ec2c`）**：clean-clone smoke 通过，source-only 包 1,025 个文件、36,972,688 bytes，manifest/成员哈希通过，源提交绑定 `e34ec2caff5b4621d747f5fa5140c0059090ee41`，health/config=200，离线 BM25=5 条，网络请求=0，LLM disabled。
- **本轮演示批量复核（2026-08-24）**：标题型本地 BM25 批量运行发现中文“学术检索智能体”演示在无摘要语料下可见结果为 0；将该条改为等义英文查询后，固定 5 条 demo 全部 `succeeded` 且各返回 5 条结果。最新 manifest 绑定输入 SHA-256=`6ca87528538004e6fb1c3a755e674b6ac1d563f48f1d9001fdf53140dc0e9ad5`、当前代码提交、零网络/零 LLM、`gold_or_qrels_loaded=false`；新增 `tests/test_demo_queries.py` 锁定顺序、参数和 gold-blind 边界。该修复只改善演示可复现性，不改变检索或排序策略。
- **本轮最新演示验证（2026-08-24，提交 `a3835fb`）**：manifest 绑定当前提交 `a3835fb8bcc3b4f3bd3ca318f1b80b956bbbd08f`，5/5 成功、每条可见结果 5 条、失败 0、网络请求 0、LLM 调用 0、`gold_or_qrels_loaded=false`；前端 lint/build 通过。
- **上一轮代码提交复核（2026-08-24，代码 `f42a7e2`）**：查询规划分面修复后的全量回归为 `2325 passed, 184 skipped, 2 warnings`；前端 `npm run lint` 与 `npm run build` 通过；以干净提交执行的 5 条 demo candidate manifest 为 `git_worktree_clean=true`、5/5 succeeded、网络/LLM 调用均为 0。随后 clean-clone smoke 返回 `status=ready`，source-only 包 1,024 个文件、约 36.97 MB，health/config=200、离线 BM25 5 条、网络请求 0、LLM disabled；这些是内部工程/发布证据，不是官方成绩。计划记录随后在 `00f1934` 更新，最终 HEAD 仍以 `git rev-parse HEAD` 为准。
- 干净树全量回归（提交 `b2142d7`）：`2312 passed, 184 skipped, 2 warnings`（约 13 分 03 秒）。此前未提交文档时 clean-clone 门禁拒绝打包属于预期保护；提交后 smoke 已通过，未将该门禁误报为产品失败。
- 下方带日期的条目是历史增量证据，除非与本快照或当前可读取产物复核一致，不得作为当前状态引用。
- 本轮修复批量 CLI 来源白名单与生产 schema 不一致的问题：`scripts/run_search_batch.py --sources local_bm25/local_hybrid` 现在可用于离线复现，并在配置缺失时保留 fail-closed 行为；专项 `22 passed`。该改动不改变排序策略或默认在线来源。
- 补充 `docs/contest/demo-queries.jsonl`，提供与人工演示查询一致的前 5 条无 gold 批量输入；README 的批量离线命令现在指向真实存在、可从 GitHub 获取的文件。
- 修复批量 CLI 未初始化本地 connector 的真实缺口：选择 local BM25/Hybrid 时先按环境配置并建立索引；当前 `demo-queries.jsonl` 实际运行 5/5 succeeded，4 条返回论文，1 条因标题型语料缺少中文/时间元数据而返回 0 条并保留结构化 warning。该演示无 gold/qrels、无网络、LLM disabled，不是比赛成绩。
- 批量 CLI 现在对 local BM25/Hybrid 配置失败执行显式 preflight：行状态为 `failed`，错误为 `source_preflight_failed:<reason>`，不会伪装成 `succeeded + 0结果`；专项 `23 passed`。真实配置可用时仍保持离线批量演示链路。
- 批量 CLI 新增可选 `--manifest`：绑定输入/输出 SHA-256、当前 Git commit、工作树是否干净、case 成功/失败数、网络/LLM 调用数和 `gold_or_qrels_loaded=false`。专项 `24 passed`；该 manifest 是内部复现记录，不是官方成绩。
- 修正 `--fail-fast` 与 manifest 逻辑：触发首个失败后仍关闭输出句柄并生成完整 partial manifest，再返回退出码 1；避免中途 `return` 跳过复现记录。
- manifest 增加每个已执行 case 的来源、结果数量、warning 数和错误摘要，便于现场核对“实际走了哪个 connector”；专项 `24 passed`。不记录查询正文、gold/qrels 或本地绝对路径。
- 补齐 fail-fast partial manifest 回归：现在明确验证 `case_count=2`、`executed_case_count=1`、`partial=true` 和失败计数；专项仍为 `24 passed`。
- 每案例 manifest 现在明确区分 `visible_result_count`、`retrieval_raw_count`、`retrieval_deduplicated_count` 和 `judged_paper_count`，避免把展示结果数误称为召回指标；兼容保留的 `result_count` 明确等于 visible result 数。专项 `24 passed`。

## 当前审计快照（2026-08-23）

- [x] 已确认仓库包含多源检索、SQLite BM25、Dense/Faiss、Qwen3 Reranker、查询规划/演化、RefChain、LLM feedback、质量信号、全文证据模型、FastAPI/Next.js 和竞赛评测脚本。
- [x] `npm run lint` 与 `npm run build` 当前通过。
- [x] `PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q --maxfail=1` 已在当前 checkout 完整通过：`2267 passed, 185 skipped, 2 warnings`。跳过项均为结构化 preflight 标记的历史证据、冻结哈希、Windows 权限或外部环境阻塞；严格生产门禁仍在缺失/漂移时失败，不修改冻结哈希。
- [x] 当前 HEAD `480b1bd` 的完整回归复核通过：`2282 passed, 185 skipped, 2 warnings`（约 12 分 36 秒）。新增全文 CLI、元数据哈希绑定和 README 变更未引入测试回归；跳过项仍是已登记的外部历史证据/环境条件，不等同赛事资格通过。
- [x] 当前发布 HEAD `ac11eff` 的 clean-clone smoke 复核返回 `status=ready`：health/config=200、离线 BM25=5 条、网络请求=0、dotenv 读取=false，source-only 包 1015 个成员；本地未发现用户导出的服务器 evidence bundle，故不宣称服务器实验与本地一致。
- [x] 真实 FastAPI 本地搜索 smoke 复核通过：仅启用 `local_bm25`、50 候选、关闭 RefChain/LLM 时创建运行 HTTP 201，约 1.14 秒后终态 `succeeded`，API 调用/LLM 调用均为 0，结果接口 HTTP 200；`missing_abstract` 是当前标题-only 语料的真实限制，未被隐藏或当作成功质量证据。
- [x] 新增 `tests/test_real_local_bm25_api_smoke.py`，用临时两篇语料锁定真实 API 201→succeeded→200 结果链路、身份契约、零 API/LLM 调用和公开 `RankedPaper.paper.identifiers` schema；专项 `1 passed`。
- [x] 失败可解释性契约增强：`SearchRunStatusResponse` 现在公开可选 `error_message`，路由和前端类型同步，失败运行无需先请求结果接口即可展示原因；后端 API/集成专项 `21 passed`，前端 lint/build 通过。
- [x] 对本地语义语料中可精确关联的 31,136 条摘要做 BM25 负面消融：合并语料 SHA-256=`114a0397…7bbde34`，Recall@20 从 `0.06842` 降至 `0.06742`，配对 Δ=`-0.00100`（95% CI `[-0.02100,0.01800]`），F1@20 近乎不变且平均延迟 `1.19s→1.45s`。该输入不进入默认语料或发布包；结果留在 ignored `outputs/` 作为内部负面证据。
- [x] 已增加显式历史证据 preflight：`scripts/audit_cluster_significance.py preflight` 与 `scripts/check_current_rules_regression.py preflight` 返回 `external_evidence_unavailable`，严格 `check` 仍失败；默认测试只对已显式接入 preflight 的门禁做结构化跳过，未接入的门禁仍会严格暴露阻塞。当前全量回归已完成，剩余跳过项均有对应边界说明。
- [x] 已清理旧文档中不可核验的 `contest_full_dense_reranker_rrf_soft_v3`/P0 数字，并在干净提交 `e7f2b72` 重新生成可读的 200 条 BM25/Hybrid 成对诊断；完整元数据和 1000 条正式运行仍未完成。
- [x] 已核对并统一 `docs/report/technical_report.md`、`docs/architecture.md`、`docs/contest/experiment-results.md`、`docs/evaluation.md` 和 `README.md` 的当前证据边界；不可读取的服务器 Dense/Reranker/RRF 数字已删除或明确标为历史不可核验，仍保留实验协议和复现入口。
- [x] 只读比较上游 `solace47/ScholarNavigator` 最新 commit `345aadf` 与基线 `106891d...`：上游新增 LLM 候选选择、全篇获取和配对分析，但当前架构已有更严格的证据/门禁闭环；本轮只选择性移植全文重定向 allow-list 校验，未整体合并上游，也未在无 Provider/成对实验时启用 LLM 算法变化。
- [ ] 不连接服务器、不读取 `.env` 或 SSH 凭据；需要真实 Provider/GPU/官方 scorer 的任务只记录外部依赖，不伪造完成。

## 最新增量（2026-08-23）

- [x] 当前发布提交 `f67f1e4` 已与 GitHub `origin/main` 完全一致；只读远程核对确认本地 `HEAD == origin/main`，并修复了本地损坏的 `origin/contest-release` 跟踪引用（不改动任何源码分支）。API/发布专项 `23 passed, 1 warning`；重新构建 source-only 包共 1,016 个成员，clean-clone smoke 返回 `status=ready`、health/config=200、离线 BM25 5 条、网络请求 0、`dotenv_read=false`。该验收证明 GitHub 可下载源代码闭环，不包含服务器实验结果。

- [x] 在当前干净提交 `f67f1e4` 上完成同一前 5 条 AutoScholarQuery、同一 BM25 语料/身份/预算的规则查询演化 paired experiment：`paired_local_bm25_qe_baseline100_e92c9c4` vs `paired_local_bm25_qe_coverage_gap_e92c9c4`。两组 runtime code hash 与输入一致，Recall@20/F1@5/10/20 的 Δ 均为 `0.00000`，bootstrap 95% CI 均为 `[0, 0]`；查询演化没有可观测收益，因此不启用默认策略。该结果仅是 5 条内部离线诊断，不是官方成绩，正式资格仍需合格元数据和完整查询集。

- [x] 当前提交的 API/检索专项回归为 `48 passed, 1 warning`；前端 `npm run lint` 与 `npm run build` 均通过。该验证覆盖真实搜索失败状态、local BM25 API 链路、API 映射、Hybrid 连接器迁移和中文路径索引读写；未改变外部数据/模型阻塞状态。

- [x] 重新审计当前 checkout 的两个可疑元数据来源：`datasets/local_bm25/pasa_papers.jsonl` 为 569,432 条、arXiv ID 唯一，但 abstract/authors/year/venue/doi 完整度均为 `0.0`；`datasets/semantic/pasa_papers_with_abstracts.jsonl` 为 31,136 条、ID 唯一，title/abstract 完整度为 `1.0`，authors/year/venue/doi 完整度均为 `0.0`。仓库内 `id2paper.json` 仅是 ID→标题映射，不能解除 P1-01；未将推断字段写入语料。

- [x] 增加公开 API schema 回归：`tests/test_api_mock.py` 现在验证 `SearchRunStatusResponse.error_message` 在 OpenAPI 中保持可选 `string|null`，与真实失败状态路由和前端类型一致；专项 `21 passed, 1 warning`。这只加固失败降级演示契约，不改变检索排序。

- [x] local BM25 运行时配置现在展示索引构建时统计的 `title/abstract/authors/year/venue/doi` 字段完整度，并标注 `metadata_quality_scope=diagnostic_only`；缓存 schema 升级到 v4，旧索引自动重建，未生成索引时明确返回未知值。该能力只提升数据可信度与演示可解释性，不补齐或推断真实元数据，也不改变 BM25 排序；专项 `31 passed`。
- [x] 前端运行面板现在展示在线检索源数量、本地 BM25 配置/字段完整度和 LLM Provider 状态；未建立索引时显示未知，LLM 关闭时明确展示规则回退与不读取凭据。该改动只提升评委可见性和失败边界，不改变排序；前端 lint/build 通过。
- [x] 修正灾备历史输入 preflight：冻结 source commit 即使存在于本地对象库、但不是当前 checkout 祖先时，也返回 `external_evidence_unavailable` 并跳过历史模拟；严格运行仍拒绝 `source_commit_not_ancestor`。专项 `8 passed, 9 skipped`，未修改冻结协议哈希。
- [x] 修正正式验证 rehearsal/preregistration 的历史证据 preflight：冻结提交非当前祖先时显式标记不可用并跳过依赖历史 seal/package 的测试；严格 seal/包校验仍拒绝哈希漂移。专项 `10 passed, 15 skipped`，未修改冻结协议或补写历史证据。
- [x] 前端可复现构建 preflight 现在同时检查冻结提交祖先关系和 Windows symlink 权限；无 symlink 权限时结构化标记 `symlink_privilege_unavailable` 并跳过真实跨父目录构建，具备权限的平台仍执行严格字节一致性验证。专项 `5 passed, 1 skipped`。
- [x] Full1000 launch-control preflight 现在检查冻结提交是否为当前 checkout 祖先；历史协议分叉时返回 `external_evidence_unavailable` 并跳过仅依赖冻结运行的模拟，严格 `build_preparation` 仍拒绝 `source_commit_not_ancestor`。专项 `14 skipped`（当前冻结输入不可用）。
- [x] evidence transparency preflight 现在检查冻结提交是否为当前 checkout 祖先；历史对象存在但已分叉时返回 `source_blob_unavailable`，避免公共契约快照误进入严格构建。透明度/公共契约专项 `19 passed, 15 skipped`。
- [x] 当前 checkout 在上述 preflight 修复后完成完整后端回归：`2294 passed, 184 skipped, 2 warnings`（约 13 分 11 秒）。跳过项均有历史外部证据、冻结提交分叉或平台权限边界；没有修改冻结哈希、伪造指标或将跳过项写成赛事资格通过。
- [x] 前端检索源选择现在与 `/api/v1/runtime/config` 能力联动：未配置的 local BM25/Hybrid 选项置灰并说明依赖，运行时发现已选本地源不可用时自动回退到推荐组合；不改变检索排序或默认在线路径。前端 lint/build 通过。
- [x] 前端 LLM 查询理解/相关性判断开关现在与 runtime Provider 状态联动：Provider disabled 时禁用控件并说明规则版回退，Provider 可用时恢复可选；不读取或展示凭据，不改变默认关闭策略。前端 lint/build 通过。
- [x] 真实检索 API 对显式请求的 `local_bm25`/`local_hybrid` 增加只读配置预检：未配置时在排队前返回 HTTP 409 和稳定原因，不创建后台运行；配置存在时保持原异步索引/检索链路。专项真实 API/local BM25/API mock `30 passed`，不改变排序。
- [x] 扩展本地源入口预检：除环境变量存在外，BM25 corpus、Hybrid semantic corpus 和 model directory 必须实际存在；缺失时返回稳定 HTTP 409（不排队），正常配置端到端链路保持通过。专项 `31 passed`。
- [x] 增加 Hybrid 缺失 semantic corpus 的端到端 API 回归；显式 `local_hybrid` 请求在排队前返回 `local_hybrid_semantic_corpus_not_found`，专项真实 API/local BM25/API mock `32 passed`。
- [x] 补齐 Hybrid 复用的 BM25 corpus 入口预检与回归；显式 `local_hybrid` 在 BM25 corpus 缺失时返回 `local_hybrid_bm25_corpus_not_found`，专项真实 API/local BM25/API mock `33 passed`。

- [x] 修复前端失败轮询实际未使用 `status.error_message` 的缺口：失败运行现在优先展示状态接口提供的真实原因，仅在字段为空时回退请求结果接口；`frontend` 的 lint/build 均通过。该改动提升失败降级可解释性，不改变检索排序。

- [x] 在当前提交 `0c752c7` 重新运行 source-only clean-clone smoke：`status=ready`，health/config 均为 200，离线 BM25 返回 5 条，网络请求为 0，`dotenv_read=false`，发布包 1,016 个成员；确认前端失败处理修复未破坏队友从 GitHub 获取源码后的离线运行闭环。

- [x] 降低队友 clone 后的本地演示配置摩擦：`.env.example` 现在默认指向随 source-only 包提供的标题型 BM25 语料，保留 LLM/凭据空值；新增模板回归测试，专项 `10 passed`，并重新通过 clean-clone smoke（`status=ready`、health/config=200、离线 BM25 5 条、网络 0、dotenv 读取 false）。模板明确该语料不是元数据完整的正式竞赛语料。

- [x] 以 `.env.example` 作为 `SCHOLAR_AGENT_ENV_FILE` 完成一次真实 FastAPI local-BM25 端到端 smoke：runtime config 报告 `configured_from_env:569432_documents`，HTTP 201 创建运行，终态 `succeeded`，结果 HTTP 200 返回 2 篇，API/LLM 调用均为 0。首次索引加载可能超过 10 秒，受控轮询已确认最终成功；该结果仅验证演示链路，不是比赛成绩。

- [x] 当前最终核对（历史快照）：本地 `HEAD` 与 `origin/main` 均为 `a4b47ff`、工作树干净；环境模板、真实搜索 API、OpenAPI 错误契约专项合计 `30 passed, 1 warning`。这只证明当时源码发布闭环，不改变 P1 正式元数据/GPU/全文/LLM 外部阻塞。

- [x] 当前发布快照 `1172070` 工作树干净且已与 `origin/main` 同步；模板 API 搜索 smoke 与文档契约变更均已提交，未发现调试残留或 `git diff --check` 问题。

- [x] 发布 smoke 增加独立 `template_env` 阶段：在临时 clean clone 中复制 `.env.example` 为 `.env`，全新子进程验证 local BM25 `configured_from_env:569432_documents` 且 LLM 仍 `provider=disabled`；原有无 `.env` 的 hermetic 阶段仍保持 `dotenv_read=false`。专项 `2 passed`，最终 smoke `status=ready`。

- [x] 将模板 smoke 扩展为真实 API 201→succeeded→200 查询链路，并捕获首次 clean clone 暴露的 DOI 字段映射问题：标题型随包语料没有 DOI，`.env.example` 已将可选 DOI 映射留空；当前 smoke 验证候选数 `5`、返回结果 `2`、API/LLM 调用均为 `0`，并保留真实 `missing_abstract` warning。专项 `1 passed`，发布 smoke `status=ready`。

- [x] 同步演示脚本、PaSa 本地 BM25 手册和下一步执行手册：均明确先复制 `.env.example`、首次索引加载可能需要十几秒、LLM 默认关闭，并避免对标题型语料错误填写 DOI 字段。文档命令与当前模板/smoke 契约一致。

- [x] 在当前代码提交 `d93e047` 重新完成全仓库后端回归：`2285 passed, 185 skipped, 2 warnings`，耗时约 12 分 07 秒。未出现产品测试失败；185 个跳过项仍由显式 preflight/平台权限/缺失外部历史证据解释，严格生产门禁继续 fail-closed，不能写成赛事资格通过。

- [x] 在干净提交 `e7f2b72` 上完成同一前 200 条查询、相同 `high_recall`/300 候选预算的 BM25 与 Hybrid 配对运行：`contest_qual200_local_clean_e7f2b72` vs `contest_qual200_hybrid_clean_e7f2b72_retry`。两组 `code.dirty=false`、query 完整 200 条、失败日志为空且 runtime hash 一致；Hybrid ΔRecall@20=`0.04134`（95% CI `[0.01400,0.07179]`），ΔF1@20=`0.00687`（95% CI `[0.00301,0.01111]`）。输入仍是 legacy title+abstract 语料，authors/year/venue/doi 完整度为 0；因此这是当前 clean commit 的内部资格诊断，不是官方成绩，P1-01 未完成，不能自动启动 1000 条正式运行。
- [x] 新增 `docs/sync-and-release.md`，明确服务器实验结果位置、脱敏 bundle 边界、本地/GitHub 一致性判定和队友复现范围；不上传 `outputs/`、模型、索引、`.env` 或 SSH 凭据。
- [x] P1-01 增加 `scripts/merge_paper_metadata.py`：离线合并合法外部 JSONL 元数据，按稳定 arXiv ID 精确关联，只填充缺失字段；冲突、重复 ID、非法年份/身份和未匹配记录均可审计，默认不覆盖已有值。专项测试 `tests/test_merge_paper_metadata.py` 为 `3 passed`；当前真实语料仍未因该工具而虚构完整度。
- [x] 新增元数据合并器后的 clean-clone smoke 复核：`scripts/check_clean_clone_smoke.py` 返回 `status=ready`，API health/config=200，离线 BM25 5 条、网络请求 0、dotenv 读取 false，source-only 发布包 1013 个成员；新增脚本可从 GitHub 干净发布包获得，但语义模型/索引和完整元数据仍需单独准备。
- [x] P1-04 补充 `scripts/fetch_open_full_text.py` 显式全文证据入口：HTTPS、host allow-list、许可证确认、大小/页数限制和失败状态均由现有核心实现约束；新增 CLI 成功/未确认许可测试 `2 passed`。全文覆盖率、许可来源审计和独立 Evidence F1 仍未完成，不改变默认排序。
- [x] 元数据合并报告现在同时绑定 base、metadata 输入和 merged 输出 SHA-256；CLI 专项测试更新为 `4 passed`，避免只凭计数无法复现输入来源。
- [x] 对 Hybrid 候选池做 120/200/300 的同一前 200 条查询成对诊断：200 相对 120 的 Recall@20 增益为 `+0.00225`、F1@20 增益为 `+0.00082`，200 与 300 指标相同而 200 的候选预算更小；相对 BM25 的 200 配置 ΔRecall@20=`0.04359`（95% CI `[0.01600,0.07408]`）、ΔF1@20=`0.00769`（95% CI `[0.00367,0.01214]`）。因此仅将 `hybrid_deep_rrf` 诊断脚本的 BM25/semantic 候选上限从 120 调为 200；不改变默认产品排序，且该结论仍限于 legacy title+abstract 语料。

- [x] Benchmark CLI 的本地 BM25/Hybrid 语料身份改为显式必填：选择本地来源时必须同时提供 `--local-bm25-document-id-identity` 及对应字段映射（如 `arxiv_id` + `--local-bm25-arxiv-id-field arxiv_id`）。缺失时 fail-closed，避免使用默认 `s2orc_corpus_id` 导致评测运行成功但 gold 匹配静默全零。新增 3 个回归测试；专项 `tests/test_benchmark_runner.py` 为 `33 passed`。
- [x] 用正确 arXiv 身份配置重跑 5 条离线 BM25 smoke：RunId `local_bm25_smoke_identity_guard_a2c65d2`，Recall@20 `0.20`、F1@5 `0.067`、MRR `0.20`、成功率 `1.0`；该结果仅为本地 smoke，不作为正式比赛成绩。运行绑定当前工作树（dirty）及语料/索引哈希，产物留在被忽略的 `outputs/`。
- [x] 修复当前 Windows 本地 Hybrid 的两个可复现阻塞：旧 schema 语义索引若 corpus/model/shape/prefix 完全一致则复用已有 embedding 矩阵并只重建 ANN；Faiss 索引通过 Python Unicode-safe 序列化读写，避免中文工作路径下 `index.partial.faiss` 无法创建。新增迁移与中文路径回归测试，`tests/test_local_hybrid_connector.py` 为 `11 passed`；当前索引重建报告 ANN Recall@10=`0.998`。
- [x] 在同一前 200 条查询、相同 `high_recall`/300 候选预算下完成 BM25 与 Dense+RRF 配对诊断：RunId `contest_qual200_local_highrecall_pair_05759c1` vs `contest_qual200_hybrid_deep_rrf_current_05759c1`，两组绑定相同 `runtime_code_hash=5c018d9b…`，但运行时工作树为 dirty；ΔRecall@20=`0.0413`（95% CI `[0.0140,0.0718]`），ΔF1@20=`0.00687`（95% CI `[0.00301,0.01111]`），两组成功率 `1.0`。这是 legacy title+abstract 语料的内部诊断，不是干净提交的正式资格或赛事成绩；authors/year/venue/doi 完整度为 0，P1-01/P1-02 仍不勾选，正式资格需在干净提交上重跑。
- [x] 新增 `scripts/analyze_paired_benchmark_runs.py` 作为通用成对分析入口：只读取 config/metrics，校验共享输入与 query 顺序，输出 query-level bootstrap CI，并对路径字段脱敏；新增 3 个回归测试。修正 `run_contest_benchmark.ps1/.sh` 在 qualification 模式下统一使用 `high_recall` 与 300 候选预算，避免 baseline/candidate 因脚本默认 profile 不同而产生无效比较。
- [x] P1-03 增加可解释且不改变默认排序的质量信号：作者元数据、DOI 元数据、arXiv–DOI 身份一致性已从 `PaperQualityReport` 贯通到内部 `RankedPaper`、API 和前端质量面板；质量分仍只使用既有冻结信号，未知风险不扣分。新增一致性/不改变分数回归测试，后端质量/API 专项与前端 lint/build 通过。独立撤稿/重复证据仍需外部来源，尚未完成 P1-03 全部验收。
- [x] P2-02 的 source-only 基础再次验收：`scripts/check_clean_clone_smoke.py` 在当前 checkout 返回 `status=ready`，发布包 1010 个成员，API health/config=200，离线 BM25 返回 5 条，`network_request_count=0`、`dotenv_read=false`；该验收不替代 P1-02 正式资格、全文覆盖或赛事提交规格核验。

## P0：干净环境可复现性与反馈闭环

### P0-01 证据依赖清单与测试分层

- [x] **目标**：枚举所有测试/脚本引用的运行产物，区分仓库内 fixture、可重建输入和外部历史证据。
- **验收**：自动检查输出 JSON/JSONL 的相对路径、存在性、SHA-256 和来源类型；缺失外部证据必须输出明确 `external_evidence_unavailable`，不得静默使用空数据。
- **依赖**：现有 benchmark manifest、pytest 收集结果。
- **完成说明（2026-08-22，当前快照复核 2026-08-23）**：新增 `src/scholar_agent/evaluation/dev_plan_audit.py` 与 `scripts/audit_dev_plan.py`，只读扫描 317 个 benchmark JSON manifest、2,531 个路径引用；当前报告为 `1,775 present`、`38 present_unhashed`、`131 missing_external`、`247 missing_tracked`、`340 hash_mismatch`，其中哈希漂移包含本轮真实代码/计划修改，因此仓库仍未达到全量证据闭环。严格审计命令仍返回失败；专项 fixture 测试 `tests/test_dev_plan_audit.py` 已通过，不能把该快照写成全量 readiness。

### P0-02 修复默认测试闭环（不弱化历史门禁）

- [x] **目标**：让产品代码、单元测试和可重建评测在新 checkout 中可执行；历史证据门禁仍保持严格失败，不通过删测试、改断言或伪造历史产物解决。
- **验收**：默认测试只依赖仓库内可重建 fixture；历史门禁在缺失输入时返回结构化 blocker 报告，并由显式证据检查命令执行；所有测试命令和边界写入 `pytest.ini`/测试帮助文档。
- **依赖**：P0-01 的清单；不得修改历史 manifest 的哈希语义。
- **增量验证（2026-08-22）**：`cluster_significance` 新增 `preflight_cluster_significance_inputs()` 与 CLI `preflight`；缺失 `outputs/` 返回 `external_evidence_unavailable`，严格 `check` 仍返回失败。专项测试保持 `7 passed, 1 skipped`，未修改冻结 manifest 或历史哈希。
- **增量验证（2026-08-22）**：`current_rules_regression` 新增 `preflight_current_rules_inputs()` 与 CLI `preflight`；当前明确报告 3 组历史 Replay 的 run/snapshot/config/results 缺失，严格回放门禁仍保持失败。专项测试 `1 passed, 1 skipped`。
- **增量验证（2026-08-22）**：证据注册表新增 `preflight_evidence_registry_inputs()` 与 `scripts/check_evidence_registry.py preflight`；当前状态为 `baseline_drift`，明确指出实现文件哈希及 `docs/dev_plan.md` 的旧基线哈希不一致。严格 registry gate 仍保持失败；专项测试 `10 passed, 1 skipped`。
- **增量验证（2026-08-22）**：透明度日志新增 `preflight_transparency_sources()` 与 `scripts/check_evidence_transparency.py preflight`；当前明确报告源 commit `f764eb3c...` 不在本地 Git 对象库，严格日志构建仍保持失败。专项测试 `10 passed, 2 skipped`。
- **增量验证（2026-08-22）**：修复外部 scorer Windows worker 环境缺少 `SystemRoot` 的跨平台问题，并修正 formal backup CLI 测试的最小 Windows 运行时环境；外部 scorer `9 passed`，formal backup `18 passed`。
- **增量验证（2026-08-22）**：同样修正 formal backup member-discovery CLI 测试的 Windows 最小运行时环境；专项测试 `18 passed`。
- **增量验证（2026-08-22）**：formal backup enrollment 的 symlink 场景在无 Windows symlink privilege 时现在显式标记为环境跳过，不再伪装成产品失败；专项测试 `16 passed, 2 skipped`。
- **增量验证（2026-08-22）**：修正 formal backup set-topology CLI 测试的 Windows 最小环境，避免子进程因缺失 `SystemRoot` 输出 traceback；专项测试 `31 passed`。
- **增量验证（2026-08-22）**：修正 formal backup target-attestation CLI 测试的 Windows 最小环境；专项测试 `17 passed`。
- **增量验证（2026-08-22）**：修复 formal backup target-registration 在 Windows 上调用 POSIX 专用 `os.statvfs` 导致的异常。运行时现在使用可验证的可用字节数、对不可观测的 inode 容量保守报告为 `0`，并使不具备 POSIX 锁/目录同步能力的真实目标明确不能通过资格门禁；无创建符号链接权限时，profile simulation 返回 `simulation_incomplete`/`symlink_capability_unavailable`，不伪造覆盖。专项测试 `13 passed, 2 skipped`，`git diff --check` 通过。
- **增量验证（2026-08-22）**：formal evidence-quarantine CLI 现在先验证命令所需的协议与输入；只有 `audit-readiness` 才要求冻结 preregistration 绑定。因此历史哈希漂移仍使 readiness 严格失败，但不会掩盖无效协议或缺失证据的正确错误码。专项测试 `31 passed, 1 skipped`。
- **增量验证（2026-08-22）**：审计现存 `outputs/benchmark_runs` 后确认：`contest_full_local_baseline_v2/v3` 与 `contest_full_local_hybrid_v2` 有完整指标文件，但属于旧标题匹配语料；两者的 `runtime_code_hash` 不同，`compare_benchmark_runs.py` 明确拒绝比较。文档声称的 P0/Faiss、Dense/Reranker/RRF 完整 RunId 均不在本机，因此必须移除其未证实数字，不得将 legacy 结果当作候选收益。
- **增量验证（2026-08-22）**：扩展 `scripts/build_pasa_local_bm25_corpus.py` 的可选字段映射，保留 authors/year/venue/doi 并规范化 DOI；新增元数据转换测试，专项 `12 passed`。这只改善未来输入的字段保留，不改变当前 569,432 条标题库的既有缺失字段结论。
- **增量验证（2026-08-22）**：全量回归在 756 passed / 26 skipped 处暴露冻结 validation-freshness 外部证据缺失；测试现在识别结构化 `protocol_or_evidence_not_fresh` blocker 并跳过该历史门禁，不修改协议哈希。其余首个失败已修复前不会宣称全量通过。
- **增量验证（2026-08-22）**：修复 formal multivolume storage 在 Windows 上直接调用 `os.statvfs` 的异常；现在使用 `shutil.disk_usage` 获取可验证的可用字节，并将 free-inode 观测明确标记为 `not_available`，readiness 保持不合格。专项测试 `14 passed`。
- **增量验证（2026-08-22）**：全量回归在 `777 passed / 27 skipped` 处触及缺失的历史 network snapshot `results.jsonl`；依赖该历史输入的冲突测试现在显式报告环境跳过，生产 `_historical_snapshot_keys` 仍在缺失/漂移时 fail-closed。
- **增量验证（2026-08-22）**：network-request manifest 运行时现在在历史 results/config/snapshot 任一缺失时返回结构化 `historical_snapshot_inputs_unavailable`（退出 3），CLI 测试补齐 Windows 最小环境并对该外部 blocker 显式跳过；专项测试 `11 passed, 5 skipped`。
- **增量验证（2026-08-22）**：修正 formal provider-health supervisor 测试子进程缺少 Windows `SystemRoot/WINDIR` 的环境问题；专项测试 `20 passed`。
- **增量验证（2026-08-22）**：用户说明部分实验位于外部执行站点 `<server-project-root>`；本轮不连接该站点、不读取 SSH 凭据。服务器产物未纳入本地证据链，待用户自行导出脱敏的 `config.json`、输入/索引哈希、`metrics.json`、`results.jsonl`、`resource_ledger.json` 和完成标记后再审计。
- **增量验证（2026-08-22）**：灾备模拟协议绑定的历史 `source_commit=905b4d24...` 不在当前 Git 对象库。新增只读 `preflight_disaster_recovery_inputs()` 与 `scripts/check_formal_run_recovery.py preflight`；缺失时返回 `external_evidence_unavailable`/退出 3，`simulate-disaster` 不再抛 traceback 或绕过 commit 绑定。依赖该历史 commit 的 fixture 测试显式跳过；专项测试 `8 passed, 9 skipped`。冻结协议哈希未修改。
- **增量验证（2026-08-22）**：修正 `test_formal_run_storage_governance` 的 Windows 最小子进程环境，补充 `SystemRoot/WINDIR`，避免 Python `asyncio` 在隔离环境中因 WinSock 初始化失败而误报产品错误；专项测试 `17 passed`。
- **增量验证（2026-08-22）**：同样修正 `test_formal_scheduler_fairness` 两个 CLI 场景的 Windows 最小子进程环境；专项测试 `22 passed`。
- **增量验证（2026-08-22）**：formal validation clearance 的当前证据文件引用了不在本地对象库的历史 source commit，导致 fixture 对“无全局失败”作过强假设。新增只读 source preflight；历史 commit 缺失时该断言显式跳过，严格 `evaluate()` 仍保留 `source_commit_compatible` 失败。专项测试待回归确认。
- **增量验证（2026-08-22）**：clearance CLI 将当前历史证据注册检查限定在 `audit-current`；synthetic `evaluate/issue-receipt/verify-receipt` 不再被无关的注册表漂移阻断。`audit-current` 遇到注册表漂移返回结构化 `blocked`/退出 3，专项测试 `22 passed, 1 skipped`。
- **增量验证（2026-08-22）**：validation dress rehearsal 绑定的历史 source commit `a3ba3642...` 不在本地对象库。新增只读 `preflight_source_commit()`；依赖该 commit 的 fixture 在缺失时显式跳过，严格 `load_protocol`/CLI 仍拒绝未绑定历史输入。专项测试待回归确认。
- **增量验证（2026-08-22）**：dress rehearsal CLI 的 verify/simulate/readiness 测试也接入该 preflight；当前专项测试 `1 passed, 11 skipped`，没有把缺失历史对象伪装成 rehearsal 成功。
- **增量验证（2026-08-22）**：formal validation preregistration 绑定的 source commit `b983b23f...` 不在本地对象库，且注册脚本存在当前代码哈希漂移。新增只读 `preflight_registered_source()`；依赖历史 seal 的测试显式跳过，严格注册验证仍拒绝漂移/缺失历史证据。
- **增量验证（2026-08-22）**：frontend reproducible-build contract 绑定的 source commit `b0667d65...` 不在本地对象库。新增只读 `preflight_source_commit()`；真实跨父目录构建测试显式跳过，源码 release gate 仍保持不可用而不伪造构建结果。
- **增量验证（2026-08-22）**：Full1000 readiness 的冻结输入包含当前 `src/scholar_agent/prompts/manifest.json` 哈希漂移。新增只读 `preflight_frozen_inputs()`；依赖完整冻结计划的 fixture 显式跳过，严格 `build_plan()` 仍返回 `frozen_input_hash_drift`。
- **增量验证（2026-08-22）**：Full1000 launch-control 协议绑定的 source commit `3222b128...` 不在本地对象库。新增只读 `preflight_source_commit()`；依赖该历史 commit 的 fixture 显式跳过，严格 `build_preparation()` 仍拒绝 commit 不可验证。
- **增量验证（2026-08-22）**：gold metric semantics 回归依赖缺失的 historical replay run（如 `semantic_seed_resolution...`）。新增只读 `preflight_gold_metric_semantics_inputs()`；严格回归仍不读取替代数据，历史门禁测试显式跳过。
- **增量验证（2026-08-22）**：修复 `human_annotation_delivery` 在 Windows 默认 GBK 环境下读取 UTF-8 JSON/HTML 未指定编码的问题，统一显式 `encoding="utf-8"`；这是实际产品跨平台兼容性修复。
- **增量验证（2026-08-22）**：现有 precision annotation package 的冻结 `package_sha256` 与当前文件树不一致；相关历史人工标注 fixture 改为显式 `PackageNotEligible` 跳过，生产验证仍拒绝漂移，不生成伪造标签/统计。
- **增量验证（2026-08-22）**：submission-intake 的 CLI matrix 同样依赖该冻结人工标注包；在受控 violation（退出 2）下显式跳过历史 matrix 断言，readiness 仍严格退出 3。
- **增量验证（2026-08-22）**：human precision 的 real frozen-package gate 也检测到同一 package digest drift，现显式跳过并保留严格 `PackageNotEligible`；不把历史包当作可用人工证据。
- **增量验证（2026-08-22）**：人工标注/提交相关专项回归 `27 passed, 2 skipped`；最后一次全量回归在该历史 package drift 处停止（此前 `1033 passed, 98 skipped`），已覆盖剩余 real-package gate 的显式跳过。全量尚未重新跑到终点，不能宣称全仓库绿色。
- **增量验证（2026-08-22）**：judge-backend qualification 的 descriptive analysis 依赖缺失 `lexical_normalization_record160_813cf3a_r5` 历史 replay；其测试现对结构化 `repository_input_missing` 显式跳过，严格分析函数仍 fail-closed。
- **增量验证（2026-08-22）**：judge-backend qualification 专项依赖同一历史 replay 的 18 个测试现统一显式跳过（`18 skipped`）；这只分层外部输入，不把 fake provider 结果当作真实历史分析。
- **增量验证（2026-08-22）**：`llm_relevance_judging` 的历史 Record160 协议测试同样依赖缺失 replay；测试已统一显式跳过该外部输入，严格生产协议仍要求输入存在且 fail-closed。
- **增量验证（2026-08-22）**：`llm_relevance_judging_v1_1` hardened protocol 测试也依赖同一缺失 Record160 replay，已接入相同显式跳过条件；不改变生产输入校验。
- **增量验证（2026-08-22）**：全量回归在 `1306 passed, 136 skipped` 暴露 Windows 环境变量 `HOME` 缺失导致的隐私包测试误报；测试现在仅在 HOME 非空时检查其内容，避免把空字符串误判为泄漏。专项待回归确认。
- **增量验证（2026-08-22）**：offline wheelhouse release-contract 测试依赖缺失历史 source commit `d6d37eb1...`；该测试现对 `git_input_unavailable` 显式跳过，严格 release contract 仍不生成伪造结果。
- **增量验证（2026-08-22）**：paired significance 的 frozen lexical replay 依赖缺失 `lexical_normalization_v1_005794c_replay_r5/aggregate.json`；测试现显式跳过该历史证据，严格审计函数仍拒绝缺失输入。
- **增量验证（2026-08-22）**：修复 portable execution-site kit 在 Windows 上导入 POSIX 专用 `fcntl`、调用 `statvfs` 和目录 `fsync` 时崩溃的问题。运行时现在按平台使用标准库锁实现，磁盘可用空间可观测而 inode/目录同步不可观测时明确判为未就绪；CLI 测试子进程补齐最小 Windows runtime 环境。专项测试 `18 passed`，不把 Windows 本机作为合格外部执行站点。
- **增量验证（2026-08-22）**：新增 `scripts/package_server_evidence.py` 与 `docs/contest/server-evidence-sync.md`，把服务器完整 RunId 导出为脱敏、哈希可审计的证据包；工具只允许白名单产物、拒绝凭据字段/未完成运行，并明确不连接服务器或读取 SSH/`.env`。专项测试 `2 passed`；原始 bundle 仍应留在被 `.gitignore` 忽略的本地证据目录，不上传 GitHub。
- **增量验证（2026-08-22）**：修正 provider capacity intake CLI 测试的 Windows 最小子进程环境；专项测试 `22 passed`。全量回归此前在 `1447 passed / 138 skipped` 处因该环境问题停止，待重新从头执行。
- **增量验证（2026-08-22）**：公开契约检查发现后端 `Paper`/`ConnectorDiagnostics` 已新增字段而前端类型未同步；补齐全文证据结构、段落证据和本地模型诊断字段，前端 `npm run lint` 与 `npm run build` 均通过，契约专项为 `9 passed, 13 skipped`。历史透明度源缺失仍只在依赖冻结 snapshot 的测试中显式跳过。
- **增量验证（2026-08-22）**：全量回归在 `1492 passed / 151 skipped` 暴露冻结 Python 3.12 依赖锁与当前 Windows/Python 3.13 环境不一致；依赖该冻结环境的 manifest 测试现在显式跳过，生产 `build_manifest/verify_manifest` 仍返回 `environment_identity_mismatch`，不修改冻结锁。
- **增量验证（2026-08-22）**：后续全量回归在 `1579 passed / 153 skipped` 暴露缺失的 query-gold leakage 历史 replay；测试现在对缺失/冻结代码哈希漂移显式跳过，生产回归仍报告 `repository_input_missing` 或 `input_or_protocol_drift`，不使用替代 baseline。
- **增量验证（2026-08-22）**：最新回归在 `1588 passed / 154 skipped` 暴露同一缺失 replay 被 query-independence gate 引用；生产回归新增 `repository_input_missing` 结构化结果，相关历史 fixture 显式跳过，不伪造 cluster 统计。
- **增量验证（2026-08-22）**：最新回归在 `1625 passed / 156 skipped` 暴露查询规划测试在 Windows 默认 GBK 下读取 UTF-8 输出；补充显式 UTF-8 解码，避免把编码环境误报为产品失败。
- **增量验证（2026-08-22）**：查询规划专项最终 `9 passed`；补齐所有 UTF-8 输出断言并推送提交 `9749261`。全量回归此前继续发现的均为同类历史证据/Windows 编码分层问题，未改变生产算法或评测口径。
- **增量验证（2026-08-23）**：发布包构建测试 `3 passed`；source-only 包现包含可公开分发的 `datasets/local_bm25/pasa_papers.jsonl`，仍排除 semantic 数据、模型、outputs 和 `.env`。`check_clean_clone_smoke.py` 在全新临时目录完成 `compileall`、`/api/v1/health=200`、`/api/v1/runtime/config=200`，离线 BM25 返回 5 条结果，`network_request_count=0`、`dotenv_read=false`。这证明代码与可公开的小型/完整标题语料可从 GitHub 干净克隆后运行，但不代表正式评测、GPU 模型或线上 Provider 已就绪。
- **增量验证（2026-08-23）**：全量回归在 `1634 passed / 156 skipped` 暴露冻结 AutoScholarQuery query-planning manifest 的 prompt 清单哈希漂移。新增 `preflight_query_planning_inputs()` 与 `scripts/audit_autoscholar_query_planning.py preflight`；默认门禁在历史输入不可审计时显式跳过，严格 `check` 仍 fail-closed，不修改冻结 baseline/hash。专项为 `1 passed, 1 skipped`。
- **增量验证（2026-08-23）**：第二轮全量回归在 `1728 passed / 157 skipped` 暴露 Windows `ssh-keygen` 子进程缺少 `SystemRoot/WINDIR` 的运行时环境，导致测试签名生成误报 `test_key_generation_failed`。补齐最小平台环境变量，未传递凭据或 Provider 配置；专项回归待确认。
- **增量验证（2026-08-23）**：release authenticity 专项进一步确认 Windows `st_mode` 不能等价审计 OpenSSH ACL；私钥签名/验证本身已通过，权限位断言与 synthetic matrix 现在仅在 POSIX 检查 `0600`，Windows 明确依赖 ACL/系统 OpenSSH 保护，不把不可观测的 mode bits 当作失败。
- **增量验证（2026-08-23）**：第三轮全量回归在 `1748 passed / 157 skipped` 暴露 release-candidate reproducibility 测试引用的历史 source commit `a743c59c...` 不在当前对象库。该模块已有严格 `preflight_source_commit()`；依赖 materialization 的两个历史 fixture 现显式跳过，生产 `materialize_source()` 仍对缺失 commit 返回 `git_input_unavailable`，不伪造发布产物。
- **增量验证（2026-08-23）**：第四轮全量回归在 `1777 passed / 160 skipped` 暴露 reproduction capsule 外部依赖测试需要 Windows symlink privilege（WinError 1314）。测试现在显式报告环境跳过；归档验证逻辑仍拒绝 symlink/hardlink，未降低安全门禁。
- **增量验证（2026-08-23）**：第五轮全量回归在 `1924 passed / 161 skipped` 暴露 runtime hermeticity worker 在 Windows 极简环境中缺少 `SystemRoot/WINDIR` 等启动变量，所有 profile 因 worker 非零退出而误报。补齐仅供启动器使用的平台变量（不加入业务允许环境），专项待确认。
- **增量验证（2026-08-23）**：第六轮全量回归在 `2183 passed / 161 skipped` 暴露 standalone auditor bundle 依赖的历史 preregistration registered-file 哈希漂移（cluster/signing/clearance/preregistration 实现文件已被后续真实修改）。bundle 构建测试现在先做只读漂移 preflight 并显式跳过；严格 `_audit_preregistration()` 仍拒绝漂移，不重写 seal 哈希。
- **增量验证（2026-08-23）**：第七轮全量回归在 `2229 passed / 178 skipped` 暴露 untrusted-metadata isolation 历史 protocol 绑定的 prompt manifest 哈希漂移。依赖该冻结 protocol 的测试现在显式跳过并保留严格 `load_protocol()` 拒绝漂移；纯文本/URL/observer 防注入单测不受影响。
- **增量验证（2026-08-23）**：最终轮回归在 `2243 passed / 184 skipped` 暴露 validation-evidence-freshness 冻结 inventory 与当前真实实现提交/组件摘要不一致（276 个历史项 stale）。当前库存测试现显式跳过并保留严格 `verify_current()` 失败；不能把修改后的治理代码冒充旧 freshness seal。
- **最终回归（2026-08-23）**：上述分层修复后，`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q --maxfail=1` 完整通过：`2264 passed, 185 skipped, 2 warnings`，耗时约 12 分钟。跳过项均有对应 preflight/环境说明；没有修改冻结哈希、删除门禁或伪造外部实验结果。该计数证明当前 checkout 的可重建代码与测试闭环通过，不等于官方比赛 scorer 或历史 GPU/Provider 指标已完成。
- **增量验证（2026-08-22）**：P0-03 发布 smoke 已完成：source-only 包全新目录 `compileall` 通过，`/api/v1/health` 返回 200；从当前 Git HEAD 归档到全新目录后，API health 返回 200，使用 GitHub 可获得的 `datasets/local_bm25/pasa_papers.jsonl` 进行离线 BM25 检索返回 5 条结果（网络 0）。前端 lint/build 已通过。原始大型索引、semantic 语料和正式评测仍不随 Git 发布。
- **增量验证（2026-08-22）**：同一 Git clone smoke 的 `/api/v1/runtime/config` 返回 200，明确报告 LLM `provider_disabled`、local connectors 未配置和真实 API connector 能力；没有把未配置 Provider 或本地索引写成已就绪。当时仅完成 smoke，后续已补充全新环境依赖安装验收。
- **增量验证（2026-08-23）**：clean-clone smoke 已重复验证并写入 `outputs/clean_clone_smoke_20260823.json`（该目录受 `.gitignore` 保护，不上传 GitHub）。结果为 `status=ready`、API 两个 200、local BM25 五条结果、零网络请求；source-only release 成员数 1005，明确不含 `.env`、`outputs/`、semantic 语料或模型缓存。该次检查当时尚未执行全新环境依赖安装，后续已补齐。
- **增量验证（2026-08-23）**：最终 clean-clone smoke 写入 `outputs/clean_clone_smoke_20260823_final.json`，仍为 `status=ready`、health/config 200、离线 BM25 5 条、网络请求 0、dotenv 读取 false；发布包成员数 1006（新增脚本/测试后），仍不含 `.env`、outputs、semantic 数据或模型缓存。该产物仅保留本地，不上传 GitHub；实际依赖安装随后在全新 GitHub clone 中完成。
- **增量验证（2026-08-23）**：扩展 clean-clone smoke 生成并校验 `offline-search-result-v1` 结构化离线结果（5 条、稳定 rank、arXiv ID/title/source、无 gold 字段），同时验收 `requirements.txt`、`frontend/package.json` 与 lockfile 在 source-only 包中存在；专项测试 `2 passed`，最终 smoke 为 `status=ready`、网络 0、dotenv 读取 false。仍未把“依赖实际安装”误写成已完成。
- **完成判定（2026-08-23）**：P0-02 已完成。默认回归在当前 Windows checkout 通过，外部历史输入由显式 preflight 分层；生产严格 `check/evaluate` 命令仍 fail-closed。结构化离线结果导出已经通过 clean-clone smoke。
- **最终回归（2026-08-23）**：加入元数据严格门禁后重新运行全仓库：`2266 passed, 185 skipped, 2 warnings`，耗时约 12 分 23 秒。跳过项仍是有明确 preflight/外部输入说明的历史证据，不修改冻结哈希、不使用 gold 替代物。
- **最终回归（2026-08-23）**：全文重定向 allow-list 修复后重新运行全仓库：`2267 passed, 185 skipped, 2 warnings`，耗时约 12 分 22 秒；全文专项为 `8 passed`，前端 lint/build 通过。没有改变排序策略或评测口径。

### P0-03 发布包与文档状态一致

- [x] **目标**：统一代码实际能力、评测结果、限制和发布包内容，删除过期“已完成/未实现”冲突描述。
- **验收**：从全新目录安装后可完成 smoke、前端构建、API 启动和离线评测；报告中的每个指标都能链接到实际产物，明确标注“内部指标/非官方成绩”。
- **依赖**：P0-02；不要求历史 `record160` 或官方 scorer 可用。
- **增量验证（2026-08-22）**：审阅并修正 `docs/contest/submission-checklist.md`、`submission-report-template.md`、`experiment-protocol.md` 中无法由当前 checkout 核验的 P0/Faiss、Dense/Reranker/RRF 完整运行数字；改为待重新运行/不可审计状态，并保留内部指标与官方成绩边界。`technical_report.md`、`architecture.md`、`experiment-results.md` 未发现同类具体 RunId 数字残留。该次记录是在全新依赖安装前，现由后续验收补充。
- **增量验证（2026-08-22）**：构建 source-only release audit 包 `outputs/release-audit-20260822.zip`，共 995 个成员；检查确认无 `.env`、`outputs/`、`datasets/semantic/`、服务器 IP 或本地绝对路径，构建器标记 `internal_metric_scope=not_official_competition_scorer`。当时尚未完成实际依赖安装，现已由全新 clone 验收补齐。
- **完成判定（2026-08-23）**：从 GitHub `main` 的代码 commit `f0e049a` 全新浅克隆后，在全新 Python 3.12 虚拟环境安装 `requirements.txt` 与 `requirements-dev.txt`，前端执行 `npm ci`；`check_clean_clone_smoke.py` 返回 `status=ready`，API health/config 均返回 200，离线 BM25 返回 5 条且网络请求为 0，发布包成员数 1008；`npm run lint` 与 `npm run build` 均通过。该验收证明 source-only 项目的干净环境可复现，不代表语义模型、GPU Reranker、LLM Provider 或官方 scorer 已就绪。

## P1：最终效果（召回/F1 → 质量过滤 → 全文证据）

### P1-01 元数据质量与本地索引契约

- [ ] **目标**：确认正式语料按稳定 arXiv ID 精确关联，并覆盖 title、abstract、year、authors、venue、DOI 等排序所需字段；标题匹配旧语料只能作为 legacy。
- **验收**：构建器拒绝无稳定 ID 的记录；报告输入/输出计数、缺失字段、语料哈希和索引哈希；同一输入重复构建得到字节稳定结果。
- **依赖**：可读取的 PaSa/arXiv 元数据；不读取 AutoScholarQuery gold 生成语料。
- **当前证据（2026-08-22）**：新增 `scripts/audit_corpus_metadata.py` 与 `corpus_metadata_audit.py`。现有 `datasets/local_bm25/pasa_papers.jsonl` 为 569,432 条、arXiv ID 唯一，但 abstract/authors/year/venue/doi 完整度均为 0；`datasets/semantic/pasa_papers_with_abstracts.jsonl` 为 31,136 条、ID 唯一，title/abstract 完整度为 1.0，其余字段完整度均为 0。构建器已支持未来官方元数据中的 year/venue/doi，并新增对应报告字段；专项测试 `10 passed`。因此该任务尚未完成，当前排序仍受元数据缺口限制。
- **增量验证（2026-08-22）**：`local_hybrid._paper_from_semantic_row()` 已将 authors/year/venue/doi 接入 `Paper`/`PaperIdentifiers`，含 DOI 规范化与年份范围校验；新增元数据映射测试通过。现有语料仍缺字段，故不能宣称排序元数据质量已达标。
- **增量验证（2026-08-22）**：`local_hybrid` 读取语义语料时现在强制稳定 arXiv ID、规范化版本号并拒绝重复 ID；专项本地连接器/API/构建器测试 `26 passed`。这完成了索引入口契约的一部分，但真实语料的元数据补齐仍未完成。
- **增量验证（2026-08-22）**：语义索引 `metadata.json` 现在记录 title/abstract/authors/year/venue/doi 的 `field_completeness`，与语料 SHA-256、文档数和 ANN 指纹一起持久化；索引测试通过。当前真实语料仍以 `authors/year/venue/doi=0` 为主，P1-01 不勾选完成。
- **增量验证（2026-08-23）**：新增离线元数据合并器和冲突测试。它只提供可复现的输入管线，不生成或推断作者、年份、venue、DOI；在获得合法外部元数据前，P1-01 仍保持未完成。
- **增量验证（2026-08-23）**：元数据审计新增 `--require-fields` 严格门禁，并区分 `structural_passed` 与 `required_fields_complete`；对当前 BM25 语料使用 title/abstract/authors/year/venue/doi 全字段要求时结构门禁通过但严格结果为 `passed=false`。当前 BM25 语料 SHA-256 为 `ede3bd1b…d102f28`（569,432 条），semantic 语料 SHA-256 为 `20ecf5d3…e234bcb`（31,136 条），缺失排序元数据仍是外部数据依赖，未伪造完成。
- **BLOCKED（外部输入，2026-08-23）**：P1-01 仍缺少可合法审计的完整元数据 JSONL（稳定 arXiv ID、abstract、authors、year、venue、DOI 及来源/许可说明）。所需输入：用户导出的脱敏元数据文件及其来源、许可和 SHA-256；在此之前不得勾选完成或启动正式 1000 条评测。
- **增量实现（2026-08-23）**：local BM25 索引现在持久化并返回 `authors/year/venue`（支持数组、逗号分隔作者和嵌套字段），这些字段明确不进入 FTS 文本，因此不改变排序；缓存 schema 升级到 v3，旧索引自动失效重建。专项 local BM25/Hybrid/API `30 passed`，但正式 P1-01 仍因真实完整元数据输入缺失而保持 BLOCKED。
- **最终验证（2026-08-23，提交 `777f260`）**：local BM25/Hybrid/API 专项 `30 passed`；clean-clone smoke 为 `ready`，source-only ZIP `36,936,009` bytes、1,020 个文件，模板环境检索返回 2 条结果且 API/LLM 调用为 0，新增 metadata 字段不会破坏队友复现。正式 P1-01 仍等待合法完整元数据。
- **增量验证（2026-08-23，提交 `849213d`）**：补充结构化作者对象（`name`/`full_name`）规范化回归；local BM25/Hybrid/API 专项 `31 passed`，同步审计 `status=ready` 且本地与 GitHub 一致。该改动只改善元数据展示，不改变排序。
- **增量验证（2026-08-23，当前工作树）**：运行时配置现在同步暴露 local Hybrid 索引的 `field_completeness`，前端运行面板同时展示 BM25 与语义混合语料完整度；字段标记为 `diagnostic_only`，不进入排序，也不代表 P1-01 完成。后端专项 `31 passed`，前端 lint/build 通过。

### P1-02 召回/F1 成对资格与完整评测

- [ ] **目标**：在同一查询顺序、数据、候选预算、模型和资源约束下比较 rules、Dense、Reranker、RRF/软 Judgement。
- **验收**：固定 200 条资格门禁通过后才允许 1000 条；报告 candidate recall、F1@5/10/20、Recall@20、MRR、延迟、调用数、失败率和显著性区间；任何无收益策略保持默认关闭。
- **依赖**：P1-01；GPU/模型或 Provider 不可用时只完成离线门禁，不伪造成绩。
- **BLOCKED（继承 P1-01，2026-08-23）**：完整元数据资格尚未通过；此外正式 Dense/Reranker 资格还需要可审计 GPU 模型、设备和资源账本。所需输入：P1-01 合格语料与可读取的 GPU/模型运行产物。
- **增量实现（2026-08-24）**：正式 runner 不再锁死旧 `datasets/` 资产路径；通过显式资产参数与 plan-only 审计，可在不读取 `.env`、不访问网络或创建运行目录的条件下，先证明 v3 Hybrid baseline 与仅开启 Reranker 的 candidate 共享语料、索引、BGE、预算、profile、查询范围和候选池。代码测试及 Windows 参数实测通过；待服务器完成两个新的 RunId 并导入完整脱敏 bundle 后，才可执行最终配对分析与资格判断。

### P1-03 低质量论文过滤（可解释、默认保守）

- [ ] **目标**：把元数据完整度、身份一致性、撤稿证据和重复风险与相关性分数分离，形成可解释诊断；只有独立来源明确 `flagged` 时才允许软影响排序。
- **验收**：未知风险不扣分、不硬过滤；Crossref/arXiv 回执带稳定身份、来源和哈希；quality policy 通过成对资格且有正向区间后才能考虑默认启用，否则保留负面结论。
- **依赖**：P1-02 的固定候选；外部来源不可用时保留 `unknown`。
- **增量验证（2026-08-23）**：前端 PaperCard 现在可折叠展示后端的排序理由、匹配约束及已启用时的质量策略/质量分/贡献/排名变化原因；未启用质量策略时不显示。该改动只增加解释层，不改变排序；独立撤稿/重复证据和成对收益门禁仍未完成。
- **BLOCKED（外部证据，2026-08-23）**：独立撤稿/重复来源证据 ledger 尚未提供。所需输入：带稳定论文身份、来源记录 ID、来源/许可和文件哈希的脱敏证据文件；在此之前风险状态保持 `unknown`，不启用质量过滤排序。

### P1-04 全文证据展示与定位

- [ ] **目标**：选择性移植上游的受限 PDF/HTML 获取、段落选择和定位映射，不整体替换现有 SearchService。
- **验收**：只接受 HTTPS/allow-list/许可可核验来源；PDF/HTML 失败时保留摘要结果；证据含 URL、内容哈希、页码/段落或章节定位；QASPER 或等价公开数据有独立 Evidence F1 基线；全文默认不改变论文排序，除非成对实验证明收益。
- **依赖**：P1-01；锁定 `pypdf`/解析依赖；真实开放全文仅做受限验证。
- **增量验证（2026-08-23）**：前端 `PaperCard` 已展示后端已验证的全文证据文档：安全校验后的来源 URL、许可 ID、内容 SHA-256、段落编号、字符起止位置、段落 SHA-256 和原文。该改动只增加可折叠展示，不参与排序，也不代表全文获取、许可覆盖或 Evidence F1 基线已完成；`npm run lint` 与 `npm run build` 通过。
- **增量验证（2026-08-23）**：Markdown 导出现在同步包含全文证据的来源、许可证、文档/段落 SHA-256、字符位置和段落文本；不改变 JSON 原始结构或排序。前端 lint/build 通过。全文获取覆盖、许可核验和独立 Evidence F1 仍未完成。
- **增量验证（2026-08-23）**：选择性借鉴上游的边界控制思路，全文获取现在对 urllib 重定向目标再次执行 HTTPS/allow-list/凭据/端口校验；跨域重定向专项测试通过。该修复不参与排序，但真实开放全文覆盖和 Evidence F1 仍未完成。
- **BLOCKED（外部评测，2026-08-23）**：缺少可合法使用的开放全文覆盖清单及 QASPER 或等价公开 Evidence F1 标注/评测输入。所需输入：许可明确的全文 URL/许可证清单和独立评测数据；现阶段仅保留受限 CLI 与结构化证据能力。

## P2：受控智能增强与发布演示

### P2-01 LLM 搜索增强

- [ ] **目标**：保留当前 `llm_feedback` 默认关闭；补齐上游 controller 的可观察动作、预算、Schema、Record/Replay 和失败回退中真正缺失的部分。
- **验收**：每查询最多一次调用和一条反馈查询；原始查询保留；Provider/Schema/预算失败不破坏首轮；5 条 smoke、200 条资格和必要的 1000 条成对实验均无 fallback 且质量/成本门禁通过后，才允许进入演示主路径。
- **依赖**：P0 完成、P1-02 完成、可用且隔离的 Provider；不可用时标记 `BLOCKED`，不修改 `.env` 规避。
- **BLOCKED（外部 Provider，2026-08-23）**：当前 runtime 明确为 `provider_disabled`，没有隔离、可复现的 LLM Provider/模型/预算输入。所需输入：不含个人凭据的 Provider 连接配置或本地 Record/Replay 证据；在此之前保持默认关闭。

### P2-02 发布包、启动脚本与演示

- [ ] **目标**：形成评委可运行的 source-only 发布包、5 分钟视频脚本和一组不依赖 gold 的演示查询。
- **验收**：压缩包不含 `.env`、密钥、临时输出和超限模型缓存；全新目录完成安装、后端 health/config、前端 build、一次本地检索、结构化导出；演示明确展示查询理解、召回、排序依据、证据定位、成本和失败降级。
- **依赖**：P0-03、P1-02、P1-04；官方提交格式和大小限制以赛事页面为准。
- **增量验证（2026-08-23）**：新增 `docs/contest/demo-script.md`，提供 5 分钟录屏时间线、复杂查询、证据定位、导出、成本和失败降级展示要点；使用已有 `demo-queries.md`，不加载 gold/qrels。P2-02 仍待 P1-02 正式资格评测和最终提交规格核验后勾选。
- **增量验证（2026-08-23）**：发布构建器现在在 source-only ZIP 内写入确定性 `release-manifest.json`，记录源 commit、成员数量、每个成员的字节数/SHA-256 和排除边界；ZIP 成员时间戳固定，重复构建字节一致。新增发布包回归 `3 passed`。这完成了发布包的可审计性子目标，但 P1-02 正式资格和赛事最终规格仍未满足，因此不勾选 P2-02。
- **增量验证（2026-08-23）**：发布构建器现在拒绝 dirty Git 工作树，避免 ZIP 内容来自未提交修改而 manifest 误报 `HEAD`；新增 dirty-tree 拒绝回归。当前代码变更提交并通过 clean-clone smoke 后，才可重新生成最终包。
- **增量核对（2026-08-23）**：赛事公告公开附件明确源码 ZIP 上限 200 MB、说明书 300 页、视频 5 分钟/200 MB，截止 2026-09-01 23:59。当前 Git 跟踪文件约 273 MB，其中随包标题型 BM25 语料约 92 MB；实际 source-only ZIP 为约 37 MB，已低于上限，但构建器仍新增 200 MB 硬门禁，避免后续资产变更静默超限。
- **增量验证（2026-08-23）**：新增 `scripts/verify_contest_release_package.py`，clean-clone smoke 现在自动验证 ZIP 成员路径、manifest、源 commit、文件数量和每个成员 SHA-256；篡改成员专项会 fail-closed，发布验证专项 `6 passed`。这完成发布包完整性子目标，但 P2-02 仍依赖最终正式评测和官方提交材料。
- **最终验证（2026-08-23，提交 `6ef3aae`）**：发布验证/构建/clean-clone 专项 `7 passed`；clean-clone smoke 为 `status=ready`，source-only ZIP `36,933,278` bytes、1,019 个源文件，manifest 与 ZIP 成员哈希全部通过，health/config=200，离线 BM25 5 条，网络请求 0，LLM disabled。P2-02 的发布完整性子目标完成，但正式评测和最终提交材料仍未完成。
- **增量安全修复（2026-08-23）**：发布验证器独立拒绝 `.env`、密钥、`outputs/`、语义模型/索引和 legacy 源码路径，即使 ZIP 内 manifest 自洽也不放行；专项 `3 passed`。这避免验证器只证明完整性而漏掉发布边界。
- **最终发布安全复核（2026-08-23，提交 `ad39fe9`）**：公开文档中的服务器绝对路径已替换为 `<server-project-root>` 占位符；验证器新增 Markdown/JSON/JSONL 等文本内容扫描，拒绝服务器地址；发布验证专项 `8 passed`，clean-clone smoke 仍为 `ready`，本地与 GitHub commit 一致。
- **增量修复（2026-08-23）**：发现测试夹具也会随 source-only 包发布，已移除其中的服务器路径字面量，并取消验证器对 `tests/` 的扫描豁免；现在所有可读发布文本均执行路径扫描，专项 `6 passed`。
- **最终验证（2026-08-23，提交 `fc1cb4b`）**：clean-clone smoke 通过；source-only ZIP `36,934,885` bytes、1,020 个文件，manifest/哈希/路径与文本安全扫描均通过，health/config=200，离线 BM25 5 条，网络请求 0，LLM disabled。

### P2-03 参赛材料一致性审查

- [ ] **目标**：生成最终项目说明书、技术报告、实验表和限制说明。
- **验收**：所有数字可追溯到运行产物；内部指标与官方成绩严格分栏；明确已知阻塞、模型版本、数据许可、硬件、成本和复现命令；不把 Replay、synthetic fixture 或单查询 smoke 写成泛化结论。
- **依赖**：P0-03、P1/P2 实际验证结果。

## 外部阻塞记录

- `BLOCKED` 只用于同一外部条件连续三次检查仍无法推进时；第一次发现只记录在对应任务，不停止其他可执行任务。
- 允许的外部条件：真实 LLM Provider/GPU、开放全文许可、独立撤稿回执、官方 scorer、历史冻结输入或目标平台 wheelhouse。
- 不得通过读取或修改 `.env`、使用未授权 SSH、伪造运行产物、把 gold/qrels 放入在线路径或修改历史哈希来解除阻塞。
