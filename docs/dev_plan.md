# ScholarNavigator 开发主计划

本文件是后续开发的唯一主计划。状态以当前工作树、代码、测试和可读取的运行产物为准；旧报告只作为线索，不能替代验证。

执行循环：读取本计划 → 选择最高优先级且可执行的未完成项 → 修改代码/测试 → 运行针对性验证 → 自我审查 → 更新本计划。任何检索、排序、Query Evolution、Reranker、LLM 或全文改动都必须有可复现的成对实验；没有收益不得默认启用。gold/qrels 只能在检索完成后的离线 evaluator 中使用，不能进入在线查询、索引、Prompt 或 connector。

## 当前审计快照（2026-08-22）

- [x] 已确认仓库包含多源检索、SQLite BM25、Dense/Faiss、Qwen3 Reranker、查询规划/演化、RefChain、LLM feedback、质量信号、全文证据模型、FastAPI/Next.js 和竞赛评测脚本。
- [x] `npm run lint` 与 `npm run build` 当前通过。
- [ ] `PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q` 仍不通过；current-rules 缺失历史输入已通过 preflight 显式跳过后，当前首个失败为 `test_frozen_evidence_registry_gate`，原因是已修改的 `docs/dev_plan.md` 与旧证据注册表基线哈希不一致。该漂移必须通过新的人工审查/基线提案处理，不能直接改基线哈希。
- [x] 已增加显式历史证据 preflight：`scripts/audit_cluster_significance.py preflight` 与 `scripts/check_current_rules_regression.py preflight` 返回 `external_evidence_unavailable`，严格 `check` 仍失败；默认测试只对已显式接入 preflight 的门禁做结构化跳过，未接入的门禁仍会严格暴露阻塞。当前全量测试仍需继续清理其他非历史环境阻塞。
- [ ] 文档中声称的 `contest_full_dense_reranker_rrf_soft_v3`、P0 精确元数据/Faiss 运行产物不在当前工作树中；不得把相应数值视为可审计成绩，必须先改正文档并重新完成可读产物的成对实验。
- [ ] `technical_report.md`、`architecture.md`、`experiment-results.md` 存在状态描述不同步，必须在发布前统一。
- [ ] 不连接服务器、不读取 `.env` 或 SSH 凭据；需要真实 Provider/GPU/官方 scorer 的任务只记录外部依赖，不伪造完成。

## P0：干净环境可复现性与反馈闭环

### P0-01 证据依赖清单与测试分层

- [x] **目标**：枚举所有测试/脚本引用的运行产物，区分仓库内 fixture、可重建输入和外部历史证据。
- **验收**：自动检查输出 JSON/JSONL 的相对路径、存在性、SHA-256 和来源类型；缺失外部证据必须输出明确 `external_evidence_unavailable`，不得静默使用空数据。
- **依赖**：现有 benchmark manifest、pytest 收集结果。
- **完成说明（2026-08-22）**：新增 `src/scholar_agent/evaluation/dev_plan_audit.py` 与 `scripts/audit_dev_plan.py`，只读扫描 317 个 benchmark JSON manifest、2531 个路径引用；当前报告为 `1840 present`、`38 present_unhashed`、`131 missing_external`、`247 missing_tracked`、`275 hash_mismatch`，其中哈希漂移包含本轮真实代码/计划修改，因此仓库仍未达到全量证据闭环。专项 fixture 测试 `tests/test_dev_plan_audit.py` 为 `2 passed`，`compileall` 与 `git diff --check` 通过。

### P0-02 修复默认测试闭环（不弱化历史门禁）

- [ ] **目标**：让产品代码、单元测试和可重建评测在新 checkout 中可执行；历史证据门禁仍保持严格失败，不通过删测试、改断言或伪造历史产物解决。
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
- **增量验证（2026-08-22）**：用户说明部分实验位于外部服务器 `/mnt/highway1/wang/ScholarNavigator-main`；本轮不连接该服务器、不读取 SSH 凭据。服务器产物未纳入本地证据链，待用户自行导出脱敏的 `config.json`、输入/索引哈希、`metrics.json`、`results.jsonl`、`resource_ledger.json` 和完成标记后再审计。
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

### P0-03 发布包与文档状态一致

- [ ] **目标**：统一代码实际能力、评测结果、限制和发布包内容，删除过期“已完成/未实现”冲突描述。
- **验收**：从全新目录安装后可完成 smoke、前端构建、API 启动和离线评测；报告中的每个指标都能链接到实际产物，明确标注“内部指标/非官方成绩”。
- **依赖**：P0-02；不要求历史 `record160` 或官方 scorer 可用。
- **增量验证（2026-08-22）**：审阅并修正 `docs/contest/submission-checklist.md`、`submission-report-template.md`、`experiment-protocol.md` 中无法由当前 checkout 核验的 P0/Faiss、Dense/Reranker/RRF 完整运行数字；改为待重新运行/不可审计状态，并保留内部指标与官方成绩边界。`technical_report.md`、`architecture.md`、`experiment-results.md` 未发现同类具体 RunId 数字残留。P0-03 仍未完成，因尚未完成全新目录 smoke/API/frontend/release 验证。
- **增量验证（2026-08-22）**：构建 source-only release audit 包 `outputs/release-audit-20260822.zip`，共 995 个成员；检查确认无 `.env`、`outputs/`、`datasets/semantic/`、服务器 IP 或本地绝对路径，构建器标记 `internal_metric_scope=not_official_competition_scorer`。仍需全新目录安装、API health、前端 build 和离线检索 smoke，故 P0-03 不勾选完成。

## P1：最终效果（召回/F1 → 质量过滤 → 全文证据）

### P1-01 元数据质量与本地索引契约

- [ ] **目标**：确认正式语料按稳定 arXiv ID 精确关联，并覆盖 title、abstract、year、authors、venue、DOI 等排序所需字段；标题匹配旧语料只能作为 legacy。
- **验收**：构建器拒绝无稳定 ID 的记录；报告输入/输出计数、缺失字段、语料哈希和索引哈希；同一输入重复构建得到字节稳定结果。
- **依赖**：可读取的 PaSa/arXiv 元数据；不读取 AutoScholarQuery gold 生成语料。
- **当前证据（2026-08-22）**：新增 `scripts/audit_corpus_metadata.py` 与 `corpus_metadata_audit.py`。现有 `datasets/local_bm25/pasa_papers.jsonl` 为 569,432 条、arXiv ID 唯一，但 abstract/authors/year/venue/doi 完整度均为 0；`datasets/semantic/pasa_papers_with_abstracts.jsonl` 为 31,136 条、ID 唯一，title/abstract 完整度为 1.0，其余字段完整度均为 0。构建器已支持未来官方元数据中的 year/venue/doi，并新增对应报告字段；专项测试 `10 passed`。因此该任务尚未完成，当前排序仍受元数据缺口限制。
- **增量验证（2026-08-22）**：`local_hybrid._paper_from_semantic_row()` 已将 authors/year/venue/doi 接入 `Paper`/`PaperIdentifiers`，含 DOI 规范化与年份范围校验；新增元数据映射测试通过。现有语料仍缺字段，故不能宣称排序元数据质量已达标。
- **增量验证（2026-08-22）**：`local_hybrid` 读取语义语料时现在强制稳定 arXiv ID、规范化版本号并拒绝重复 ID；专项本地连接器/API/构建器测试 `26 passed`。这完成了索引入口契约的一部分，但真实语料的元数据补齐仍未完成。
- **增量验证（2026-08-22）**：语义索引 `metadata.json` 现在记录 title/abstract/authors/year/venue/doi 的 `field_completeness`，与语料 SHA-256、文档数和 ANN 指纹一起持久化；索引测试通过。当前真实语料仍以 `authors/year/venue/doi=0` 为主，P1-01 不勾选完成。

### P1-02 召回/F1 成对资格与完整评测

- [ ] **目标**：在同一查询顺序、数据、候选预算、模型和资源约束下比较 rules、Dense、Reranker、RRF/软 Judgement。
- **验收**：固定 200 条资格门禁通过后才允许 1000 条；报告 candidate recall、F1@5/10/20、Recall@20、MRR、延迟、调用数、失败率和显著性区间；任何无收益策略保持默认关闭。
- **依赖**：P1-01；GPU/模型或 Provider 不可用时只完成离线门禁，不伪造成绩。

### P1-03 低质量论文过滤（可解释、默认保守）

- [ ] **目标**：把元数据完整度、身份一致性、撤稿证据和重复风险与相关性分数分离，形成可解释诊断；只有独立来源明确 `flagged` 时才允许软影响排序。
- **验收**：未知风险不扣分、不硬过滤；Crossref/arXiv 回执带稳定身份、来源和哈希；quality policy 通过成对资格且有正向区间后才能考虑默认启用，否则保留负面结论。
- **依赖**：P1-02 的固定候选；外部来源不可用时保留 `unknown`。

### P1-04 全文证据展示与定位

- [ ] **目标**：选择性移植上游的受限 PDF/HTML 获取、段落选择和定位映射，不整体替换现有 SearchService。
- **验收**：只接受 HTTPS/allow-list/许可可核验来源；PDF/HTML 失败时保留摘要结果；证据含 URL、内容哈希、页码/段落或章节定位；QASPER 或等价公开数据有独立 Evidence F1 基线；全文默认不改变论文排序，除非成对实验证明收益。
- **依赖**：P1-01；锁定 `pypdf`/解析依赖；真实开放全文仅做受限验证。

## P2：受控智能增强与发布演示

### P2-01 LLM 搜索增强

- [ ] **目标**：保留当前 `llm_feedback` 默认关闭；补齐上游 controller 的可观察动作、预算、Schema、Record/Replay 和失败回退中真正缺失的部分。
- **验收**：每查询最多一次调用和一条反馈查询；原始查询保留；Provider/Schema/预算失败不破坏首轮；5 条 smoke、200 条资格和必要的 1000 条成对实验均无 fallback 且质量/成本门禁通过后，才允许进入演示主路径。
- **依赖**：P0 完成、P1-02 完成、可用且隔离的 Provider；不可用时标记 `BLOCKED`，不修改 `.env` 规避。

### P2-02 发布包、启动脚本与演示

- [ ] **目标**：形成评委可运行的 source-only 发布包、5 分钟视频脚本和一组不依赖 gold 的演示查询。
- **验收**：压缩包不含 `.env`、密钥、临时输出和超限模型缓存；全新目录完成安装、后端 health/config、前端 build、一次本地检索、结构化导出；演示明确展示查询理解、召回、排序依据、证据定位、成本和失败降级。
- **依赖**：P0-03、P1-02、P1-04；官方提交格式和大小限制以赛事页面为准。

### P2-03 参赛材料一致性审查

- [ ] **目标**：生成最终项目说明书、技术报告、实验表和限制说明。
- **验收**：所有数字可追溯到运行产物；内部指标与官方成绩严格分栏；明确已知阻塞、模型版本、数据许可、硬件、成本和复现命令；不把 Replay、synthetic fixture 或单查询 smoke 写成泛化结论。
- **依赖**：P0-03、P1/P2 实际验证结果。

## 外部阻塞记录

- `BLOCKED` 只用于同一外部条件连续三次检查仍无法推进时；第一次发现只记录在对应任务，不停止其他可执行任务。
- 允许的外部条件：真实 LLM Provider/GPU、开放全文许可、独立撤稿回执、官方 scorer、历史冻结输入或目标平台 wheelhouse。
- 不得通过读取或修改 `.env`、使用未授权 SSH、伪造运行产物、把 gold/qrels 放入在线路径或修改历史哈希来解除阻塞。
