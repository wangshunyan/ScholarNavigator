# 开发路线

本文件是基于当前 ScholarNavigator 审计结果制定的唯一开发路线。每次只实施优先级最高的一个未完成小任务；每个任务应可独立提交，遵循“目标 → 实现 → 验证 → 失败处理 → 完成条件”。实验结论必须由独立运行产物证明，内部指标不等同赛事官方 scorer。

## 当前审计基线

- P0-01 的离线反馈闭环已实现；P0-02 的脱敏快照与回放契约已实现并通过专项回归。
- `contest_qual200_dense_reranker_llm_feedback_v19` 仍是未完成的服务器资格运行；当前只保留为运行中诊断，不把未完成结果写入正式成绩。
- 当前真实 LLM 资格/完整运行、全文证据链、质量过滤独立证据、Linux 离线依赖和历史冻结证据仍未完成。
- 可离线完成：代码、fake provider、fixture、Record/Replay、解析器、质量 policy、审计逻辑和报告格式；需要真实 Provider/GPU：LLM smoke/qualification/full；需要外部数据或官方接口：开放全文、撤稿/出版元数据和赛事官方 scorer。

## P0：LLM 结果反馈驱动搜索闭环

### [x] P0-01 受限 LLM 首轮结果反馈查询演化

- **任务编号**：P0-01；**独立提交**：反馈策略、Schema、Prompt 与测试可单独提交。

- **目标能力**：在首轮检索、Judgement 与排序完成后，让 LLM 依据受控的候选摘要和规则覆盖缺口生成最多一条第二轮补充查询。
- **当前缺口**：`llm_semantic` 只在首轮前规划；`seed_expansion` 和 `coverage_gap` 仅为规则式查询演化，无法在不读取 gold/qrels 的条件下利用首轮结果形成 LLM 反馈闭环。
- **实现范围**：新增默认关闭的 `llm_feedback` 查询演化策略、版本化 Prompt、严格 JSON 输出、本地校验、预算/回退诊断、CLI/API 策略枚举和最小集成测试。
- **实现方案**：只向 LLM 发送原始查询、显式/规则约束、覆盖缺口和最多三条已排序候选的 title/abstract excerpt/matched terms；候选元数据作为不可信数据封装。固定 temperature=0，每查询最多一次 LLM 调用，最多一条反馈查询；必须保留原始查询。任何超时、预算耗尽、Schema 失败、注入式元数据或本地质量门拒绝均不启动第二轮检索。
- **验证方式**：离线 fake LLM 单元/服务测试验证调用次数、数据边界、原始查询保留、第二轮检索、预算与失败回退；运行相关测试和 lint 检查。
- **失败处理**：外部 Provider 不可用时仅记录 `llm_unconfigured` 或失败原因，不伪造实验成绩；保留默认规则路径，不启用该策略。
- **外部依赖**：可选 OpenAI-compatible Provider；实现与测试不依赖 `.env`、网络、gold/qrels 或真实模型。
- **完成条件**：代码、Prompt、Schema、测试和架构说明同步；相关测试通过；计划记录实现与验证结果。真实 200/1000 评测另列为 P3，不能因实现完成而宣称质量提升。
- **验收标准**：默认策略行为不变；原始查询保留；每条最多一次调用、最多一条反馈查询；gold/qrels 不进入在线输入；Provider/Schema/预算失败不破坏首轮结果。
- **自动化验证方式**：fake provider 单测、SearchService 集成测试、Prompt 边界测试、相关 lint/compile；真实效果不由本任务验收。
- **可用性与阻塞**：代码和测试可离线完成；真实 Provider 仅在 P3-01 验证，不能用 smoke 数字宣称正式提升。
- **实际验证（2026-08-21）**：`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/test_llm_feedback_evolution.py tests/test_query_evolution.py tests/test_query_evolution_coverage_gap.py tests/test_query_evolution_api_policy.py tests/test_llm_query_planning.py tests/test_llm_planning_snapshots.py tests/test_search_service.py`，`129 passed`。
- **真实评测状态**：实现与离线验证完成，不代表真实质量提升；P3-01 的 5 条 smoke 和 200 条资格仍依赖可用 Provider、隔离 GPU、P0/Faiss/BGE/reranker 一致性及新的 RunId。
- [x] 已完成

### [x] P0-02 反馈闭环快照与运行审计

- **任务编号**：P0-02；**独立提交**：反馈快照、审计字段与回放测试可单独提交。

- **目标能力**：为通过 P0-01 的 LLM 反馈请求建立可回放请求身份、输入边界和第二轮运行审计。
- **当前缺口**：现有 LLM 初始规划有独立快照机制，首轮结果反馈尚无专用审计/回放契约。
- **实现范围**：在 P0-01 稳定后扩展快照与 benchmark 元数据，不改变默认策略。
- **实现方案**：冻结脱敏请求 identity、Prompt hash、候选摘要哈希、调用与回退账本；复放时零网络/零 LLM。
- **验证方式**：record/replay 字节稳定测试、失败闭合测试与审计脚本。
- **失败处理**：缺失快照 fail closed，不能回退 live 或生成正式成绩。
- **外部依赖**：P0-01 已验证接口；真实 Provider 只用于后续受控 smoke。
- **完成条件**：快照、审计和恢复测试通过，P0-01 计划条目不被改写。
- **验收标准**：请求键由脱敏身份稳定生成；快照不含 Prompt/查询/候选正文；Replay 零网络、零 LLM；缺失或失败快照 fail closed；非 live 审计拒绝缺少快照键或 Replay 实际调用。
- **自动化验证方式**：Record/Replay 字节稳定、内容哈希篡改、缺失快照、无 provider 回放、审计门禁和 benchmark 回归测试。
- **可用性与阻塞**：实现完全离线可验证；真实 Provider 不属于本任务完成条件，真实质量仍由 P3-01 证明。
- **完成说明（2026-08-21）**：新增独立 `llm_feedback` 快照存储与运行时边界。每个请求只保存 Prompt 版本/哈希、Provider 身份、请求参数、原始查询/约束/覆盖缺口/候选包络的 SHA-256 身份，以及响应、调用和传输账本；不保存 Prompt、原始查询或候选输入正文。`replay` 零网络、零 LLM；缺失或失败快照 fail closed，并记录 `snapshot_missing` 或失败原因。基准运行结果与 metrics 增加 `llm_feedback_cost_report` 和聚合的 `llm_feedback_costs`，默认策略保持不变。
- **实际验证（2026-08-21）**：`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/test_llm_feedback_snapshots.py tests/test_llm_feedback_evolution.py tests/test_audit_contest_llm_run.py tests/test_benchmark_runner.py`，`50 passed`。真实 Provider 的 smoke/qualification 仍属于 P3-01，未因离线回放验证而声明质量提升。
- [x] 已完成

## P1：全文获取、解析与证据链

### [x] P1-01-A 离线纯文本段落证据定位契约

- **任务编号**：P1-01-A；**独立提交**：仅包含离线证据模型、稳定分块和 fixture 测试；不包含网络下载。

- **目标能力**：把已经获得且许可已核验的纯文本稳定切分为可定位、可校验的段落证据。
- **当前缺口**：现有系统只有标题/摘要 EvidenceItem，没有全文段落的来源、内容哈希和字符定位契约。
- **实现范围**：新增纯文本规范化、段落模型、来源许可状态、全文/段落 SHA-256 和字符区间；不下载网络内容、不改变默认搜索。
- **实现方案**：调用方先提供文本和许可回执；未核验许可或空文本 fail closed；段落边界固定为连续非空行块，保留规范化后的字符区间。
- **验收标准**：相同文本在 CRLF/LF 下生成相同内容哈希和证据 ID；每个区间能定位对应段落；未核验许可不产生证据；默认检索路径不受影响。
- **自动化验证方式**：`tests/test_full_text_evidence.py` fixture 测试覆盖稳定哈希、定位、许可拒绝和空文本失败；编译检查和 diff 检查。
- **失败处理**：异常只返回明确错误，后续集成层应保留摘要并标记证据缺失；本任务不猜测全文或许可。
- **外部依赖**：无，完全可离线完成；真实开放全文获取属于 P1-01-B。
- **完成条件**：模型、实现、测试和本记录提交；P1-01-B 仍保持未完成。
- **实际验证（2026-08-21）**：`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/test_full_text_evidence.py`，`3 passed`；`compileall` 和 `git diff --check` 通过。
- [x] 已完成

### [x] P1-01-B 许可来源获取与结果侧证据映射

- **任务编号**：P1-01-B；**独立提交**：网络获取/解析适配和结果映射必须与 P1-01-A 分开。
- **目标能力**：只从许可明确的开放来源获取全文，并把段落证据安全映射到候选结果。
- **当前缺口**：尚无受限下载器、HTML/PDF 解析适配、许可元数据核验和结果 API 映射。
- **实现范围**：允许列表来源、超时/大小限制、解析失败状态、段落证据引用；不抓取受限内容，不阻断摘要搜索。
- **实现方案**：调用方先提供可审计的许可回执；受限获取器仅接受 HTTPS 和显式 allow-list 主机，对响应状态、媒体类型、超时和字节数 fail closed。纯文本/HTML 解析进入 P1-01-A 的稳定段落证据模型；未安装 PDF 解析器时返回 `parser_unavailable`。只有成功产物才写入 `Paper.full_text_evidence`，API 复用现有不可信文本/URL净化边界输出证据链。
- **验收标准**：来源许可可追溯、下载和解析失败闭合、结果保留摘要降级、证据 URL/哈希/段落 ID 可审计。
- **自动化验证方式**：离线 HTML/PDF fixture、模拟 HTTP 状态、许可拒绝、大小/超时和 API 映射测试；真实来源只做独立资格验证。
- **失败处理**：外部来源不可用时不伪造全文，保留 `evidence_unavailable`；未有真实开放来源回执不得完成。
- **外部依赖**：开放许可全文、解析库和可能的来源 API；需要真实网络授权时单独记录。
- **完成条件**：fixture 与至少一个真实、许可可核验来源均可追溯，默认搜索回归通过。
- **离线实现与验证（2026-08-21）**：已完成 allow-list、许可前置、HTTP 状态/媒体类型/大小限制、HTML 可见文本解析、PDF `parser_unavailable` 和 API 结果映射；`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/test_full_text_evidence.py tests/test_api_mapper.py tests/test_dedup.py tests/test_top20_delivery_fidelity.py tests/test_structured_output_provenance_gate.py`，`53 passed`，随后 `compileall` 与 `git diff --check` 通过。
- **真实来源验证（2026-08-21）**：对 PMC Open Access 记录执行只读核验：来源返回许可证属性和下载记录；在 allow-list、HTTPS、2 MB 与 10 秒限制下解析公开 HTML 全文成功，生成段落证据和内容 SHA-256。首次临时 `parse_failed` 被正常 fail closed，后续独立重试成功；未保存正文、未写入检索语料或正式评测结果。
- **完成说明**：P1-01-B 只覆盖许可前置的受限获取、文本/HTML 解析、失败闭合和 API 映射。PDF 实际文字提取单列 P1-01-C，当前环境未安装 PDF 解析依赖，不能以 `parser_unavailable` 伪称完成。
- [x] 已完成

### [x] P1-01-C PDF 全文解析适配

- **任务编号**：P1-01-C；**独立提交**：仅增加可锁定的 PDF 解析依赖、受限解析适配和 fixture，不改检索或排序默认路径。
- **目标能力**：对已通过 P1-01-B 许可/来源校验的 PDF 产生相同的稳定段落证据。
- **当前缺口**：当前环境没有 `pypdf`、`PyPDF2` 或系统 `pdftotext`；PDF 响应正确返回 `parser_unavailable`，尚未解析实际 PDF 正文。
- **实现范围**：锁定一个 Python PDF 文本解析器，加入页数/字节/异常边界、PDF fixture、稳定哈希及 API 映射回归；不进行 OCR、不抓取受限 PDF。
- **实现方案**：仅当 P1-01-B 已完成许可和 URL allow-list 后，使用锁定版本解析内存字节；无文本层、加密、超页数或库错误均 fail closed，保留摘要降级。
- **验收标准**：fixture PDF 在重复解析时得到相同段落证据；加密/损坏/图片型 PDF 不产生虚构文本；真实、许可可核验 PDF 通过独立只读验证；默认搜索不受影响。
- **自动化验证方式**：离线 PDF fixture、损坏/加密 fixture、依赖锁检查、API 映射和 P1 回归测试。
- **失败处理**：依赖或真实许可 PDF 不可用时保持 `parser_unavailable` / `parse_failed`，不安装未锁定依赖、不标记完成。
- **外部依赖**：锁定并可安装的 PDF 解析库，以及真实开放许可 PDF。
- **完成条件**：代码、测试、锁文件、真实来源验证和本记录齐全。
- **完成说明（2026-08-21）**：新增 `pypdf==6.16.1` 作为精确运行时依赖，并以受限内存字节解析 PDF；加密、损坏、无文本层、超页数和库错误均返回失败状态而非生成文本。现有 Linux 离线 wheelhouse 不因本次依赖新增而宣称已验证，仍由 P4-01 处理。
- **自动化验证（2026-08-21）**：`tests/test_full_text_evidence.py` 覆盖文本 PDF、HTML、许可/allow-list、大小限制、未知媒体类型、损坏和加密 PDF，`7 passed`。
- **真实来源验证（2026-08-21）**：对已核验许可证的公开 PDF 执行只读受限解析，获得 `succeeded`、`application/pdf`、25 个段落和内容 SHA-256；未保存正文、未写入检索语料或正式评测结果。
- [x] 已完成

## P2：论文质量过滤

### [x] P2-01-A 可解释离线质量信号

- **任务编号**：P2-01-A；**独立提交**：只新增独立质量诊断模型和 fixture，不改变当前排序或过滤。

- **目标能力**：将年份、来源完整性、撤稿/重复风险和元数据质量作为可解释的软质量信号。
- **目标能力**：把论文元数据完整性、来源交叉确认、稳定标识符、全文证据和已知风险状态拆成独立、可审计的软质量诊断。
- **当前缺口**：现有 reranker 已混合使用引用/来源/元数据启发式，但没有单独的质量报告，无法区分“低相关”与“质量未知”。
- **实现范围**：新增默认不影响排序的 quality report 和稳定软分数；不读取 gold/qrels，不新增网络请求，不把未知字段视为负面或不相关。
- **实现方案**：为每个可验证元数据维度记录 `present`/`missing`/`unknown`，对本地可见的完整性和多来源交叉确认计算有限软分数；撤稿和重复风险只在没有外部回执时标记 `unknown`。
- **验收标准**：相关性分不被改写；缺失撤稿/出版数据不扣分或过滤；相同 Paper 得到确定性报告；风险/缺口清晰可解释。
- **自动化验证方式**：fixture 覆盖完整元数据、字段缺失、多来源、全文证据、零/负引用和风险未知；运行质量模型与 reranker 回归测试。
- **失败处理**：输入异常返回明确失败，不猜测撤稿或质量；默认搜索与排序继续采用现有路径。
- **外部依赖**：无，完全离线；真实撤稿/出版数据接入属于 P2-01-B。
- **完成条件**：独立模型、测试、验证记录和本条 checkbox 更新；质量效果不因实现完成而宣称提升。
- **完成说明（2026-08-21）**：新增 `PaperQualityReport` 与 `assess_paper_quality`，输出 metadata completeness、source corroboration、stable identifier coverage、licensed full-text evidence 和两个明确 `unknown` 的外部风险状态。报告只基于当前 `Paper` 事实，未修改 Judgement、reranker、检索或默认 API 排序。
- **实际验证（2026-08-21）**：`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/test_paper_quality.py tests/test_full_text_evidence.py tests/test_dedup.py`，`26 passed`。没有运行质量指标实验，因此不声明 F1、Recall 或官方 scorer 改善。
- [x] 已完成

### [ ] P2-01-B 默认关闭的质量软评分 policy

- **任务编号**：P2-01-B；**独立提交**：只在 P2-01-A 已完成后，以新 policy 接入 reranker 与 qualification runner。
- **目标能力**：在不改变 `current_rules` 的前提下，将 P2-01-A 的质量报告作为可审计软排序信号。
- **当前缺口**：质量 policy 已进入独立 ranking policy，但尚无独立外部风险回执带来的可审计质量收益；既有 200 条资格运行没有产生 F1/Recall 提升。
- **实现范围**：新增默认关闭 policy、配置哈希、排名诊断和 200 条资格方案；不做硬过滤，不覆盖基线。
- **实现方案**：限定低权重质量贡献；撤稿/重复未知不产生惩罚；只有已验证风险回执才影响信号，并保留原始相关性/检索分数。
- **验收标准**：`current_rules` 字节级/回归行为不变；新 policy 通过 fixture 和资格门禁才允许完整运行；未证明改善时不进入正式成绩。
- **自动化验证方式**：policy 选择、序列化、排序稳定性、缺失风险、配对 bootstrap 和资源账本测试。
- **失败处理**：质量来源不可用时回退为 unknown，不用默认值伪造风险；资格失败只保留诊断。
- **外部依赖**：真实撤稿/出版回执、200 条资格集和可用实验环境。
- **完成条件**：代码、测试和真实资格证据齐全；未通过质量门禁时保持未完成。
- **实现说明（2026-08-21）**：新增默认关闭的 `quality_soft_v1` 排名 policy。它只把 P2-01-A 的本地质量报告转为最大 `0.02` 的低权重排序贡献，仅在既有 Judgement 类别内比较；不硬过滤、不改写 `final_score`、检索分数或相关性分数。撤稿与重复风险仍是 `unknown`，贡献固定为零惩罚。每条输出记录 policy、质量分、贡献、配置 SHA-256、原排名及换序原因；runner 的 `config.json` 同步记录固定配置及哈希。`current_rules` 仍为默认且未修改。
- **自动化验证（2026-08-21）**：`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/test_paper_quality.py tests/test_reranker.py tests/test_rrf_fusion.py tests/test_ranking_policy_api.py tests/test_benchmark_runner.py tests/test_full_text_evidence.py tests/test_dedup.py tests/test_api_mapper.py`，`87 passed`；随后 `git diff --check` 与 `compileall` 通过。补齐 Linux `scripts/run_contest_benchmark.sh` 的 `dense_reranker_quality` 入口后，相关 Python 回归为 `49 passed`，shell `bash -n` 语法检查通过。
- **真实资格状态**：新的 200 条资格集已执行并完成；虽然运行、reranker 审计和资源账本均可审计，但没有 F1/Recall 提升证据，因此本任务保持未完成，不能进入正式成绩，也不运行质量策略全量实验。
- **失败处理记录（2026-08-21）**：首次新 RunId `contest_qual200_dense_reranker_quality_v1` 在零结果时因 Linux wrapper 漏传语义语料配置而退出，错误为 `local_hybrid_config_required`；该 RunId 保留为启动诊断，不计入资格结果。已修复 wrapper 条件并将下一次运行改用全新 RunId。
- **资格运行状态（2026-08-21）**：`contest_qual200_dense_reranker_quality_v2` 已完成 200/200，生成完整 metrics、stage_metrics、error_analysis、resource_ledger、summary 和 generation `RUN_COMPLETED`；运行使用 `quality_soft_v1`、GPU1、单 worker。尚未把它标记为通过，需用配对的 `contest_qual200_reranker_v4_gpu1` 基线执行资格门禁，确认 F1/Recall 的固定 bootstrap、零失败、reranker 审计和资源账本。
- **资格门禁结果（2026-08-21）**：使用服务器主目录中已完成的 `contest_qual200_reranker_v4_gpu1` 作基线执行 `scripts/check_contest_qualification.py`。200 条、零失败、reranker 审计和资源账本均通过；但 F1@20 与 Recall@20 的平均成对差值均为 `0.0`，固定 5000 次 bootstrap 的 95% 区间均为 `[0.0, 0.0]`，因此 `eligible_for_full_1000=false`，不运行质量策略全量实验。离线分析显示 v2 没有任何 Top-20 顺序变化；该策略保留为诊断实现，不写入正式成绩。
- **失败处理补充（2026-08-21）**：对已完成 v2 的候选顺序做 gold-blind 阈值分析：在当前质量信号分布和固定 `quality_weight <= 0.02` 约束下没有可发生的换序；第一个潜在换序约需 `0.024` 权重。该阈值来自结果后的诊断，不能用于事后调参，因此不提高权重、不重跑同一配置、不宣称质量收益。若未来获得独立、可审计的撤稿/重复风险回执，应以新 policy 版本和新 RunId 重新预注册资格。
- **诊断修正（2026-08-21）**：发现 API 映射层丢弃质量策略诊断字段，已补齐 `quality_policy`、`quality_score`、`quality_contribution`、配置哈希和换序原因的公开结果映射；相关回归共 `76 passed`。这不改变已完成 v2 的失败门禁结论，也不构成质量提升证据。
- **离线证据契约（2026-08-21）**：新增 `VerifiedQualityEvidence`。调用方必须提供精确稳定论文标识、来源、来源记录 ID 和 `clear`/`flagged` 状态；模块不联网、不保存外部正文，记录 ID 只以 SHA-256 进入诊断。无匹配回执继续保持 `unknown` 且不扣分；同一风险信号出现冲突回执时 fail closed。默认 `assess_paper_quality(paper)` 的分数和范围保持不变。
- **自动化验证补充（2026-08-21）**：质量、reranker、API 映射、资格门禁和 benchmark 相关回归共 `79 passed`，编译检查和 `git diff --check` 通过；同步前端 API 类型后 `npm run lint` 和 `npm run build` 均通过。
- **证据传递链修正（2026-08-21）**：`SearchService.run_search`、模块级 `run_search`、PRF 预排序以及初始/语义扩展/查询演化/引用链重排入口现在均接收并转发可选 `verified_quality_evidence`。默认空集合保持旧行为；匹配回执才影响 `quality_soft_v1` 诊断，未匹配回执保持 `unknown`，冲突回执从完整排序入口 fail closed。未修改当前 Prompt、`current_rules` 或既有实验产物。
- **自动化验证（2026-08-21）**：新增排序策略与完整 `SearchService.run_search` 入口回归，覆盖无证据兼容、匹配/未匹配回执和冲突闭合；`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/test_paper_quality.py tests/test_rrf_fusion.py tests/test_search_service.py tests/test_benchmark_runner.py`，`92 passed`；编译检查通过。
- **Registry 集成修正（2026-08-21）**：实现枚举的 26 个策略中此前有两个未登记项，已将 `query_evolution_llm_feedback` 与 `quality_soft_v1` 加入 manifest；两者均保持默认关闭、`evidence_unavailable`，分别保留 fallback/无质量提升结论，不生成正式成绩。按离线生成器刷新 registry、matrix、summary 及 manifest 三个基线哈希；`check_evidence_registry.py check` 返回 `passed=true`、`drift_count=0`，registry 专项 `9 passed`，网络/LLM/benchmark 请求均为 `0`。
- **回归失败记录（2026-08-21）**：P2/registry 组合回归首次为 `111 passed, 1 failed`，唯一失败是已有 Windows spawn 隔离超时测试在整组运行时偶发把第二个正常任务判为 timeout；同一测试单独重跑为 `1 passed`。未删除、标记 skip 或放宽该测试；该运行波动仍保留为全仓验证风险。
- **隔离调度修正（2026-08-21）**：将隔离子进程的最小启动/回传窗口从 `1.2s` 调整为 `1.4s`，将有界清理窗口从 `0.5s` 调整为 `0.65s`，仍满足原测试的总延迟上限；不改变检索结果、质量 policy 或 benchmark 配置。完整 `tests/test_search_service.py` 为 `50 passed`，隔离超时测试连续 3 次通过；P2/registry 回归为 `62 passed`，registry gate `drift_count=0`。
- **回执 Ledger（2026-08-21）**：新增只读 `paper-quality-evidence-ledger-v1` JSONL loader，严格要求规范化稳定论文标识、单行对象、无重复 JSON key、无非有限数和每个 `(paper_identifier, signal_name)` 唯一记录。输出文件 SHA-256 与顺序无关的语义 SHA-256，载入后的回执可直接接入现有质量评估；不联网、不保存来源记录正文，不改变默认排序。该 loader 只准备未来外部回执接入，当前没有独立回执或质量提升证据，P2-01-B 仍未完成。
- **离线运行接入（2026-08-21）**：benchmark runner 新增 `--quality-evidence-ledger`，且只在显式 `quality_soft_v1` policy 下接受。运行配置仅记录 schema、文件/语义 SHA-256 和记录数，不保存账本路径或来源记录；每条查询把已验证回执传入完整检索链。缺少 policy 或 policy 不匹配时 fail closed。`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/test_paper_quality.py tests/test_benchmark_runner.py` 为 `40 passed`；P2 组合回归（quality、RRF、SearchService、benchmark、qualification、registry）为 `119 passed`。这只完成外部回执的离线可复现实验接线，仍缺独立回执和通过资格门禁的质量提升，任务保持未完成。
- **全仓验证状态（2026-08-21）**：按 `AGENTS.md` 执行 `PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q` 得到 `2097 passed, 230 failed, 58 errors`。失败主要是既有 evidence registry 基线未登记 `quality_soft_v1`、历史冻结输入/提交缺失和发布环境门禁漂移；本次 P2 相关专项保持全绿。前端 `npm run lint`、`npm run build`、`compileall` 和 `git diff --check` 通过。未跳过、删除或弱化全仓测试。
- **外部阻塞**：当前没有独立、可审计的撤稿/重复风险来源回执，因此未把该证据契约接入正式 200 条运行，也未宣称质量收益；获得独立回执后必须使用新 policy 版本和新 RunId 重跑资格门禁。
- [ ] 实现与离线验证完成；真实资格已执行但门禁未通过

### [x] P2-01-B1 Crossref 明确撤稿回执采集

- **任务编号**：P2-01-B1；**独立提交**：仅包含可选 Crossref 适配、严格 ledger 输出和离线 fixture，不改变默认搜索或排序。
- **目标能力**：将 Crossref 对 DOI 的明确“被撤稿”关系转换为可由 `quality_soft_v1` 消费的 `retraction_status=flagged` 回执。
- **当前缺口**：质量策略只能读取调用方提供的严格 ledger，缺少一个不依赖密钥、不会把“未发现”误写为安全状态的独立来源适配。
- **实现范围**：新增只接受规范 `doi:` 标识的 Crossref 收集器和显式 CLI；只识别 `is-retracted-by` 或类型为 `retraction` 的更新关系，输出 compact report 与有证据时的 JSONL ledger；不抓取全文、不写入检索语料、不创建 `clear` 回执。
- **实现方案**：固定 HTTPS Crossref works 地址、JSON 响应、1 MiB 上限和 1--30 秒超时；每项网络/HTTP/格式失败均只记为 `unknown`，不产生 ledger 行。报告只保存输入文件 SHA-256、数量和终态计数，绝不保存响应正文；已有输出默认 fail closed，避免覆盖旧回执。
- **验收标准**：明确撤稿关系仅产生一个 `flagged` 回执；关系缺失、404、网络故障与非法响应都不产生质量分；输入必须是规范 DOI，默认检索行为不变。
- **自动化验证方式**：离线 mock 覆盖关系、更新类型、404、网络故障、非 JSON、规范标识、空 ledger 拒绝及 compact 输出；compile 与 diff 检查。
- **失败处理**：没有明确关系或来源不可达时只留下 `no_explicit_evidence` / 终态计数，不运行质量资格实验，不用默认值伪造 `clear`。
- **外部依赖**：实际运行需要 Crossref 可访问及由合法 P0 语料导出的规范 DOI 列表；实现与 fixture 完全离线，不读取 `.env` 或 API Key。
- **完成条件**：收集器、CLI、离线测试和本记录同步；P2-01-B 的真实独立回执覆盖与质量资格提升仍单独要求。
- **完成说明（2026-08-21）**：新增 `src/scholar_agent/core/quality_evidence_sources.py` 与 `scripts/collect_crossref_retraction_evidence.py`。它只在来源明确表示论文被撤稿时生成 `flagged` 回执；不因无关系生成 `clear`。CLI 需要用户显式提供 DOI 文件并写入新路径，未发现证据时只产生报告并以非零状态退出。
- **实际验证（2026-08-21）**：`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/test_quality_evidence_sources.py tests/test_paper_quality.py tests/test_benchmark_runner.py`，`50 passed`；`compileall`、CLI help 与 `git diff --check` 通过。未调用 Crossref，不产生真实 ledger 或质量指标。
- [x] 已完成

### [x] P2-01-B2 P0 arXiv 标识的受限 DOI 解析与风险探测

- **任务编号**：P2-01-B2；**独立提交**：仅扩展 B1 的外部回执输入适配，不修改 P0 语料、索引、默认排序或任何实验结果。
- **目标能力**：对 P0 语料已有的精确 `arxiv:` 标识，从公开 arXiv 元数据提取明确 DOI 后再查询 Crossref，并把任何风险回执绑定回原始 `arxiv:` 身份。
- **当前缺口**：正式 P0 语料只保留 arXiv ID；直接以 `10.48550/arXiv.*` 探测 Crossref 无法覆盖已出版 DOI，且原始本机 metadata CSV 无可用稳定 arXiv ID，不能离线关联。
- **实现范围**：新增最多 20 条的 exact-ID arXiv Atom 解析、1 MiB 响应上限、严格 media type/ID 规范化、可选 Crossref 查询与 CLI `--identifier-kind arxiv`；不使用标题匹配、不读取 gold/qrels、不保存响应正文、不产生 `clear`。
- **实现方案**：arXiv 只接受请求集合中的精确 ID，返回缺 DOI、未返回、HTTP/网络/格式错误均保持 `unknown`；只有 arXiv 明确给出 DOI 才查 Crossref。明确撤稿关系生成的 ledger 行始终使用原 `arxiv:` ID，避免将检索语料或身份语义改成 DOI。
- **验收标准**：最多一个 arXiv metadata 请求且最多 20 个规范 ID；明确 DOI 风险仅映射到同一 arXiv ID；缺 DOI、来源异常和无关系不生成 ledger；CLI 报告标记 `arxiv_then_crossref` 来源链。
- **自动化验证方式**：mock Atom + Crossref fixture 覆盖 DOI 解析、风险映射、缺 DOI、网络失败、格式/输入/批量拒绝和 CLI provenance；P2 回归、compile 与 diff 检查。
- **失败处理**：无 DOI、无明确风险或来源不可用只记录聚合终态；不改变质量分、不开启 `quality_soft_v1`、不运行质量资格实验。
- **外部依赖**：真实探测需要公开 arXiv 与 Crossref 网络；实现和 fixture 离线。输入必须来自 P0 非评测语料，不接受 `AutoScholarQuery`/gold/qrels 路径。
- **完成条件**：exact-ID 解析、CLI、测试与真实小规模来源诊断均有记录；P2-01-B 的独立风险覆盖和质量提升仍需新资格运行证明。
- **完成说明（2026-08-21）**：新增 `collect_arxiv_crossref_retraction_evidence`，受限为单次最多 20 个规范 arXiv ID。解析得到的 DOI 只作为 Crossref lookup key；正式 ledger 仍以 `arxiv:` ID 绑定。CLI 现在要求显式 `--identifier-kind doi|arxiv`，报告不保存任何远端正文。
- **实际来源诊断（2026-08-21）**：从 P0 本地语料按 arXiv ID 顺序选取 5 条非评测论文进行两次一致的只读探测。每次均为 `arxiv:resolved=2`、`arxiv:no_doi=3`、`crossref:no_explicit_retraction_relation=2`、`flagged_evidence_count=0`；因此未生成 ledger、未改变质量分，也没有使用 gold/qrels、标题关联或正式指标。
- **实际验证（2026-08-21）**：`PYTHONPATH=src .venv\\Scripts\\python.exe -m pytest -q tests/test_quality_evidence_sources.py tests/test_paper_quality.py tests/test_benchmark_runner.py`，`58 passed`；`compileall`、CLI help 和 `git diff --check` 通过。
- [x] 已完成

## P3：真实评测与指标闭环

### [ ] P3-01 P0-01 的离线资格与真实运行门禁

- **任务编号**：P3-01；**独立提交**：资格 runner、审计和报告生成可先离线提交，真实运行产物另行提交。

- **目标能力**：在 P0-01 完成后，以不泄漏 gold/qrels 的运行链路评估质量、成本和稳定性。
- **当前缺口**：LLM 反馈策略尚无 5 条 smoke、200 条资格或配对 bootstrap 证据。
- **实现范围**：新 RunId 的 smoke、qualification、审计和报告；不覆盖既有运行。
- **实现方案**：固定数据/索引/查询顺序，gold/qrels 仅在检索后离线评价；比较 F1@20、Recall@20、MRR、P50/P95、Token、调用、失败和 fallback。
- **验证方式**：零失败/零 fallback、调用账本、资源账本、配对 bootstrap 和完整产物检查。
- **失败处理**：任一门禁失败仅保留诊断，不启动完整 1000 条。
- **外部依赖**：可用 Provider、GPU、P0/Faiss/BGE/reranker 与隔离资源。
- **完成条件**：只有满足预注册门禁的完整运行可进入正式内部报告。
- **验收标准**：5 条 smoke、200 条 qualification 和必要的 1000 条 full 使用新 RunId；零失败/零 fallback、调用账本完整、配对 bootstrap 支持提升、资源与哈希一致后才可写正式内部指标；内部指标标注非赛事官方 scorer。
- **自动化验证方式**：fake provider/Record-Replay runner、审计脚本、资源账本校验、配对 bootstrap 和完整产物清单测试；真实运行需服务器 GPU 与 Provider 回执。
- **可用性与阻塞**：runner、审计和统计可离线完成；真实 LLM/API 授权、GPU、固定语料/索引和赛事官方 scorer 是外部依赖，任一缺失只记录阻塞。
- [ ] 未开始

## P4：文档、依赖和工程一致性

### [ ] P4-01 发布一致性与阻塞透明化

- **任务编号**：P4-01；**独立提交**：文档、依赖锁、发布包校验和阻塞记录分别保持可回滚。

- **目标能力**：保持代码、依赖锁、实验文档、发布包和历史证据状态一致。
- **当前缺口**：Linux 离线 wheelhouse 与 `record160` 历史冻结输入仍阻塞最终 release tag。
- **实现范围**：更新真实状态、运行 registry/发布包门禁，不删除、跳过或弱化测试。
- **实现方案**：复用现有 evidence registry、锁验证和允许列表发布包工具。
- **验证方式**：专项测试、registry check、发布包禁止路径检查和完整测试报告。
- **失败处理**：保留明确阻塞记录，只推送普通审计提交。
- **外部依赖**：Linux Python 3.12 wheelhouse、历史 commits 与冻结输入。
- **完成条件**：所有门禁有真实证据后才允许最终 tag。
- **验收标准**：代码、文档、锁文件、实验清单和允许列表发布包一致；禁止路径与密钥排除校验通过；Linux/Python 3.12 离线安装和历史冻结证据有真实回执；未清零时不创建最终 tag。
- **自动化验证方式**：registry gate、lock/wheelhouse check、发布包内容扫描、后端全测、前端 lint/build 和 Git 分支/tag 一致性检查。
- **可用性与阻塞**：文档、扫描和本机校验可离线完成；Linux wheelhouse、历史输入和官方发布要求需要外部环境/证据，不能用占位文件替代。
- [ ] 未开始
