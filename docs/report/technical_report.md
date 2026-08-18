# ScholarNavigator 技术报告

## 报告口径

本报告只记录当前代码中可以定位、可以测试的状态。赛题要求见 [赛题要求摘要](../contest/requirements.md)，系统结构见 [当前架构](../architecture.md)，评测口径见 [评测说明](../evaluation.md)。当前实验均为固定小样本工程验证，不作为正式性能结论。

## 系统定位

ScholarNavigator 面向复杂学术查询，将自然语言问题转换为多子查询检索任务，合并多来源论文元数据，执行相关性判断、重排和证据约束归纳，并通过前后端分离界面展示结果与运行诊断。

系统采用真实检索单路径：产品前端调用异步 Real Search 生命周期接口，不使用产品级 fake 结果兜底。fake 数据只存在于离线评测与测试。

## 已实现并测试

### 检索与编排

- `SearchService` 串联查询理解、多子查询检索、跨源去重、相关性判断、重排、可选查询演化、可选单层 RefChain 和规则证据归纳。
- OpenAlex、arXiv、Semantic Scholar、PubMed connector 均已实现字段转换、超时或限速处理、错误诊断，并有隔离外网的测试。
- 去重保留 DOI、arXiv、OpenAlex、Semantic Scholar、PubMed 等标识符，并支持 title+year 后备键。
- 单个 connector 或子查询失败时保留其他来源结果，同时输出来源统计和 warning。
- 逻辑子查询在 connector 前生成安全原查询与可选核心补充查询；信息保留不足时回退，同一 run 的真正等价调用只执行一次，全部逻辑来源和用途保留在阶段 provenance 中。
- 初始查询规划可选择旧规则及版本化实验策略；`current_plus_disjunctive` 和 `facet_union` 都完整保留旧查询，基线检索结束后只用剩余预算追加至多一条补充查询，并把组合模式、策略、版本和 provenance 返回 API。
- arXiv adapter 将 `any` 查询安全转换为有界 OR 表达式，转义短语、括号和特殊字符；显式 must-have 位于 OR 组外，其他来源使用确定性安全回退。规划器不包含来源语法。
- 可选 `llm_semantic` 使用独立 Markdown Prompt 和严格 JSON Schema，在保留原查询的前提下最多生成两条语义补充查询；本地校验、稳定回退、独立快照和动态依赖均已通过隔离外网测试。
- Semantic Scholar 429、arXiv 请求间隔、OpenAlex 非法查询、瞬态失败重试与 run 内连续失败熔断均有隔离外网的行为测试。

### LLM 边界

- 后端支持 OpenAI-compatible provider，默认关闭。
- 查询理解可使用 LLM JSON 增强；响应必须经过 Schema 校验和来源白名单过滤，失败时回到规则计划。
- 相关性判断可分批调用 LLM，但只提供候选论文元数据；失败批次回到规则判断。
- LLM 调用次数及 provider 返回的 prompt、completion、total token usage 可进入结果统计。
- LLM 语义规划快照键包含模型、Prompt hash、规则分面和显式约束；快照不保存密钥、gold、候选或完整 Prompt，replay 缺键时不回退 live。

### API 与前端

- FastAPI 提供健康检查、运行配置、任务创建、状态、结果、SSE 和取消接口。
- 后台线程执行任务，run store 对终态任务执行 TTL 与数量清理。
- 前端支持来源选择、运行参数、状态轮询、SSE 事件、取消、结果卡片、证据归纳、引用图和本地 JSON/Markdown 导出。
- API mapper、Pydantic Schema 与前端 TypeScript 类型有对应测试或构建检查。

### 评测与工程检查

- fake fixture 离线 evaluator 可比较 baseline、仅查询演化、仅 RefChain 和两者同时启用四组配置。
- AutoScholarQuery Adapter 可确定性加载 1000 条 test 查询和 2403 条 arXiv gold；只读检查报告确认没有空 gold、无效案例或重复 qid。
- Benchmark Runner 复用真实 SearchService、统一结果选择、全标识符匹配、F1/Precision/Recall/MRR/nDCG 与 success-only/end-to-end 聚合，并支持原子输出和 resume。
- 可选阶段诊断追踪初始检索、去重、Judgement、Reranking、查询演化、RefChain 和最终返回，输出 gold 丢失原因、来源贡献和可测试的瓶颈标签；诊断不反向影响搜索。
- Query Evolution 与 RefChain 诊断记录逐 seed、逐扩展动作、唯一新增候选、事后 gold、分类分布、Judgement/Top-K 丢失、API 与阶段耗时，并用确定性规则给出小样本结论。
- Benchmark Runner 可在适配后检索请求与 RefChain seed 请求边界记录规范化公开响应，并按稳定 SHA-256 键离线回放；动态键通过绝对离线规划和有界串行采集迭代发现，快照原子写入并校验 Schema、内容哈希和 manifest 配置。Replay 缺键时禁止网络回退，执行成本与快照记录的 live 成本分别报告。
- 规则 Judgement 的参数已集中为严格 Schema 和稳定哈希；候选级特征向量记录分面与字段命中、约束结果、元数据完整度、可加和分数组件及分类原因。硬约束保护与可校准软参数分离，并有防 gold 泄漏和确定性测试。
- 固定保留集对比器锁定 AutoScholarQuery offset 20–49，校验两组候选指纹、零网络回放和不可变配置，输出阶段 gold 丢失、查询侧切片及固定随机种子的配对 bootstrap。
- 当前分支通过后端 pytest、前端 lint 和前端生产构建。

上述“测试”指单元测试、集成测试、mock connector 测试和构建检查，不代表外部检索质量或比赛指标已经验证。

## 已实现但未正式 benchmark 验证

| 能力 | 已有实现 | 尚缺证据 |
| --- | --- | --- |
| 多源检索 | 四个真实 connector、聚合与去重 | 未在完整公开或官方数据上验证召回率 |
| 查询演化 | 按初始结果覆盖缺口生成受控补充查询并执行质量门 | 固定小样本未新增 gold，默认关闭 |
| 初始查询规划 | `current_rules`、版本化实验策略、逐查询诊断和快照隔离 | 单 OR 与独立分面均未净增 gold，规则式规划已冻结，默认保持旧规则 |
| LLM 语义查询规划 | `llm_semantic` 的受限输入、校验、回退、成本诊断和独立 Record/Replay | 当前 provider disabled，尚未运行固定开发集与独立验证集的 LLM 对比 |
| 规则相关性判断 | `current_rules` 与冻结的 `calibrated_rules_v1`、版本化配置、候选诊断和离线网格校准 | 小样本只达到无回归，未证明 F1 或 gold false negative 改善，默认仍为旧策略 |
| RefChain | 通过 OpenAlex 做单层引用扩展 | 未量化新增召回与噪声、延迟的权衡 |
| LLM 查询理解与判断 | 可选 JSON 调用、校验和规则回退 | 未完成模型版本对比、消融和成本收益评测 |
| 规则重排 | 综合相关性、时效性、权威性和元数据等信号 | 未完成排序指标与阈值校准 |
| 证据归纳 | 从判断证据行生成带 citation key 的结论 | 未进行人工事实性与引用覆盖评审 |
| 运行效率 | 记录来源调用、缓存命中、LLM 用量和延迟 | 未按比赛口径形成可复现效率报告 |

AutoScholarQuery 原始顺序前 5 条已用 balanced、单一 arXiv 源完成真实 smoke：成功率 1.0，端到端 F1@5/10/20 为 0.0444/0.0286/0.0357，平均 API 请求 2.4、平均 Token 0、平均延迟 0.78 秒。样本很小且只使用单一来源，不能替代完整 Benchmark 或比赛成绩。

固定前 10 条的阶段诊断完成 arXiv-only 和三源 baseline。三源配置把初始候选 Recall 从 0.150 提高到 0.170、F1@20 从 0.0179 提高到 0.0259，但平均 API 请求由 2.7 增至 8.2、平均延迟由 4.01 秒增至 29.78 秒，并出现 0.402 的来源错误率。规则标签指向检索召回、Judgement false negative 和来源可靠性；没有发现已保留 gold 被排到 Top 20 外的 Reranking 瓶颈。其余三组因持续 429/超时安全中止，不能声明 Query Evolution 或 RefChain 的收益。

同一固定前 10 条上的 A2/B2 只改变来源查询适配与可靠性策略。A2 相对 A 的平均 API 由 2.7 降至 2.4、错误率保持 0，但候选 Recall 由 0.150 降至 0.025、F1@20 由 0.0179 降至 0.0083，arXiv 礼貌间隔使平均延迟增至 10.57 秒。B2 相对 B 的平均 API 由 8.2 降至 5.1、错误率由 0.402 降至 0.059、平均延迟由 29.78 降至 19.93 秒，OpenAlex 未再出现 400，Semantic Scholar 请求由 51 次降至 8 次；候选 Recall 由 0.170 降至 0.045、F1@20 由 0.0259 降至 0.0163。该结果只支持可靠性改善，不支持召回质量提升。

回归修复后的 A3 在相同前 10 条、arXiv-only 配置中使用安全原查询保底和有限核心补充：候选 Recall 为 0.250、Recall@20 为 0.225、F1@20 为 0.0274、平均 API 3.6、平均延迟 32.73 秒，来源错误率为 0。独立 5 条验证中，hybrid 与 safe-original 的候选 Recall、Recall@20 和 F1@5/10/20 相同。三源 B3 因 Semantic Scholar 连续最终 429 在 9/10 安全停止，只生成明确标注的部分指标，不据此声明完整多源质量收益。

本阶段增加 adaptive：先执行安全原查询，只在首轮候选量、核心或约束覆盖、元数据完整性不足且预算与来源状态允许时执行核心补充。A4 开发集上 adaptive 与 safe 的候选 Recall、Recall@20、F1@20 相同（0.150/0.125/0.0179），平均 API 2.7、延迟 25.37 秒；hybrid 为 0.250/0.225/0.0274、3.5 次 API、30.96 秒。adaptive 的 compact 执行率为 3.85%，事后 gold 增量为 0。

独立 V2 五条在三组执行完成后统一查看：safe、hybrid、adaptive 的候选 Recall、Recall@20 和 F1@5/10/20 完全一致；adaptive 使用 2.2 次平均 API、50.50 秒平均延迟，hybrid 为 3.0 次、54.00 秒，adaptive 未执行 compact。该结果通过本轮“不低于 safe”门槛并支持将 adaptive 作为产品默认，但不构成完整 Benchmark 的质量或延迟结论。

无 Semantic Scholar Key 的 M4 双源 adaptive 已完成：候选 Recall 0.170、Recall@20 0.145、F1@20 0.0259、平均 API 5.7、延迟 54.61 秒，compact 执行率 1.92%。OpenAlex 最终发生两次超时和一次 429，HTTP 400 为 0；结果如实保留，不能声明双源可靠性已稳定达标。

本轮按统一 120 秒预算启动 Query Evolution/RefChain 四组双源消融。开发集 baseline 的候选 Recall、Recall@20、F1@5/10/20 为 0.150、0.125、0.0222/0.0143/0.0179，平均 API 3.1、延迟 32.81 秒；OpenAlex 在 10 条中 0 次成功，并在 timeout 重试、HTTP 429 后持续 cooldown。依照预设保护条件，其余三组及独立验证集未启动；产物明确标记为不完整，当前不能判断两个模块的独立或组合边际收益，也不据此建议默认开启。

动态快照规划已在固定前 10 条上收敛，四组均完成 10/10 纯离线回放，执行期 HTTP、重试和网络等待为 0。baseline、Query Evolution、RefChain、组合组的必需键 success/failed 分别为 27/3、52/2、27/6、52/8；RefChain 相关 OpenAlex 请求均以 429 或 TLS timeout 的失败快照冻结，失败覆盖不表示成功检索。四组 F1@20 均为 0.0179；Query Evolution 新增 197 个唯一候选但没有新增 gold，记录为 `new_candidates_but_no_gold`，RefChain 没有生成候选且受来源失败主导，记录为 `no_action_generated` 和 `source_failure_dominated`。这些结果均为 `small_sample_diagnostic_only` 和 `insufficient_sample`，不能据此声明模块有效或建议默认开启。

随后将 Query Evolution 重构为 `off`、旧 `seed_expansion` 和 `coverage_gap` 三种策略。新策略从初始规则判断计算主题、方法、数据集、必要词、论文类型、venue 和时间覆盖，只在可行动缺口与可靠 seed 同时存在时生成最多两条短查询，并用通用约束质量门过滤补充候选。arXiv-only 冻结快照显示：开发 10 条和独立验证 5 条的 F1@20/Recall@20 均未低于 baseline，新策略请求量分别为 41/18，低于旧策略的 53/27；无效候选占比分别由 0.7462/0.8657 降至 0.6584/0.5938。但两组均没有新增 gold，尚未证明真实召回收益，故 API 与前端默认关闭 Query Evolution，旧策略只用于复现。

初始查询规划阶段加入 `facet_balanced` v1.2：从最终 QueryAnalysis 提取 topic、method、dataset、task、paper type、venue 和 temporal facet，优先显式约束，在 balanced 配额内保留原查询并选择最多两条互补查询。开发 10 条相对 `current_rules` 的候选 Recall/F1@20/Recall@20 持平，唯一候选 315→388、重复率 0.4167→0.2538、平均记录请求 2.7→2.6。冻结后独立验证 5 条的三项质量指标仍持平，唯一候选 145→168，但平均记录请求 2.4→2.6 且未新增 gold；因此未通过验收，产品默认仍为 `current_rules`。所有 Replay 执行期网络成本均为 0，结果不构成完整 Benchmark 结论。

`controlled_relaxation` v1.4 针对召回审计中“查询未匹配/过度约束”问题，仅使用查询与结构化约束生成至多两条来源无关补充查询；显式 must-have 保留，规则推断 must-have 不作为全量强 AND。规则在 AutoScholarQuery offset 50–69 的 20 条开发集后冻结，并在 offset 70–89 的 20 条独立验证集只运行一次。开发集候选 Recall、唯一 gold、F1@20 和 R@20 均与旧策略相同（0.1750、4、0.0093、0.0750），唯一候选 626→682、重复率 0.3479→0.2745、平均记录 API 2.40→2.45。验证集上述四项仍相同（0.1500、4、0.0139、0.1000），唯一候选 689→762、重复率 0.3848→0.3196、平均记录 API 均为 2.80，但补充查询新增 gold 为 0，MRR 0.0219→0.0183、nDCG@20 0.0356→0.0329。候选未通过“至少新增一个 gold”的预设门槛，产品默认继续使用 `current_rules`；20+20 小样本只说明本策略尚无切换证据。四次最终 Replay 的 HTTP、重试和网络等待均为 0。

来源互补性评测随后在 offset 90–109 开发集冻结流程，并对 offset 110–129 独立验证集只运行一次。arXiv-only 与 arXiv+OpenAlex 在验证集的候选 Recall、F1@20、R@20、MRR、nDCG@20 均为 0.0792、0.0133、0.0792、0.0205、0.0330，唯一 gold 均为 3；OpenAlex-only 没有候选，最终路径的记录请求全部失败，双源没有独占新增 gold。预设门槛未通过，默认来源和产品 profile 均未修改。该结论受当前 OpenAlex 429/TLS 失败环境限制，只能否定本次运行中的互补收益，不能证明来源本身长期无效。

`disjunctive_facets` v1.5 随后在 offset 130–149 开发集冻结，并对 offset 150–169 独立验证集只运行一次。验证集相对 `current_rules` 的候选 Recall 为 0.0750→0.1000、唯一 gold 为 2→3、唯一候选为 583→680、重复率为 0.3798→0.3061；F1@20 与 R@20 分别保持 0.0093 和 0.0750，平均记录 API 为 2.40→2.50，弱相关与无关候选占比为 0.5377→0.5254。预设验收全部通过，但 OR 子查询的 153 个独占候选没有事后独占 gold，新增 gold 来自整体规划组合，且样本仅 20 条；因此策略保持实验状态，产品默认仍为 `current_rules`。最终 Replay 执行期 HTTP、重试和网络等待均为 0。

全新 offset 170–209 的 40 条保留集在公共源冷却后沿用冻结协议续跑完成；三轮动态计划补齐基线 107 个键和析取组 99 个键，两组最终 Replay 均为 40/40 成功且执行期零网络。`disjunctive_facets` 相对 `current_rules` 的候选 Recall 为 0.1104→0.1229，唯一 gold 均为 8；F1@20、Recall@20、MRR、nDCG@20 则由 0.0154、0.1021、0.0516、0.0542 降至 0.0131、0.0896、0.0266、0.0389。OR 查询有 276 个独占候选和 1 个事后独占 gold，但整体未净增 gold。固定种子成对 bootstrap 未改变这一结论，预设门槛未通过；策略不进入 high_recall profile，不再围绕该保留集调参，默认继续使用 `current_rules`。

为隔离“替换旧规划”与“增加召回查询”的影响，本阶段实现 `current_plus_disjunctive` v1.6：旧查询内容、顺序和候选优先级不变，只在其全部完成且候选、延迟预算仍有余量时追加一条通用 OR。开发集 offset 210–229 中两组候选 Recall 和唯一 gold 同为 0.1250 与 4，OR 增加 107 个独占候选但无新增 gold；冻结后独立验证 offset 230–249 中两组候选 Recall、唯一 gold、F1@20、R@20、MRR、nDCG@20 同为 0.0625、2、0.0089、0.0625、0.0350、0.0391。加法策略保留全部基线 gold，新增 105 个 OR 独占候选但净新增 gold 为 0，平均记录 API 为 2.45→3.40。预设验收未通过，因此停止继续扩展 OR，不进入 high_recall profile，产品默认继续使用 `current_rules`。四组最终 Replay 执行期 HTTP、重试和网络等待均为 0；该 20+20 结果不代表完整 Benchmark。

随后实现 `facet_union` v1.7：在完整基线后按 dataset、method、task、topic 的稳定优先级追加至多一条独立分面查询，不使用 OR 或多分面强 AND。开发集 offset 250–269 中，两组候选 Recall 和唯一 gold 同为 0.1625 与 6，分面查询新增 387 个独占候选但没有独占 gold。规则冻结后，独立验证 offset 270–289 中两组候选 Recall、唯一 gold、F1@20、Recall@20 同为 0.1625、5、0.0228、0.1625；基线 gold 全部保留，但 MRR 0.0979→0.0965、nDCG@20 0.0933→0.0923，391 个独占候选仍无新增 gold。平均记录 API 2.40→3.55，弱相关与无关率 0.5562→0.6596，预设验收未通过。该策略仅作实验选项，产品默认保持 `current_rules`；规则式 OR、放宽和分面组合规划至此冻结，后续转向 LLM 语义规划或其他语义检索能力。四组最终 Replay 的 HTTP、重试和网络等待均为 0。

本阶段增加 `llm_semantic` 语义规划及独立快照机制。当前运行环境返回 `provider_disabled`，依照预设规则没有调用模型、没有生成 LLM 查询，也没有伪造对比指标；只对同一冻结快照重放 `current_rules`。开发 10 条与验证 5 条的候选 Recall/F1@20/Recall@20 分别为 0.1500/0.0179/0.1250 和 0.2000/0.0190/0.2000，回放执行期请求、重试和网络等待为 0。由于 LLM 组未运行，验收未执行，产品默认保持 `current_rules`，`llm_semantic` 默认关闭。

规则 Judgement 校准在同一批冻结 arXiv 候选上预先固定 128 个组合。开发 10 条选出的配置只改变高度相关阈值 0.72→0.68 与标题主题权重 0.12→0.10；F1@20、Precision@20、Recall@20 和 gold false negative 均未改善，MRR 0.027692→0.045000，平均返回量 11.4→10.2。配置冻结后，独立验证 5 条的各项排名指标与 gold false negative 均不变，平均返回量 11.2→9.4。两组候选召回一致且回放执行期网络成本为 0；该结果只支持保留为显式校准候选，不支持切换产品默认或声明真实性能提升。

未参与选参的固定保留集使用 AutoScholarQuery offset 20–49、单一 arXiv Snapshot 复核上述候选。两组 candidate Recall 都是 0.008333：65 个 gold 仅 1 个被 Retrieval 找到。`current_rules` 的 F1@20/Recall@20/MRR/nDCG@20 为 0.002778/0.008333/0.016667/0.008210；`calibrated_rules_v1` 将该 gold 判为弱相关，四项指标均为 0。固定 seed、5000 次配对 bootstrap 的 F1@20 差值为 -0.002778，95% 区间为 `[-0.008333, 0]`，其余核心指标区间也包含 0。这不证明旧规则总体更优，但足以否定校准配置在该保留集上的稳定优势；产品默认继续使用 `current_rules`，下一优先项是 Retrieval 召回，而非依据该保留集重新调阈值。

随后只做事后召回审计，不修改生产策略。65 个 gold 均可按 arXiv ID 获取，exact-title 搜索均排第 1；实际 adapted query 扩到 Top-20/50/100 只找回 1/5/7，说明扩大结果深度最多额外恢复 6 个。原始 64 个缺失中，查询未匹配 25、过度约束 23、词汇错配 10、排名截断或低排位 6；没有检测到 adapter 术语丢失、元数据不匹配或 gold 级来源不可用。规范化标题与核心词 oracle 分别找回 58/65 和 64/65，曾支持开展非 gold 驱动的规则放宽与分面实验；上述连续独立验证均未形成稳定净新增 gold，因此该方向现已冻结。全部分析通过冻结的 274 键离线 Replay 生成，执行期网络成本为 0。

## 未实现

- 完整 1000 条 AutoScholarQuery 的固定配置基线、重复实验和正式报告。
- 全文 PDF 获取、段落检索和可定位到原文片段的证据链。
- 持久化任务队列、跨进程 run store、共享缓存和服务重启恢复。
- 强制中止已经发出的 connector 请求。
- 对重排序、查询演化、RefChain 和归纳阶段的 LLM 实现；当前这些阶段为规则逻辑。

## 可靠性与可解释性

外部请求具有超时、有限重试或限速处理。错误不会被替换成示例论文，而是进入 `source_stats`、`warnings`、`missing_evidence` 和 SSE 事件。检索缓存与来源冷却可减少短期重复调用，但均为进程内机制。

最终归纳只消费已有候选和证据行；没有证据时返回证据不足或限制信息。当前证据来自标题、摘要、venue 与元数据，因此“带引用”表示引用到返回论文记录，不表示系统核验了全文。

## 验证结论

当前版本形成了可运行、可测试的真实检索工程闭环，但尚不能声明达到任何正式检索质量、效率或比赛成绩。下一阶段工作及验收标准见 [路线图](../roadmap.md)。
