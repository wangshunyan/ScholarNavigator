# 路线图

## 当前已验证基线

- 离线与批量评测共享结果选择、全标识符匹配及 Precision、Recall、F1、MRR、nDCG 实现；跨路径一致性已有离线测试。
- 默认正式结果包含高度相关和部分相关论文，不包含弱相关、不相关或证据不足论文。
- 报告同时输出仅成功案例与端到端指标，并区分失败、结果缺失和 gold 缺失。
- 四组查询演化与 RefChain 消融可在 sample fixture 上确定性运行，并输出质量与效率字段。
- 四个 connector、缓存、查询演化和 RefChain 已统一真实请求、重试、错误与等待统计；离线 mock 已验证，真实上游运行的长期成本分布仍待正式评测采集。
- SearchService 真实阶段事件、SSE 顺序回放、协作式取消和唯一终止事件已通过离线测试；已开始的单次 HTTP 请求仍需自然结束。
- AutoScholarQuery 的 1000 条查询和 2403 条 arXiv gold 已接入统一 Adapter；inspect、原始顺序子集、原子输出和 resume 已有离线测试。
- 已完成原始顺序前 5 条、单一 arXiv 源的真实 smoke；该结果只验证运行链路，不代表完整 Benchmark 性能。
- 阶段快照、gold drop reason、Judgement/Reranking 错误、来源独占贡献和规则瓶颈标签已接入 Benchmark Runner；固定前 10 条完成两组基线，剩余配置因公共源持续 429/超时暂停。
- 查询适配已改为安全原查询保底和核心查询补充，信息保留保护、精确 run 去重和完整 provenance 已通过离线测试；固定前 10 条 arXiv 开发诊断恢复候选 Recall，独立 5 条验证未低于 safe-original，但三源运行仍受持续 429 限制。
- 自适应查询策略已按候选充分性、预算和来源状态按需执行核心补充，并记录触发、跳过、成本和事后 gold 增量；固定开发集、独立验证集和无 Semantic Scholar 的双源结果仍需结合小样本限制解释。
- Query Evolution 与 RefChain 已补齐逐 case seed、动作、去重新增候选、事后 gold、Judgement/Top-K 丢失及边际成本诊断；固定开发集前 10 条四组均已纯离线验证，结果只作为小样本诊断。
- Benchmark 已支持动态快照的离线规划、有界串行采集、固定点检查、失败环境冻结和四组覆盖审计；四组均 replay-ready 且 replay-verified，回放执行期 HTTP、重试和网络等待为 0。
- Query Evolution 已增加覆盖缺口策略、候选质量门、策略级快照键和逐查询诊断；固定开发/验证子集的查询数、请求数和无效候选均低于旧 seed 扩展，但未新增 gold，产品开关保持默认关闭，真实召回增益仍待更大样本验证。
- 初始查询规划已增加 `facet_balanced` v1.2、facet provenance、版本化快照键和逐查询成本诊断；固定开发集质量持平且重复率下降，独立验证未新增 gold 且请求略增，因此产品默认保持 `current_rules`。
- 已实现默认关闭的 `llm_semantic`：外置版本化 Prompt、严格输出 Schema、确定性质量门、规则回退、独立 Record/Replay 和动态快照依赖均有离线测试；当前无可用 LLM 配置，尚未产生开发/验证收益证据。
- 规则 Judgement 已集中为版本化配置并输出可加和特征向量；冻结候选上的 128 组开发校准与一次独立验证均为零网络执行，候选配置只达到无回归，产品默认保持 `current_rules`。
- 固定 `offset=20, limit=30` 保留集已复用同一 arXiv Snapshot 比较两种 Judgement；校准配置未复现优势且过滤唯一召回 gold，默认保持 `current_rules`，64/65 gold 在初始 Retrieval 缺失。
- holdout30 召回审计确认 65/65 gold 仍可按 ID 与 exact title 获取；当前查询 Top-100 仅找回 7 个，64 个缺失中 48 个归于查询构造、10 个词汇错配、6 个排名截断，adapter 术语丢失为 0。
- 已实现 `controlled_relaxation` v1.4：原查询加至多两条通用放宽查询，显式必要词保持硬约束、推断必要词软化；固定 20 条开发集后在独立 20 条验证集只运行一次，新增候选但未新增 gold，故未切换默认策略。
- 已在固定 20 条开发集和独立 20 条验证集完成 arXiv/OpenAlex 单源与双源互补性 Replay；OpenAlex 最终路径全部失败且未新增 gold，故未形成 `high_recall` 候选，默认来源不变。
- 已实现实验性 `disjunctive_facets` v1.5：保留原查询，使用至多一条有界 OR 分面查询和一条可选组合查询；固定 20 条开发集冻结后，独立 20 条验证集新增 1 个唯一 gold，候选 Recall 提升且 F1@20、Recall@20 不退化，API 比为 1.0417，但 OR 查询本身未产生独占 gold，产品默认仍为 `current_rules`。
- 全新 `offset=170, limit=40` 保留集已在冷却后续跑完成：析取策略提高候选 Recall 但未净增 gold，F1@20、Recall@20、MRR 与 nDCG@20 均回退，故不进入 high_recall profile，不再围绕该保留集调参。
- 已实现 `current_plus_disjunctive` v1.6：完整保留旧查询后只用剩余候选预算追加一条 OR；固定 20 条开发集与独立 20 条验证集均未净增 gold，验证集 API 增至 1.39 倍，故停止继续扩展 OR，产品默认仍为 `current_rules`。
- 已实现实验性 `facet_union` v1.7：基线后最多追加一条独立分面查询；开发集与独立验证集均未净增 gold，验证集排序指标和噪声门槛未通过。规则式 OR、放宽和分面组合规划已冻结，产品默认保持 `current_rules`。
- sample fixture 只证明工程链路可运行，不代表检索性能已通过正式 benchmark 验证。

## P0

1. **完成正式基线**：在固定代码、数据、来源和预算下运行完整 AutoScholarQuery；验收标准是一条命令复现 1000 条逐查询 F1@5/10/20、端到端汇总和效率报告。
2. **校准官方口径**：核对 gold 转换、K 值和官方计分器差异；验收标准是共享样例与官方输出逐项一致。
3. **扩大查询适配验证**：在不使用 gold 生成查询的前提下扩大异质查询集；验收标准是多次运行中来源可靠性不退化且候选 Recall 不低于原始查询基线。
4. **复核来源互补性**：在 OpenAlex 可稳定成功的独立时间窗复跑冻结三组；验收标准是新增至少一个独占 gold、F1@20 与 Recall@20 不退化且 API 不超过 arXiv-only 两倍。
## P1

1. **扩大演化策略消融**：在同一正式数据与预算下扩大 `off`、`seed_expansion`、`coverage_gap` 对比；验收标准是新策略在不降低 F1@20/Recall@20 时产生可复现的新增 gold，且调用量不高于旧策略。
2. **增加误差切片**：按领域、查询意图和失败类型汇总；验收标准是所有切片沿用同一论文匹配与失败聚合口径。
3. **执行 LLM 规划消融**：在固定模型、Prompt 与双重 replay 下完成开发/独立验证对比；验收标准是验证集质量不退化、产生新增 gold 或明确排序提升，且检索 API 不超过规则基线 1.5 倍。
4. **暂缓 Judgement 再校准**：先提高冻结候选的 gold 覆盖再复核；验收标准是预注册保留集上的 F1@20 或 gold false negative 可复现改善，且 Precision@20、Recall@20 和返回量不退化。

## P2

1. **增加统计稳定性**：执行重复运行与显著性检验；验收标准是报告包含方差、置信区间和固定随机性配置。
2. **接入持续回归**：保存版本化基线并检测退化；验收标准是质量或效率越过预设阈值时自动失败。
