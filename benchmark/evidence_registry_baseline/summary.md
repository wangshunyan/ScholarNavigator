# 实验证据矩阵

> 仅汇总仓库已跟踪的内部 Benchmark/审计证据，不是官方赛题成绩。

- 策略数：24
- 默认开启：current_rules
- 证据状态：{"evidence_unavailable": 20, "tracked_machine_evidence": 4}

| 策略 | 类型 | 默认 | 证据 | 决策 | 指标版本 | 数据范围 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `calibrated_rules_v1` | judgement | 否 | evidence_unavailable | negative | `legacy_gold_records_v1` | AutoScholarQuery dev 0/10; AutoScholarQuery val 10/5; AutoScholarQuery holdout 20/30 | 独立 holdout 将唯一召回 gold 判为弱相关，最终指标归零；历史文档要求继续默认关闭。 |
| `concept_projection` | query_planning | 否 | evidence_unavailable | negative | `legacy_gold_records_v1` | SciFact fixed 50; AutoScholarQuery dev 0/10; AutoScholarQuery val 10/5 | SciFact 与 validation 持平，但 development 的 Recall@20/F1@20 退化，未形成跨数据集净增。 |
| `controlled_relaxation` | query_planning | 否 | evidence_unavailable | negative | `legacy_gold_records_v1` | AutoScholarQuery dev 50/20; AutoScholarQuery val 70/20 | 验证集新增独占候选但新增 gold 为 0，未通过预注册门槛。 |
| `current_plus_disjunctive` | query_planning | 否 | evidence_unavailable | negative | `legacy_gold_records_v1` | AutoScholarQuery dev 210/20; AutoScholarQuery val 230/20 | 完整保留基线的组合查询仍未增加 gold，并增加请求与候选噪声。 |
| `current_rules` | default_composite | 是 | tracked_machine_evidence | validated_default | `legacy_gold_records_v1` | SciFact fixed 50; AutoScholarQuery dev 0/10; AutoScholarQuery val 10/5 | 65 条冻结 Replay 回归基准通过；它是内部默认策略参考，不是 AutoScholarQuery 1000 条正式基线或官方成绩。 |
| `disjunctive_facets` | query_planning | 否 | evidence_unavailable | negative | `legacy_gold_records_v1` | AutoScholarQuery dev 130/20; AutoScholarQuery val 150/20; AutoScholarQuery holdout 170/40 | 早期小样本通过，但预注册 40 条 holdout 的 Recall@20/F1@20 退化，最终保持实验状态。 |
| `facet_balanced` | query_planning | 否 | evidence_unavailable | negative | `legacy_gold_records_v1` | AutoScholarQuery dev 0/10; AutoScholarQuery val 10/5 | 开发和验证均未新增 gold，验证请求成本上升，未通过验收。 |
| `facet_union` | query_planning | 否 | evidence_unavailable | negative | `legacy_gold_records_v1` | AutoScholarQuery dev 250/20; AutoScholarQuery val 270/20 | 验证集新增候选但新增 gold 为 0，且请求和弱/无关候选显著增加。 |
| `lexical_normalization_v1` | judgement_evidence_matching | 否 | tracked_machine_evidence | promising_default_off | `deduplicated_gold_identity_v2` | SciFact fixed 50; AutoScholarQuery dev 0/10; AutoScholarQuery val 10/5; AutoScholarQuery Record160 | 内部点估计为正且无已知 query 退化，但簇级区间均含 0，人工 Precision 尚未完成，必须保持默认关闭。 |
| `llm_constrained_rewrite` | query_planning_llm | 否 | evidence_unavailable | inconclusive | `legacy_gold_records_v1` | SciFact fixed 50; AutoScholarQuery dev 0/10; AutoScholarQuery val 10/5 | 历史评测存在来源终态与 fallback 可归因限制，未证明跨三个集合稳定净增。 |
| `llm_judgement` | judgement_llm | 否 | evidence_unavailable | unvalidated | `not_applicable` | 未评测 | 仅有实现与降级语义说明，没有跟踪的正式配对 Benchmark、成本收益或稳定性证据。 |
| `llm_query_understanding` | query_understanding_llm | 否 | evidence_unavailable | unvalidated | `not_applicable` | 未评测 | 仅有严格 JSON 与规则 fallback 的实现说明，没有跟踪的正式配对效果证据。 |
| `llm_semantic` | query_planning_llm | 否 | evidence_unavailable | inconclusive | `legacy_gold_records_v1` | AutoScholarQuery dev 0/10; AutoScholarQuery val 10/5 | SciFact 缺少完整策略产物，且历史阶段曾受 provider/调用完整性阻断，不能形成跨集合结论。 |
| `local_bm25` | retrieval_source | 否 | tracked_machine_evidence | promising_default_off | `legacy_gold_records_v1` | SciFact fixed 50 closed corpus | 在 SciFact 官方封闭语料上显著增加内部召回，但不是开放网络成绩，且语料配置专属，保持默认关闭。 |
| `local_bm25_original_deepening` | benchmark_only_retrieval | 否 | tracked_machine_evidence | negative | `legacy_gold_records_v1` | SciFact fixed 50 closed corpus | 只新增 1 条候选 gold，最终 Recall@20/F1@20 均未提升；不建议继续。 |
| `prf_v1` | query_planning_feedback | 否 | evidence_unavailable | negative | `legacy_gold_records_v1` | SciFact fixed 50; AutoScholarQuery dev 0/10; AutoScholarQuery val 10/5 | 严格可比子集没有增益，且全量结果受来源响应漂移影响，不满足继续门槛。 |
| `query_adapter_hybrid` | query_adapter | 否 | evidence_unavailable | unvalidated | `not_applicable` | 未评测 | 可选 adapter 已实现，但没有独立、跟踪的配对 Benchmark 证据。 |
| `query_adapter_safe_original` | query_adapter | 否 | evidence_unavailable | unvalidated | `not_applicable` | 未评测 | 可选 adapter 已实现，但没有独立、跟踪的配对 Benchmark 证据。 |
| `query_evolution_coverage_gap` | query_evolution | 否 | evidence_unavailable | inconclusive | `legacy_gold_records_v1` | AutoScholarQuery dev 0/10 | 开发 10 条只增加候选、没有新增 gold，且存在来源失败；样本不足以判断。 |
| `query_evolution_seed_expansion` | query_evolution | 否 | evidence_unavailable | inconclusive | `legacy_gold_records_v1` | AutoScholarQuery dev 0/10 | 开发 10 条未形成新增 gold，且来源失败与小样本阻止稳定结论。 |
| `refchain` | citation_expansion | 否 | evidence_unavailable | blocked | `legacy_gold_records_v1` | AutoScholarQuery dev 0/10 | 冻结 reference 请求失败主导且未产生扩展候选，不能评价引用链收益。 |
| `result_policy_highly_only` | result_filtering | 否 | evidence_unavailable | unvalidated | `not_applicable` | 未评测 | 严格过滤选项已实现，但没有独立、跟踪的配对 Benchmark 证据。 |
| `rrf_fusion` | ranking | 否 | evidence_unavailable | negative | `legacy_gold_records_v1` | SciFact fixed 50; AutoScholarQuery dev 0/10; AutoScholarQuery val 10/5 | 冻结候选排序消融在 SciFact 与 development 退化、validation 持平，未形成稳定价值。 |
| `semantic_seed_expansion` | recommendation_expansion | 否 | evidence_unavailable | negative | `legacy_gold_records_v1` | SciFact fixed 50; AutoScholarQuery dev 0/10; AutoScholarQuery val 10/5 | 官方 ID 解析增加可用 seed，但推荐没有独立 gold，最终指标未满足三集合继续门槛。 |

## 全局阻断

- `official_scorer_unavailable`：官方赛题发布材料没有精确 scorer、F1/K、身份去重及平均口径；注册表中的分数均为内部指标，不能称为官方成绩。

## 默认策略门禁

只有 `current_rules` 可以默认开启；任何实验项若无通过证据、处于负面、阻断、不可判定或证据不可用状态，默认开启都会使门禁失败。
