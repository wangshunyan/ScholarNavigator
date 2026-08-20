# 阶段一：候选召回与 Judgement 离线瓶颈分析

更新时间：2026-08-20。此文档仅分析完整成功的内部工程运行：
`contest_full_rules_v1`、`contest_full_dense_v1` 与
`contest_full_dense_reranker_v4`。所有 gold/qrels 均只在运行结束后的离线
评估与诊断中使用，未用于语料、索引、在线检索、查询改写或 Judgement 规则。
内部 F1/Recall 不等同赛事官方 scorer。

## 证据范围

分析输入为每个完整运行目录的 `metrics.json`、`stage_metrics.json` 与
`error_analysis.json`。三组均为 1000 条查询、`top_k=20`，并保存了完整的
resource ledger。以下数字是阶段诊断证据，不是对隐藏测试集或官方排名的声明。

| 运行 | 初始候选 Recall | Judgement 后 Recall | 最终 Recall@20 | F1@20 | 平均端到端延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rules | 0.13126 | 0.06437 | 0.06195 | 0.01087 | 0.711 s |
| Dense | 0.23503 | 0.16188 | 0.13508 | 0.02155 | 0.955 s |
| Dense + reranker v4 | 0.29675 | 0.19454 | 0.15010 | 0.02442 | 3.894 s |

## 检索贡献

- Dense 相对于 rules 将初始候选 Recall 从 `0.13126` 提高至 `0.23503`。
- reranker v4 链路的 BM25 命中 335 个 gold，Dense 摘要检索命中 441 个，二者
  重叠 196 个，精确并集为 580 个；这与该运行的已检索 gold 数一致。
- 因而 BM25 与 Dense 的互补性已有完整运行证据，后续变体必须继续保留两路候选
  并集和原始检索分数，不能退化为单一路径。
- 已检索之外仍有 1,822 个 gold 未进入 reranker v4 初始候选，是最大的绝对损失。
  当前只允许通过不使用 gold 的受控候选池、RRF 或查询变体在新的资格运行中测试。

## Judgement 与排序损失

reranker v4 共检索到 580 个 gold：

| 阶段 | gold 数量 | 损失 |
| --- | ---: | ---: |
| 初始候选 | 580 | - |
| Judgement 保留 | 383 | 197：135 个 weak，62 个 irrelevant |
| 最终 Top-20 | 278 | 105 个保留 gold 位于 Top-20 之外 |

Judgement 对已检索 gold 的保留率为 `0.66034`，错误过滤率为 `0.33966`。这说明
摘要缺失、词面差异或边缘满足约束的情况可能被硬阈值压低；但离线 gold 只能用于
确认这一瓶颈，不能编码进在线规则。

排序也仍有明确空间：383 个 Judgement 保留 gold 中有 105 个不在 Top-20，平均
gold rank 为 `14.36`，中位 rank 为 `8`。因此，软 Judgement 与候选排序必须分别
做独立消融，不能把任一变化的收益归因于另一变化。

## 查询规划结论

完整 reranker v4 运行使用 `current_rules`，记录 2,813 条子查询和 4,534 条适配
查询；其中 `original_query` 路径有 524 次记录。现有诊断没有一组“删除原始查询”
的配对运行，因此不能声称查询改写已经造成或没有造成主题词损失。

后续候选保持原始查询，并只允许有限的受控变体。查询规划的有效性必须以新 200
条资格运行的成对指标和 bootstrap 结果判断，而不是按本诊断推测。

## 后续实验边界

1. `dense_reranker_soft` 是独立候选，仅将 `partially_relevant_threshold` 从
   `0.45` 调为 `0.35`，保留所有硬约束、BM25+Dense、RRF、reranker、候选上限和
   查询配置。
2. 它必须先运行 `contest_qual200_dense_reranker_soft_v1`，与
   `contest_qual200_reranker_v4_gpu1` 做同一 200 条查询的配对门禁。只有 F1@20 或
   Recall@20 严格提升、bootstrap 95% 区间支持、零失败、零 fallback 和资源账本
   通过，才可启动新的完整 RunId。
3. `contest_full_dense_reranker_llm_v14` 已完成诊断审计：虽有 1000 条结果和零失败，
   但有 4 次 fallback，因此不将 LLM 作为实测提升写入结果，也不作为 v15 的资格依据。
