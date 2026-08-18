# 离线评测汇总

查询数：1

> sample fixture 仅验证评测流程，不代表真实 benchmark 性能。

## 端到端指标

| 分组 | F1@5 | F1@10 | F1@20 | R@5 | R@10 | R@20 | P@5 | P@10 | P@20 | MRR | nDCG@5 | nDCG@10 | nDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.250 | 0.154 | 0.087 | 0.333 | 0.333 | 0.333 | 0.200 | 0.100 | 0.050 | 1.000 | 0.542 | 0.542 | 0.542 |
| query_evolution_only | 0.500 | 0.308 | 0.174 | 0.667 | 0.667 | 0.667 | 0.400 | 0.200 | 0.100 | 1.000 | 0.884 | 0.884 | 0.884 |
| refchain_only | 0.500 | 0.308 | 0.174 | 0.667 | 0.667 | 0.667 | 0.400 | 0.200 | 0.100 | 1.000 | 0.688 | 0.688 | 0.688 |
| query_evolution_plus_refchain | 0.750 | 0.462 | 0.261 | 1.000 | 1.000 | 1.000 | 0.600 | 0.300 | 0.150 | 1.000 | 1.000 | 1.000 | 1.000 |

## 仅成功案例指标

| 分组 | F1@5 | F1@10 | F1@20 | R@5 | R@10 | R@20 | P@5 | P@10 | P@20 | MRR | nDCG@5 | nDCG@10 | nDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.250 | 0.154 | 0.087 | 0.333 | 0.333 | 0.333 | 0.200 | 0.100 | 0.050 | 1.000 | 0.542 | 0.542 | 0.542 |
| query_evolution_only | 0.500 | 0.308 | 0.174 | 0.667 | 0.667 | 0.667 | 0.400 | 0.200 | 0.100 | 1.000 | 0.884 | 0.884 | 0.884 |
| refchain_only | 0.500 | 0.308 | 0.174 | 0.667 | 0.667 | 0.667 | 0.400 | 0.200 | 0.100 | 1.000 | 0.688 | 0.688 | 0.688 |
| query_evolution_plus_refchain | 0.750 | 0.462 | 0.261 | 1.000 | 1.000 | 1.000 | 0.600 | 0.300 | 0.150 | 1.000 | 1.000 | 1.000 | 1.000 |

## 案例统计与效率

| 分组 | 总案例 | 成功 | 失败 | 缺少结果 | 缺少 gold | 成功率 | 平均延迟（秒） | 平均 API | 平均检索 API | 平均引用 API | 平均重试 | 平均错误 | 平均缓存命中 | 平均限流等待（秒） | 平均 LLM 调用 | 平均 LLM Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1 | 1 | 0 | 0 | 0 | 1.000 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| query_evolution_only | 1 | 1 | 0 | 0 | 0 | 1.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| refchain_only | 1 | 1 | 0 | 0 | 0 | 1.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| query_evolution_plus_refchain | 1 | 1 | 0 | 0 | 0 | 1.000 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |


## 单查询结果

### sample_llm_rerank

latest LLM reranking methods for scientific literature retrieval

| 分组 | 排名标识 | 警告 |
| --- | --- | --- |
| baseline | doi:10.123/baseline | fixture simulated source warning, fixture_missing_retrieval:LLM reranking methods scientific literature retrieval method |
| query_evolution_only | doi:10.123/baseline, doi:10.123/evolved | fixture simulated source warning, fixture_missing_retrieval:LLM reranking methods scientific literature retrieval method, fixture_missing_retrieval:llm reranking retrieval LLM reranking methods literature, fixture_missing_retrieval:LLM reranking methods literature retrieval scientific LLM reranking |
| refchain_only | doi:10.123/baseline, doi:10.123/refchain | fixture simulated source warning, fixture_missing_retrieval:LLM reranking methods scientific literature retrieval method |
| query_evolution_plus_refchain | doi:10.123/baseline, doi:10.123/evolved, doi:10.123/refchain | fixture simulated source warning, fixture_missing_retrieval:LLM reranking methods scientific literature retrieval method, fixture_missing_retrieval:llm reranking retrieval LLM reranking methods literature, fixture_missing_retrieval:LLM reranking methods literature retrieval scientific LLM reranking |
