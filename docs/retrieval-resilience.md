# 四源离线故障降级门禁

`retrieval_resilience_v1` 使用与 Snapshot Replay 相同的
`connector_result_provider` 边界，把确定性结果或故障送入生产
`retrieve_papers`、`SearchService`、统一论文模型、身份去重、Judgement、排序和事件链。
它不建立另一套检索逻辑，不加载运行时配置，不访问 gold，也不计算 Precision、Recall 或
官方成绩。协议与无 gold fixture 分别冻结于
`benchmark/retrieval_resilience_v1_protocol.json` 和
`datasets/eval_fixtures/resilience/four_source_results.json`。

## 场景与不变量

固定矩阵覆盖四源的 timeout、HTTP 429、连接失败、malformed JSON、关键字段缺失、非法
类型、分页循环、重复记录、同类稳定标识冲突、空响应、Snapshot key 未记录、部分返回后
失败、两源/三源/四源失败和未预期异常。它只复现 connector 当前有界重试计数，不改变
重试、cooldown、查询、来源或预算。

每个场景都校验：

- 健康来源结果完整保留，异常 payload 不产生候选；
- 来源终态为 `success`、`success_empty`、`partial_completion`、`failed` 或
  `not_started`，并保留稳定原因；
- 统一身份去重保留重复合并和同类标识冲突不合并的现有语义；
- adapter 调用、物理请求、重试、单源 limit 与全局候选预算不越界；
- 查询规划、Judgement、排序及实验开关仍为冻结的 `current_rules`；
- 四源全部失败时终态为 `all_sources_failed` 且候选为空，不伪装为成功空响应；
- 阶段事件唯一有序，connector started/completed 成对；
- 错误中不保留凭据、认证头、环境文件名、绝对路径或非必要原始响应。

瞬态规范化只排除协议逐字段列出的 latency、rate-limit wait 和 budget elapsed；列表顺序、
未知字段和语义事件不会递归忽略或重新排序。

## CLI 与退出码

```bash
PYTHONPATH=src python scripts/check_retrieval_resilience.py check
PYTHONPATH=src python scripts/check_retrieval_resilience.py check --fault budget_overrun
PYTHONPATH=src python scripts/check_retrieval_resilience.py audit-frozen
PYTHONPATH=src pytest -q -m retrieval_resilience_regression
```

| 退出码 | 状态 | 含义 |
| --- | --- | --- |
| `0` | `passed` | 全部离线场景满足协议 |
| `2` | `invariant_violation` | 至少一个故障降级不变量被破坏 |
| `3` | `not_eligible` | fixture 或冻结产物不足以重放来源级故障 |
| `4` | `usage_error` | 参数、协议或 Schema 无效 |

输出使用稳定 JSON；违规只暴露 scenario、source、invariant、首个路径和两侧摘要，不输出
原始敏感值。`--fault budget_overrun` 仅在第一个场景注入可控计数缺陷，用于证明退出码
`2`，不会改变生产 connector。

## 与其他验证轴的边界

- 故障降级正确性：本门禁验证异常隔离、终态、预算与脱敏。
- 跨执行方式确定性：`execution_determinism_v1` 验证重复、重排、并发、resume 和取消。
- 产物完整性：`run_manifest_v1` 验证文件哈希、配置绑定和 checkpoint 谱系。
- Snapshot 效果回归：冻结 Replay 门禁比较候选与内部指标。
- 官方质量评测：必须使用官方数据和官方 scorer，以上门禁均不能替代。

当前 AutoScholarQuery 160/1000 冻结产物没有按来源注入并重放任意故障所需的完整 fixture
元数据，因此 `audit-frozen` 返回结构化 `not_eligible`。这不是故障降级通过或质量结论，
也不会修改、补全或反推冻结产物。
