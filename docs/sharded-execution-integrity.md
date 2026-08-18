# 离线分片执行与确定性归并门禁

`sharded_execution_integrity_v1` 只验证跨运行分片的完整性与归并等价性，不读取
gold/qrels，不计算 Recall、F1、Precision 或官方成绩，也不会发起网络、LLM 或
Snapshot 写入。

## 契约

`shard_plan_v1` 在执行前绑定完整 opaque query 集合、原始顺序、数据与 Replay 摘要、
Prompt、来源、预算、seed、约束、evaluator、execution profile 和 shard 数量。唯一分配
算法 `ordered_round_robin_v1` 按完整 query 列表的稳定序号执行
`ordinal % shard_count`；输入不包含 query 结果、失败状态、耗时、gold、case ID 或质量
指标。每个 shard 内继续保持全局相对顺序。

每个 shard attempt 在 `run_manifest_v1` 和原子 generation-zero 配置中同时绑定：

- plan 文件与 SHA-256；
- shard 编号、总数及预期 query 身份摘要；
-共同执行契约摘要；
- attempt ID 与可选的 `supersedes_attempt_id`。

重试选择采用 `unique_supersession_tip_without_outcome_selection`：每个 shard 的 attempt
必须构成单根、无分支、无环的显式链，唯一末端才是最终 attempt。选择过程不读取成功率、
候选、延迟或质量结果；双末端、缺父 attempt、陈旧/断裂谱系均是完整性违规。

## 归并边界

归并器只读取已提交 generation，不复制或改写 shard 历史。只有全部 shard 的唯一末端
attempt 均完整提交时才生成新的 `shard_aggregate_v1`。aggregate 按 plan 的全局 query
顺序写入记录，保留 `succeeded`、`failed`、`cancelled`、`excluded`，并登记每个 shard
manifest、attempt、generation manifest、事件数、记录数和文件 SHA-256。只保留成功项、
共同项、较优 attempt 或任何后验筛选都会失败；未完成 shard 返回 `not_ready`，不会生成
或声称 completed aggregate。

`run_manifest_v1`、`crash_consistency_v1` generation-zero 和
`reproduction_capsule_v1` 均支持可选 shard binding。字段缺失时旧运行仍按原格式解析；
旧 AutoScholarQuery Record160/Full1000 缺少预绑定计划和独立 shard 谱系，因此只读资格
审计固定返回 `not_eligible`，不会反推 160 条为合法分片。

若 shard manifest 登记了 `resource_ledger_v1`，aggregate 只引用每个 shard 所选唯一最终
attempt 的账本路径、哈希和 authority identity；被 supersede 或未选择 attempt 的账本不会
进入聚合。资源守恒仍由独立的 `resource_accounting_integrity_v1` 门禁验证。

## CLI 与退出码

```bash
PYTHONPATH=src python scripts/check_sharded_execution.py validate-plan \
  --plan path/to/shard_plan.json

PYTHONPATH=src python scripts/check_sharded_execution.py check \
  --plan path/to/shard_plan.json \
  --attempts path/to/attempts.json

PYTHONPATH=src python scripts/check_sharded_execution.py merge \
  --plan path/to/shard_plan.json \
  --attempts path/to/attempts.json \
  --output path/to/new_aggregate.json

PYTHONPATH=src python scripts/check_sharded_execution.py check-fixture
PYTHONPATH=src python scripts/check_sharded_execution.py audit-frozen
```

退出码：`0=passed`、`2=partition_or_merge_violation`、
`3=not_ready/not_eligible`、`4=usage_error`。错误只返回 shard、attempt、opaque query
identity、invariant、首个差异路径和规范化摘要，不回显查询正文或环境配置。

该门禁与其他证明边界不同：`execution_determinism_v1` 检查单次运行的执行方式一致性；
`experiment_pairing_integrity_v1` 检查 baseline/candidate 处理变量和覆盖对称；
`crash_consistency_v1` 检查单运行提交原子性；本门禁检查多个独立运行是否构成预注册、
无遗漏且可与单体运行等价的全量聚合。以上均不替代检索质量评测或官方 scorer。
