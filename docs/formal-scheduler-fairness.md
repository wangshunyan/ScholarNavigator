# Full1000 并发调度公平性与背压

`formal_scheduler_fairness_v1` 是未来 Full1000 正式执行的离线调度控制门禁。它只消费
冻结的 opaque query identity、全局顺序、20 个 shard、4 个来源及运行状态，不读取 query
文本、论文结果、来源产出、gold、质量指标或完成速度，也不实现新的检索器。

## 调度契约

初始任务按“来源轮次→权威 query 顺序”排列，并根据 query ordinal 旋转来源，使 query、
shard 和 source 都获得有界首次服务。所有 4,000 个 query×source 初始操作获得调度机会前，
分页和 retry 不得进入 worker。后续任务按固定 round、kind、query ordinal、source ordinal
排序；完成速度只决定任务何时 ready，不改变同一 ready 集合的优先级。

预注册上限为全局 12、单来源 3、单 shard 2 个并发操作；continuation backlog 达到 8 时，
对应来源收紧为 2。分页最多两页，retry 沿用冻结计划中 arXiv/OpenAlex/Semantic Scholar
各一次、PubMed 零次的上限，整个运行不得超过 19,280 个 attempt。

## 暂停、取消与恢复

`pause_required` 或 `cancel_required` 后不再接纳新操作，已在途任务按既有语义结束并生成
唯一账本项。暂停 checkpoint 绑定初始/continuation 队列、已完成任务、账本、query 终态及
source/shard 公平游标。resume 必须引用该 checkpoint，并要求 fresh 授权、主机、健康和协议；
队列及游标不能重置。取消和失败 query 仍以原 1000 条顺序保留，aggregate 不允许只选择成功
或先完成项。

## 离线命令

```bash
PYTHONPATH=src python scripts/check_formal_scheduler_fairness.py verify-policy
PYTHONPATH=src python scripts/check_formal_scheduler_fairness.py simulate-load
PYTHONPATH=src python scripts/check_formal_scheduler_fairness.py verify-resume
PYTHONPATH=src python scripts/check_formal_scheduler_fairness.py audit-readiness
```

退出码为 `0=scheduler_controls_ready`、`2=fairness_or_backpressure_violation`、
`3=external_run_not_started`、`4=usage_error`。当前真实审计必须返回 3：离线控制已验证，
但正式联网运行尚未开始。报告中的等待步、服务间隔、队列/并发峰值和首次执行覆盖率只用于
调度正确性，不是性能基准、检索质量指标或官方成绩。
