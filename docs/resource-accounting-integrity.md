# 运行资源账本与预算守恒门禁

`resource_accounting_integrity_v1` 是运行正确性门禁，不是预算优化器、成本估算器、质量
指标或官方 scorer。它只观察生产 SearchService、connector、LLM 预算、checkpoint/resume
和分片归并已经产生的权威事件，不改变请求、重试、分页、排序、去重或预算执行。

## 权威边界

可选的 `resource_ledger_v1` 以 opaque run/query/source/attempt identity 记录逐操作事实：

- 操作类型、稳定请求序号、父操作和终态；
- 预算预留、实际消费、释放、拒绝及剩余额；
- adapter 请求、分页、重试、返回记录、cache hit/miss；
- LLM 调用与响应中真实提供的 token usage；
- 取消、超时、429、部分完成及未执行；
- 语义事件、checkpoint generation、attempt 和 run manifest 身份。

权威汇总只读取完整原子提交代中的账本。兼容文件、日志、旧报告和文件大小不能成为计费
输入；resume 只选择已提交的新 attempt，分片 aggregate 只选择各 shard 的最终 attempt。
旧 AutoScholarQuery Record160/Full1000 没有逐操作账本及完整提交代，资格审计固定返回
`not_eligible`，不会从历史汇总反推资源消耗。

供应商响应或运行配置没有提供的 token/cost 使用 `not_available`；不适用字段使用
`not_applicable`。两者都没有数值，绝不以 `0` 伪装为已测量值。供应商价格不会被推测。

## 守恒关系

门禁不依赖并发事件到达顺序，而按稳定 operation identity 聚合并验证：

1. run 汇总等于 query 汇总之和，query 汇总等于来源/操作明细之和；
2. `reserved = consumed + released`，已知剩余额等于 `limit - consumed`；
3. 已知消费非负且不超过既有上限，不允许重复扣减；
4. 每次真实 adapter/LLM 调用都有唯一账目，失败、429、超时和异常也保留；
5. cache hit、预算拒绝、未启动和取消跳过不产生伪外部调用；
6. retry 与 page 分开计数，逻辑 query 不能冒充真实 request；
7. 取消后不能新增未授权消费；
8. 未提交 attempt、被 supersede attempt 和未选 shard 不进入权威汇总。

账本观察器默认关闭。启用后只订阅既有 connector/预算/语义事件；关闭观察器与启用观察器
运行同一本地 Replay fixture 时，结果、排序、统一身份、事件和预算状态必须完全一致。

## Manifest、提交代与胶囊

新 Benchmark 可使用 `--resource-ledger` 生成 `resource_ledger.json`。该文件作为
`resource_ledger_v1` 角色进入原子 completion generation，并可通过 `run_manifest_v1`
登记其相对路径、大小、SHA-256、manifest identity 和 committed authority。复现胶囊沿用
manifest 的封闭输出清单携带该文件；分片 aggregate 只引用所选最终 shard 的 manifest、
attempt、账本路径和哈希，不复制或改写 shard 历史。

## CLI 与退出码

```bash
PYTHONPATH=src python scripts/check_resource_accounting.py check-fixture
PYTHONPATH=src python scripts/check_resource_accounting.py check-fixture --resume-shard
PYTHONPATH=src python scripts/check_resource_accounting.py check-fixture --fault double_charge
PYTHONPATH=src python scripts/check_resource_accounting.py check --ledger path/to/resource_ledger.json
PYTHONPATH=src python scripts/check_resource_accounting.py check-shards \
  --aggregate path/to/shard_aggregate.json
PYTHONPATH=src python scripts/check_resource_accounting.py audit-frozen
PYTHONPATH=src python scripts/check_resource_accounting.py audit-registry
```

退出码为 `0=passed`、`2=accounting_or_budget_violation`、
`3=not_eligible/missing_authoritative_ledger`、`4=usage_error`。违规输出只包含脱敏
run/query/source/attempt identity、invariant、账本 JSON path、期望值、实际值和规范化摘要。
协议固定在 `benchmark/resource_accounting_integrity_v1_protocol.json`；专项回归命令为：

```bash
PYTHONPATH=src pytest -q -m resource_accounting_integrity_regression
```

“预算执行”决定生产流程是否还能继续；“资源账本守恒”证明已发生消耗被完整且唯一地记录；
`run_manifest_v1` 证明静态文件身份；实验成本分析解释消耗；Recall/F1 与官方 scorer 评估
检索质量。这些证明边界互不替代。
