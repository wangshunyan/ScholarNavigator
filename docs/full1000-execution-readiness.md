# Full1000 正式基线执行就绪

`full1000_execution_readiness_v1` 是无网络预飞门禁，不是已完成的 Benchmark，也不运行
evaluator。协议精确绑定冻结的 1000 条 query-only 输入、current_rules 规划、Prompt manifest、
四源预算和相关离线运行契约。计划中的 query identity 全部是不透明摘要；本门禁不读取 gold、
qrels、case ID 或目标论文。

## 不可变执行边界

- 1000 条查询按冻结输入顺序闭合，并由 `ordered_round_robin_v1` 唯一分配到 20 个 shard；
- 所有实验开关（包括 `deterministic_tiebreak_v2`）关闭，排序与预算保持 current_rules；
- 每个 shard 预登记初始 attempt 和至多一次统一 incomplete retry。retry 必须显式
  supersede 初始 attempt，aggregate 只选择唯一可验证的谱系 tip；
- resume 只从所选 attempt 的最后完整原子提交代继续，取消后不得创建新 operation；
- 旧 Record160/162 缺少 `run_manifest_v1`、原子提交代、权威逐操作账本和完整来源血缘，
  因此只能作为历史参考，不能作为 checkpoint。新运行必须从全 1000 条开始；
- 网络预飞刻意禁用并报告 `network_not_checked`。凭据只允许正式启动时由应用配置加载器
  读取；本门禁不检查或读取 `.env`。

资源上限从冻结的 2,410 个子查询和四源计划直接计算，不从历史日志或产出反推。token、费用、
磁盘字节和供应商限额无法由冻结配置证明时保持 `not_available`。

## 命令

```bash
PYTHONPATH=src python scripts/check_full1000_execution_readiness.py build-plan
PYTHONPATH=src python scripts/check_full1000_execution_readiness.py verify-plan
PYTHONPATH=src python scripts/check_full1000_execution_readiness.py dry-run
PYTHONPATH=src python scripts/check_full1000_execution_readiness.py audit-readiness
```

`dry-run` 使用本地 fake adapter 数据执行 1000-query 的分片、原子提交/恢复、aggregate、
run-manifest 绑定、资源账本、胶囊文件上限预飞和 Top-20 交付检查；胶囊路径还在临时目录执行
一次 export→verify→replay，归档随临时目录清理。它不写 Snapshot、不持久化 fixture、不生成
质量统计，也不能作为正式运行结果。

退出码：`0=execution_plan_ready_but_network_blocked`、`2=plan_or_preflight_violation`、
`3=not_ready_missing_required_input`、`4=usage_error`。退出码 0 只表示计划和离线执行链路已
闭合；Full1000 未完成、真实人工 Precision 缺失和官方 scorer/schema 缺失三项正式阻断保持不变。
