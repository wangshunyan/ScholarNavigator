# Full1000 来源健康监督

`formal_provider_health_supervisor_v1` 是未来 Full1000 正式联网运行的安全暂停门禁。
它只消费来源请求终态、资源账本、提交代进展、容量状态和原始响应取证写入状态，
不读取 query 文本、论文身份、结果内容、gold、qrels 或质量指标。本协议与当前离线
任务均不会发起网络、LLM 或 Snapshot 写入。

## 状态与预注册阈值

状态按 `healthy → degraded → pause_required → paused → resume_eligible` 演进；
契约或事件不一致进入 `invalid`。阈值在首次正式请求前固定：

- 单来源连续 3 次失败进入 `degraded`，连续 6 次失败要求暂停；
- 20 次滚动窗口中至少 12 个观测且 429/503/timeout/连接失败比例达到 0.75 时暂停；
- 连续 10 次成功请求却没有新增解析进展时暂停；
- 3 个来源同时退化、连续 40 个外部操作没有新 `COMMITTED` 代，或至少 20 次
  无提交操作达到冻结 HTTP attempt 上限的 0.001 时暂停；
- 容量下降或原始响应取证写入失败立即要求暂停；
- 供应商费用、token 和限额无法由现有协议证明，保持 `not_available`。

这些阈值统一适用于所有 query、shard 和来源，不依据结果内容、query 类型、case ID
或效果表现选择样本。

## 安全暂停与恢复

进入 `pause_required` 后，runner 不得启动新 query、分页或 retry。已经登记为在途的
操作可按既有语义结束并进入资源账本；全部排空后才可发布完整暂停 checkpoint。
失败、取消和来源失败 query 保持权威终态，不得删除、改成排除项或仅保留共同成功项。

恢复必须同时绑定：

- 最后一个完整 checkpoint；
- fresh 的主机封印、容量观测、启动授权和协议；
- 每个来源满足冻结窗口的健康解除证据。

恢复不会清零累计失败/调用账本，不得切换来源、调整阈值或重复请求已经提交的 query。
再次退化会重新触发相同状态机。

## 离线命令

```bash
PYTHONPATH=src python scripts/check_formal_provider_health.py verify-policy
PYTHONPATH=src python scripts/check_formal_provider_health.py simulate-run
PYTHONPATH=src python scripts/check_formal_provider_health.py verify-resume
PYTHONPATH=src python scripts/check_formal_provider_health.py audit-readiness
```

前三个命令只使用本地冻结契约和 fake 事件。`audit-readiness` 当前必须返回 exit 3
`external_provider_health_not_observed`：控制链已验证，但正式运行尚未启动，也没有真实
来源健康观测。该工程能力不解除 Full1000、真实人工 Precision 或官方 scorer/schema
三项阻断，且不证明检索质量或官方成绩。
