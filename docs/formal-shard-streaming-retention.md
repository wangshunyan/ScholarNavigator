# Full1000 分片流式归档与本地释放

`formal_shard_streaming_retention_v1` 是 Full1000 存储治理的只读 addendum。
它把主盘容量模型从“20 个 shard 同时常驻”改为预注册的有限活动窗口，但不改变
query 归属、请求清单、attempt、结果、完成顺序、预算或 aggregate 语义。该能力
默认关闭；当前 addendum 固定 `active_shard_window=4`，运行中不得调整。

## 权威边界

每个可释放 shard 必须先得到唯一最终 attempt，并验证完整 `COMMITTED` 链。归档
必须同时包含 manifest、generation、resource ledger、原始响应取证、语义事件、
操作审计链和 Top-20 结果，并写入合格备份目标。逐文件哈希、父归档链和恢复演练
全部通过后，才可追加 `eviction_started`/`eviction_completed` receipt。删除中断
只留下 started receipt，原权威副本继续保留。

当前 generation、resume 点、未完成 attempt、未备份原始响应、仅有单副本的证据，
以及仍被 transparency/revocation 以本地路径引用的文件都不可释放。aggregate
只能按权威哈希混合读取本地 shard 和已验证归档；归档离线、篡改、重复 shard 或
版本漂移均 fail-closed。

## 容量模型

不计压缩、稀疏文件或未来清理收益。主盘要求为：

`active window × 单 shard 峰值 + 1 × 归档 staging + aggregate + 安全余量`

| 活动窗口 | 所需主盘字节 | 当前可观测主盘 587,336,777,728 bytes |
| --- | ---: | --- |
| 1 | 82,946,555,904 | 满足 |
| 2 | 117,977,382,912 | 满足 |
| 4 | 188,039,036,928 | 满足 |

这只解除单盘“20 shard 同时常驻”的容量结构限制。合格备份容量、inode、quota
和独立故障域仍为 `not_available`，所以真实 readiness 必须返回 exit 3，不能启动
Full1000 或解除正式验证阻断。

## 离线验证

```bash
PYTHONPATH=src python scripts/check_formal_shard_retention.py build-addendum
PYTHONPATH=src python scripts/check_formal_shard_retention.py simulate-streaming
PYTHONPATH=src python scripts/check_formal_shard_retention.py audit-readiness
```

退出码为 `0=streaming_retention_ready`、
`2=retention_or_recovery_violation`、
`3=not_ready_missing_qualified_backup`、`4=usage_error`。1000-query fake
演练覆盖窗口 1/2/4、归档释放、删除中断、篡改、备份不可用、过早释放、恢复、
双 writer 和混合 aggregate；它不访问网络、不写 Snapshot，也不产生质量指标。
