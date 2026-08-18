# Full1000 内容寻址备份压实

`formal_backup_compaction_v1` 是默认关闭的存储协议附录。它不修改 Full1000
查询、请求、预算、排序或结果，只约束已提交 shard 归档如何以 SHA-256
内容寻址、增量父链和可恢复压实基线保存。

每个封印 shard 最终只保留一份完整内容寻址归档；后续备份根只登记新增的
`COMMITTED` 代、变更索引和父根。压实点发布新的完整可恢复基线，但仍把旧 root
作为 parent/superseded root 保留在 append-only 发布链中。发布采用临时 staging
和原子提交；中断不会替换上一有效 root。唯一 blob、旧 root、审计链或当前恢复点
都不能因压实而删除。

## 最坏情况容量

旧备份资格门槛为 2,119,029,489,664 bytes 和 210,940 inode。新模型逐项相加：

- 20 个最终 shard 归档；
- 活动窗口 4；
- 1 个 shard 的压实 staging；
- 1,040 个提交代、每代最多 64 个索引项、每项 512 bytes；
- aggregate；
- 4 个 shard 的恢复工作区；
- 既有安全余量。

结果为 1,028,812,963,840 bytes 和 108,137 inode。压缩、稀疏文件、未来重复率、
未来清理收益的抵扣全部为 0；内容寻址复用只作为已存在不可变 blob 的完整性规则，
不作为预测容量收益。

## 恢复与边界

合成 1000-query 演练覆盖窗口 4 连续归档、多次压实、压实中断、旧 root 校验、
索引/父链/基线损坏、唯一 blob 误删、恢复后 resume、单 shard attempt 替代和
最终 aggregate 等价。恢复只选择最后一个完整有效 root，已提交 query 不重复调用，
资源账本只计入最终权威状态。

当前没有满足新容量、inode、quota 和独立故障域要求的真实备份目标，因此
`audit-readiness` 返回 exit 3。该工程能力不解除 Full1000、真实人工 Precision 或
官方 scorer/schema 阻断，也不证明检索质量。

```bash
python scripts/check_formal_backup_compaction.py calculate-capacity
python scripts/check_formal_backup_compaction.py simulate-compaction
python scripts/check_formal_backup_compaction.py verify-recovery
python scripts/check_formal_backup_compaction.py audit-readiness
```

退出码为 `0=backup_compaction_ready`、`2=compaction_or_recovery_violation`、
`3=not_ready_missing_qualified_backup_target`、`4=usage_error`。
