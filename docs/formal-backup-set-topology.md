# Full1000 多目标备份集拓扑

`formal_backup_set_topology_v1` 是默认关闭的存储协议附录。它把
`formal_backup_compaction_v1` 已证明的最坏情况容量拆到 2、3 或 4 个独立
backup member；既有单目标协议及默认行为保持不变。该门禁只验证基础设施容量、
身份与恢复闭合，不启动 Full1000，也不证明检索质量。

## 分配与容量

shard `s` 固定分配到 `s mod member_count`。每个 shard 的完整内容寻址归档、
manifest、generation、ledger、raw response、事件和审计链都留在同一 member；
单个 blob 不跨目标拆分，也不依赖跨文件系统原子 rename。

每个 member 独立计入自身最终 shard 归档、固定拆分的 active/recovery window、
一个原子 compaction staging、一份父链索引、安全余量份额，以及 member-0 上的
aggregate。最大单 member 门槛如下：

| member 数 | 最大 bytes | 最大 inode |
| ---: | ---: | ---: |
| 2 | 533,012,676,608 | 56,321 |
| 3 | 426,130,625,878 | 44,623 |
| 4 | 285,112,532,992 | 30,413 |

每个 member 都各自承担 staging 与索引，所以完整 set 总容量会略高于
1,028,812,963,840 bytes 的单目标压实需求。这是明确的保守开销；模型对压缩、
稀疏文件、未来去重和未来清理的容量抵扣均为零。

## 身份与恢复边界

每个 member 必须分别证明 bytes、inode、quota、writer 上限、文件系统身份及
与 primary 不同的故障域。member 之间的文件系统、quota pool 和故障域不得
重复；不同目录或同设备别名不能重复计容。

备份集是一个完整恢复单元，不声明 member 故障冗余。恢复前必须验证全部 member
root、inventory、父链、hash 和 20 个 shard 的唯一覆盖。任一 member 缺失、
离线、替换、篡改，或出现重复/遗漏 shard、旧 member 回滚、跨 set 混用、索引
冲突、部分恢复和双 writer 时均 fail closed。

## 离线命令

```bash
python scripts/check_formal_backup_set.py build-topology --members 4
python scripts/check_formal_backup_set.py calculate-capacity
python scripts/check_formal_backup_set.py verify-set
python scripts/check_formal_backup_set.py simulate-set --members 4
python scripts/check_formal_backup_set.py audit-readiness
```

退出码为 `0=backup_set_ready`、`2=topology_or_recovery_violation`、
`3=not_ready_missing_qualified_members`、`4=usage_error`。当前没有真实、完整
且独立的 backup member set，因此真实 readiness 固定返回 3；Full1000、真实
人工 Precision 与官方 scorer/schema 三项正式阻断均不变。
