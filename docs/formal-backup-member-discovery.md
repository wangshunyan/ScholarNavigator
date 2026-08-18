# Full1000 备份成员只读发现

`formal_backup_member_discovery_v1` 只枚举协议明确登记的本机目标。操作者必须以
`alias=path` 显式提供路径，协议同时绑定该路径的摘要；发现器只对该目录本身调用
`stat/statvfs`，不递归扫描用户目录、兄弟目录、网络账户或未登记挂载。报告仅保留
opaque 目标、设备、文件系统、quota pool 和故障域摘要，不保存绝对路径、主机名、
用户名、环境值或凭据。

发现记录永远保持 `candidate`。容量、inode、quota、writer、独立故障域、
atomic replace、file/dir fsync、advisory lock、恢复等任一证据未知时均记为
`not_available`，不能进入匹配。完整候选下一步仍须通过
`formal_backup_target_attestation_v1`，再由
`formal_backup_set_member_intake_v1` 绑定 slot、challenge 和真实 attestation；
发现器不会登记或激活成员，哈希也不认证设备所有权或操作者身份。

候选去重对 device、filesystem、quota pool、failure domain 和 management domain
执行传递闭包；任一已知身份相同即按一个容量域计数。2/3/4 成员匹配直接复用冻结
topology 的逐 slot bytes、inode、quota 和 writer 门槛，不做总容量替代，也不把
未知证据当作通过。

```bash
python scripts/check_formal_backup_member_discovery.py discover
python scripts/check_formal_backup_member_discovery.py verify-candidate \
  --candidate candidate.json
python scripts/check_formal_backup_member_discovery.py match-topology \
  --candidates candidates.json
python scripts/check_formal_backup_member_discovery.py simulate-profiles
python scripts/check_formal_backup_member_discovery.py audit-readiness
```

退出码固定为：`0=qualifying_candidates_discovered`、
`2=discovery_or_identity_violation`、`3=no_real_qualifying_candidates`、
`4=usage_error`。当前冻结协议未登记真实目标，因此真实审计稳定返回 3；合成
profile 只验证控制语义，不进入真实成员登记或解除 Full1000 阻断。
