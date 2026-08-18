# Full1000 独立备份目标资格

`formal_backup_target_attestation_v1` 复用 Full1000 执行计划、流式分片保留、
存储治理、灾难恢复、主机/站点封印、多卷拓扑和启动控制中的既有要求。它只验证
备份基础设施，不启动正式运行，不读取运行凭据，也不改变三项正式验证阻断。

## 资格契约

资格包以固定 ZIP 元数据封装标准库探针、验证器和一次性 challenge。探针可在无仓库
节点使用 `python -I -S` 运行，并只输出脱敏摘要。目标必须同时证明：

- 可用容量、quota 和保留空间均不少于 `2,119,029,489,664` bytes；
- 可用 inode 不少于 `210,940`，支持至少两个并发 writer；
- 支持同文件系统原子 replace、file/dir fsync、advisory lock、权限、路径能力、
  写入—校验—删除、增量父链和空目录恢复；
- 目标的设备、文件系统、管理域和故障域证据与主盘明确不同；远端服务还必须提供
  可验证的存储服务身份。不同目录、别名或挂载名不构成独立故障域。

quota、故障域或存储服务证据不足时保持 `not_available` 并 fail closed。摘要只证明
内容一致性，不证明设备所有权、主机身份或操作者身份。

## 离线命令

```bash
PYTHONPATH=src python scripts/check_formal_backup_target.py build-kit \
  --challenge <64位十六进制challenge> --issued-epoch <整数> \
  --output <临时zip>
PYTHONPATH=src python scripts/check_formal_backup_target.py verify-attestation \
  --kit <临时zip> --attestation <节点输出json>
PYTHONPATH=src python scripts/check_formal_backup_target.py simulate-targets
PYTHONPATH=src python scripts/check_formal_backup_target.py audit-readiness
```

退出码固定为：`0=backup_target_qualified`、
`2=attestation_or_failure_domain_violation`、
`3=not_ready_no_qualified_backup_target`、`4=usage_error`。真实导入还要求 fresh、
非合成的 attestation；challenge 只能消费一次，容量下降、身份替换、跨提交/计划复用
或目标漂移均需重新封印。

当前仓库没有合格的真实备份目标及独立故障域证据，因此只读 readiness 稳定返回 3。
流式保留、灾难恢复、host/site 和 launch 复审均保持阻断；合成矩阵不能替代真实资格。
