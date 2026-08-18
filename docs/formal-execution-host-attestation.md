# Full1000 正式执行主机资格与环境封印

`formal_execution_host_attestation_v1` 是正式联网启动前的离线主机门禁。它绑定
Full1000 执行计划、启动控制、存储治理、崩溃一致性、灾难恢复和运行时封闭性协议，
只证明某次观测下的主机与两个存储目标是否满足执行条件，不运行检索，也不提供主机或
发布者身份认证。

## 资格边界

- 主盘至少需要 `713,501,442,048` bytes 与 `76,980` 个可用 inode。
- 备份目标至少需要 `2,119,029,489,664` bytes 与 `210,940` 个可用 inode。
- 主备文件系统配额和独立故障域必须有可验证观测。压缩率、稀疏文件和未来清理收益
  不计入容量。
- 可用字节与 inode 分别按 1 GiB 和 100,000 向下取整为保守下界，以排除探测期间的
  瞬时文件抖动；被舍弃余量不得用于满足容量门槛。
- `nofile` soft limit 至少为 256，进程 soft limit 至少为 64，路径上限至少为
  1024。
- 同文件系统临时目录必须通过原子 replace、file/dir fsync、advisory lock、
  写入及清理、非空恢复拒绝、大小写与 Unicode 语义探测。探测文件均为最小临时
  文件并自动清理，不会填满磁盘。

封印记录 OS 系列、架构、Python 运行时、文件系统类型以及脱敏能力事实，不记录
用户名、主机名、绝对路径、环境值、凭据或 `.env`。`host_scope_identity` 和存储
目标 identity 只是重用防护摘要，不构成主机身份认证。

## 启动授权

未来正式启动必须在原有 `full1000_launch_control_v1` 授权之外绑定
`full1000_host_attestation_addendum_v1`。封印必须为 `host_qualified`，且精确匹配
当前提交、opaque host scope、主盘目标和备份目标。提交、文件系统、容量、资源限制
或目标变化后必须重新探测；legacy authorization 不能复用。

当前仓库的只读封印返回 `not_ready_unverified_or_insufficient_host`：主盘可用空间
低于冻结门槛，主盘配额、备份容量/inode/配额/文件系统与主备独立故障域不可证明。
因此它不会解除 Full1000 阻断。

## CLI

```bash
PYTHONPATH=src python scripts/check_formal_execution_host.py probe \
  --primary-root <primary-root> --backup-root <backup-root>
PYTHONPATH=src python scripts/check_formal_execution_host.py verify-attestation \
  --attestation <attestation.json> --check-current-host \
  --primary-root <primary-root> --backup-root <backup-root>
PYTHONPATH=src python scripts/check_formal_execution_host.py simulate-profile
PYTHONPATH=src python scripts/check_formal_execution_host.py audit-readiness
```

退出码为 `0=host_qualified`、`2=host_capability_violation`、
`3=not_ready_unverified_or_insufficient_host`、`4=usage_error`。本门禁只验证工程
运行条件，不证明 Precision、Recall、人工评审或官方成绩。
