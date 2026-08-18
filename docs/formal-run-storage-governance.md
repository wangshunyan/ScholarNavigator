# Full1000 存储容量、配额与保留门禁

`formal_run_storage_governance_v1` 是未来 Full1000 正式运行的离线存储控制层。它绑定既有
执行计划、启动控制、来源原始响应取证、资源账本和灾备协议，不执行真实检索，不读取凭据，
也不生成任何质量指标。

## 可证明上限

冻结执行计划给出 1,000 条 query、20 个 shard、最多 19,280 次 HTTP attempt 与 1,040 个
选定提交代。来源取证验证器已有每个原始响应 32 MiB、每代原始归档 512 MiB 的硬上限。存储
协议据此声明：

- 单个提交代最多 640 MiB/64 文件；
- 单 shard 最多 35,030,827,008 字节/3,344 文件；
- 单次运行最多 702,764,023,808 字节/66,980 文件；
- 两个保留恢复点连同一次原子 staging 的备份链最多 2,108,292,071,424 字节/200,940 文件；
- 主目标和备份目标各另保留 10 GiB 安全余量，且不把压缩、稀疏文件或未来清理收益计入容量。

这些数值是 fail-closed 配额，不是供应商响应大小预测。真实主盘/备份盘可用字节、inode 和
文件系统配额必须在启动前独立观测；当前均为 `not_available`，所以真实 readiness 返回 3。

## 写入和取证边界

新格式正式运行必须显式启用原始响应取证；产品默认仍关闭。单响应超过 32 MiB 时，attempt
以 `capture_size_exceeded` 终止，原文既不截断保存，也不进入 parser，更不会伪装成可重放
成功。所有写入在发布新代前依次执行预留、提交和释放；容量或 inode 在预留后下降时，未提交
预留被回滚，上一完整提交代不受影响。单运行目录只允许一个 writer。

清理只允许未提交临时文件，或已经完整备份、超过两代窗口且不再承担权威角色的旧代。当前
resume 点、资源账本、原始响应、操作审计、最终 attempt 和透明日志引用产物永不因保留窗口
自动删除。

## 离线命令

```bash
PYTHONPATH=src python scripts/check_formal_run_storage.py build-plan
PYTHONPATH=src python scripts/check_formal_run_storage.py simulate-pressure
PYTHONPATH=src python scripts/check_formal_run_storage.py verify-capacity \
  --primary-root <empty-formal-run-root> --backup-root <offsite-backup-root>
PYTHONPATH=src python scripts/check_formal_run_storage.py audit-readiness
```

退出码为 `0=storage_controls_ready`、`2=quota_or_retention_violation`、
`3=not_ready_capacity_unverified`、`4=usage_error`。注入式 1000-query 演练通过只证明配额、
低空间回滚、保留和并发控制有效；它不启动网络，不证明真实磁盘容量，也不解除 Full1000、
真实人工 Precision 或官方 scorer/schema 三项正式阻断。
