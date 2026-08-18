# Full1000 正式运行灾难恢复

`formal_run_disaster_recovery_v1` 是未来 Full1000 昂贵联网运行的离线工程门禁。它验证备份发布、异地恢复、断点续跑和最终聚合的完整性，不运行真实网络、LLM 或学术 API，也不计算检索质量或官方成绩。

## 权威边界

- 恢复契约绑定 Full1000 执行计划、启动授权、20 个 shard/attempt、原子 `COMMITTED` 代、资源账本、来源原始响应取证、操作审计和 aggregate。
- 备份只收录生产 `BenchmarkRunCommitStore` 中完整提交的代以及经过验证的 authority 文件。兼容镜像、临时文件、未提交代、锁文件、`.env` 和绝对路径均不属于权威恢复输入。
- 每个文件以大小和 SHA-256 登记；备份使用内容寻址对象、父备份链和稳定恢复点。发布顺序为同目录临时文件、刷新、原子替换和目录同步。
- 恢复必须进入全新空目录，先在同级 staging 中完整重建并验证，再原子发布。失败 staging 会被清理，不能留下可误用的部分权威目录。
- 若启动授权绑定的提交与恢复环境不兼容，恢复结果只能只读审计，必须重新授权后才能 resume。

## 离线演练

演练使用与 gold、qrels 和质量指标无关的 1000-query fake adapter：

1. 完成前 10 个 shard（500 个 query）并生成第一份增量备份；
2. 删除原工作目录，在新目录恢复；
3. 从已提交 cursor 继续，已提交 query 不再次调用 adapter；
4. shard 15 的首 attempt 明确失败并由 attempt 1 替代；
5. 完成 20 个 shard 后 aggregate，并与不中断控制运行比较 aggregate、资源请求、操作审计和 Top-20 交付摘要。

故障矩阵实际注入备份中断、成员遗漏、对象篡改、父链断裂、混合代际、审计链截断、原始响应缺失、旧备份回滚、非空目标、双恢复者竞争和恢复后重复计费。它们只验证恢复控制，不代表正式 Full1000 已运行。

## 命令与退出码

```bash
PYTHONPATH=src python scripts/check_formal_run_recovery.py simulate-disaster
PYTHONPATH=src python scripts/check_formal_run_recovery.py audit-readiness
PYTHONPATH=src python scripts/check_formal_run_recovery.py verify-backup \
  --backup-root /path/to/offsite
PYTHONPATH=src python scripts/check_formal_run_recovery.py restore \
  --backup-root /path/to/offsite --target /path/to/empty-run
```

- `0`: `recovery_controls_ready`
- `2`: `backup_or_recovery_violation`
- `3`: `external_run_not_started`
- `4`: `usage_error`

当前真实 readiness 必须返回 `3`。它不创建真实备份，不解除 `Full1000 未完成`、真实人工 Precision 或官方 scorer/schema 三项正式阻断，且 `formal_validation_complete=false`。
