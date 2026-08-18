# 正式验证全链路合成彩排

`formal_validation_dress_rehearsal_v1` 只验证既有离线门禁能否按正式顺序组合，不生成检索质量结论、人工 Precision 或官方成绩。

## 合成边界

- 所有运行身份以 `synthetic_rehearsal_only:` 开头；运行目录使用独立临时命名空间并在命令结束时清理。
- Full1000 阶段复用现有 1000-query、20-shard fake-adapter 执行、checkpoint/resume、单 shard 替代、资源账本、原始响应取证、aggregate、Top-20、备份恢复和复现胶囊路径。
- 人工阶段复用 471 项双人独立提交和裁决门禁；scorer 阶段复用严格合成 package、完整 1000-query canonical handoff 和隔离子进程。
- 标签、scorer 指标值、test-only receipt、运行目录和审计归档均不持久化，也不进入真实 evidence registry、readiness、正式账本或 clearance 状态。

## 顺序与失败语义

彩排固定执行预注册封印、启动授权、全量执行、取证与恢复、证据 intake、解盲/评分、隔离、freshness、合成 clearance、test-only receipt 和 standalone 审计包验证。阶段缺失或乱序、跨提交/协议混用、重复 intake/receipt、旧 attempt、部分标签/scorer 输出、证据后修改 Prompt/阈值/样本/统计/默认策略及上游撤销均必须失败。

```bash
PYTHONPATH=src python scripts/check_formal_validation_dress_rehearsal.py run
PYTHONPATH=src python scripts/check_formal_validation_dress_rehearsal.py verify
PYTHONPATH=src python scripts/check_formal_validation_dress_rehearsal.py simulate-failure
PYTHONPATH=src python scripts/check_formal_validation_dress_rehearsal.py audit-readiness
```

前三个命令在合成闭环完整时返回 `0`。`audit-readiness` 在真实外部证据仍缺失时必须返回 `3`。退出码 `2` 表示集成或隔离违规，`4` 表示用法错误。

当前合成彩排不能解除 Full1000 未完成、真实人工 Precision 缺失和官方 scorer/schema 缺失三项正式阻断；`formal_validation_complete` 始终为 `false`。
