# 正式验证阻断解除门禁

`formal_validation_clearance_v1` 是三项外部阻断的 fail-closed 状态机。它不运行检索、人工
标注或官方 scorer，也不把文档措辞或手工布尔值当作解除证据。状态为 `blocked`、
`partially_satisfied`、`eligible_for_clearance`、`cleared`、`invalid`。

## 严格解除谓词

- **Full1000**：必须绑定既有不可变计划，1000/1000 查询在权威 run manifest、原子提交代、
  逐操作资源账本和 aggregate 中一致且身份唯一。Record160/162、partial、legacy 和 fake
  dry-run 都不能满足。
- **人工 Precision**：冻结 471 项必须由两位不同匿名人工标注者完整独立覆盖，所有分歧合法
  裁决，既有 adjudication gate 为 `validated`，且 `synthetic_only=false`、来源为 human。
  空标签、合成标签和 LLM 标签不能满足。
- **官方 scorer**：官方 package 名称、版本、哈希、输入/输出 Schema 和指标命名空间都必须
  已提供；完整 1000 输入须通过隔离沙箱并产生严格验证输出。合成 scorer 不能满足。

三项同时满足后，还必须证明 freshness 无 stale、依赖闭合、实现基线与当前 HEAD 兼容、
`current_rules` 仍是唯一默认策略且 `deterministic_tiebreak_v2` 仍关闭，才进入
`eligible_for_clearance`。

## Receipt

只有 `eligible_for_clearance` 可生成 `clearance_receipt_v1`。receipt 绑定三项 blocker evidence
摘要、整体 evidence/state 摘要、协议提交和固定验证命令，使用 canonical JSON SHA-256 自校验；
不使用或保存私钥。目标文件必须不存在，重复签发会失败。测试中的合成 receipt 始终带
`synthetic_test_only=true`，不能令 `formal_validation_complete` 变为 true，也不会进入真实
readiness 或 evidence registry。

```bash
PYTHONPATH=src python scripts/check_formal_validation_clearance.py audit-current
PYTHONPATH=src python scripts/check_formal_validation_clearance.py evaluate --evidence <evidence.json>
PYTHONPATH=src python scripts/check_formal_validation_clearance.py issue-receipt --evidence <evidence.json> --receipt <receipt.json>
PYTHONPATH=src python scripts/check_formal_validation_clearance.py verify-receipt --evidence <evidence.json> --receipt <receipt.json>
```

退出码：`0=state_valid_or_cleared`、`2=evidence_or_transition_violation`、
`3=blocked_missing_external_evidence`、`4=usage_error`。当前真实审计固定返回 3，并继续保留
Full1000 未完成、真实人工 Precision 缺失和官方 scorer/schema 缺失三项阻断。
此外，权威撤销账本中的任一活动事件会先于 receipt 状态机阻断 audit、evaluate、issue 和
verify；只有经过 freshness 门禁的新证据完成 supersede/restored 链后才可恢复资格。

clearance CLI 在评估或签发 receipt 前还会验证
`formal_validation_preregistration_v1` 的当前 seal；预注册缺失、依赖漂移或后验语义修改
均阻断后续清除流程。预注册封印本身不满足三项外部证据谓词。
