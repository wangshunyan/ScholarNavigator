# 正式阻断外部输入行动包

`formal_external_blocker_action_pack_v1` 将仍待外部提供的三类输入收敛为单一、
确定性的离线交接包。它复用现有备份成员、人工标注和官方 scorer 接收链，
不增加新的资格、身份、标注、评分或统计实现。

三条链的当前机器状态都是
`engineering_ready_external_input_missing`：既有 CLI 与协议可达且新鲜，真实只读
审计保持阻断。行动包分别列出缺失输入、操作者命令、输入契约、预期退出码、
成功判据、失败回滚和禁止事项。路径占位符必须由操作者在私有离线环境中填写，
行动包本身不含真实路径、身份映射、凭据、查询或论文内容，也不猜测赛事方定义。

## 开发冻结

三条链同时处于上述状态时，总状态为 `external_action_required`。在收到真实外部
输入，或机器证据定位到具体工程缺陷之前，不应继续生成新的治理或准备任务。
这项冻结不改变三个正式阻断，也不把工程链路就绪表述为正式验证完成。

## 只读命令

```bash
python scripts/check_external_blocker_actions.py verify-pack
python scripts/check_external_blocker_actions.py audit-chains
python scripts/check_external_blocker_actions.py audit-readiness
```

前两项在契约闭合时返回 `0`；当前真实 `audit-readiness` 固定返回 `3`，直到真实
备份成员、完整真实人工证据和经验证的官方 scorer 材料分别进入既有接收链。
