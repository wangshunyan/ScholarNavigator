# Human annotation submission intake v1

`human_annotation_submission_intake_v1` 将现有
`human_annotation_delivery_v1` 锁定导出绑定到已确认的
`human_annotation_assignment_activation_v1` 签发链。它不定义新标签格式，不生成标签、
Precision、agreement、κ 或裁决结果。

## 状态与权威输入

状态机为：

`awaiting_submissions → one_submission_validated → two_submissions_validated
→ adjudication_queue_ready`

任何有效状态都可进入 `revoked` 或 `invalid`。追加事件绑定 assignment receipt、
opaque principal、角色、签发包哈希、导出字节哈希、锁定标签摘要、协议和代码提交。
只有三角色 assignment 已达到 `locked_for_submission`，且 A/B 两侧均完整通过 471 项
校验时，才允许生成裁决队列。

接收器直接调用既有 delivery loader，验证原有标签枚举、notes 边界、package identity、
alias 集合、锁定摘要和完整覆盖。草稿、部分覆盖、重复或未知 alias、A/B 包互换、
协调者代交、锁后修改、旧提交或撤销 assignment 都会 fail closed。任何一侧缺失时，
不得解盲、比较、统计或保留“共同完成项”。

## 裁决队列边界

两侧验证完成后，operator-only mapping 只在受控转换中使用。裁决者队列仅包含：

- 新生成的 `disagreement_alias`；
- query、title、abstract、year；
- A/B 原始标签及受限 notes；
- 冻结 rubric。

队列不包含 A/B alias、全局 opaque identity、arm、策略、排名、来源、分数、
gold/qrels 或 operator mapping。队列必须包含全部且仅包含真实分歧；原始 A/B 判断
保持不可变，无分歧项不能伪造裁决。私有 queue mapping 单独输出，不能交给裁决者。

## CLI

```bash
PYTHONPATH=src python scripts/check_human_annotation_submission.py audit-readiness
PYTHONPATH=src python scripts/check_human_annotation_submission.py simulate-matrix
```

其余命令为 `verify-submission`、`import-dry-run` 和
`build-adjudication-queue`，必须显式提供 assignment ledger、三份签发包及三份
assignment receipt。退出码固定为：

- `0`: `submission_chain_ready`
- `2`: `submission_or_blinding_violation`
- `3`: `not_ready_missing_real_submissions`
- `4`: `usage_error`

当前真实审计返回 3：真实 A/B/adjudicator 资格、三方签发确认以及 A/B 锁定提交均
缺失。合成矩阵只验证链路和攻击边界，产物自动清理，不进入真实 readiness。
