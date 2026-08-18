# 人工裁决激活门禁

`human_adjudication_activation_v1` 直接消费
`human_annotation_submission_intake_v1` 生成的完整分歧队列，不定义新的标签、
裁决或统计语义。状态按 `queue_ready → issued → acknowledged →
adjudication_submitted → validated → statistics_eligible` 推进；任一上游绑定漂移、
撤销、重复提交或锁后修改均失败关闭。结果接收和统计解锁都会重新验证提交账本仍
处于 `adjudication_queue_ready`，不能依赖签发时的旧状态继续推进。

裁决包仅包含真实分歧项的 query、title、abstract、year、A/B 原始标签、冻结
rubric 和专属 dispute alias。它不包含一致项、全局 item identity、arm、策略、
排名、来源、分数、gold、qrels 或 operator 映射。裁决结果必须完整覆盖且只能
覆盖这些分歧，每项必须提供合法标签和简短 rationale，原始 A/B 判断保持只读。

只有 A/B 各 471 项锁定提交、全部分歧裁决、旧 32 项判断链和绑定证据全部有效时，
门禁才调用既有 `human_precision_adjudication_v1` scorer。输出范围固定为
`human_internal_non_official` 的 change-only、cluster-aware 统计；change-only
盲包不支持绝对 Precision@20。合成演练不会写入真实 readiness、标签目录或
clearance 状态。

```bash
python scripts/check_human_adjudication_activation.py simulate-matrix
python -I -S scripts/check_human_adjudication_activation.py audit-readiness
```

当前真实审计稳定返回 3：真实 A/B/裁决者资格、三方确认、A/B 锁定提交、真实分歧
队列和裁决提交均未提供。Full1000、真实人工 Precision、官方 scorer/schema 三项
正式阻断和 `formal_validation_complete=false` 保持不变。
