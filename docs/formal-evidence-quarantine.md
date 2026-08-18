# 正式评测证据隔离门禁

`formal_evidence_quarantine_v1` 将未来真实人工标签、裁决结果、官方 scorer package 与输出
视为受隔离证据。每次 intake 必须绑定证据类型、协议、comparison plan/run manifest/query 顺序、
文件 SHA-256、允许消费者、禁止用途、生命周期和提交先后关系。历史证据和当前冻结运行不会被
回填 intake manifest。

## 边界

只有 `scholar_agent.evaluation.*`、报告入口和 formal clearance 门禁可以通过登记 reader 消费
证据，且用途只能是 `evaluation`、`reporting` 或 `clearance`。检索、query planning、Prompt、
排序、预算、默认策略、缓存以及前端搜索运行时均禁止导入隔离模块或读取证据目录。运行时 reader
只放行 manifest 登记的单个只读文件，并拒绝复制、写入和路径逃逸；错误只返回原因码和稳定摘要。

静态门禁使用 AST 检查精确 import、文件读取及配置键，不以关键字扫描替代语义边界。合法的现有
`services → evaluation.selection` 依赖不受影响，因为 selection 不读取正式证据。

## 后验污染

intake 必须证明 comparison plan 的预登记提交不晚于执行提交，执行提交不晚于 intake，报告代码
可追溯到 intake 之后。intake 后若检索、规划、Prompt、预算、排序、tie-break、默认策略或数据
身份发生变化，原证据对相应正式声明立即成为 `stale_for_claim`；必须执行新的预登记运行和独立
证据采集，不能用同一证据重新挑选方案。

```bash
PYTHONPATH=src python scripts/check_formal_evidence_quarantine.py verify-boundaries
PYTHONPATH=src python scripts/check_formal_evidence_quarantine.py audit-readiness
PYTHONPATH=src python scripts/check_formal_evidence_quarantine.py intake-dry-run \
  --artifact <synthetic-file> --evidence-root <temporary-root> \
  --evidence-type human_annotation_labels --binding <binding.json> \
  --chronology <chronology.json>
PYTHONPATH=src python scripts/check_formal_evidence_quarantine.py audit-contamination \
  --manifest <intake.json> --evidence-root <quarantine-root> --changes <changes.json>
```

退出码为 `0=quarantine_controls_ready`、`2=leakage_or_posthoc_violation`、
`3=blocked_no_real_formal_evidence`、`4=usage_error`。当前只验证临时合成 evidence 的控制链，
真实审计固定返回 3；Full1000、真实人工 Precision、官方 scorer 三项阻断及
`formal_validation_complete=false` 均不变。

所有 quarantine CLI 命令在处理 intake 前先验证
`formal_validation_preregistration_v1` 的当前 seal。封印缺失、依赖漂移或被篡改时，
即使隔离目录与 manifest 本身合法也会 fail-closed。
