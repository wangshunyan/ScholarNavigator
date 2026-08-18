# 实验对照隔离与配对完整性门禁

`experiment_pairing_integrity_v1` 是纯离线的实验设计完整性检查，不计算检索
质量指标，也不替代统计显著性、人工 Precision 或官方 scorer。它直接复用
`run_manifest_v1`、`crash_consistency_v1` 提交代和复现胶囊，不启动新的实验
执行器。

## comparison_plan_v1

每个新成对实验必须在两侧执行前生成同一份 `comparison_plan_v1`。计划绑定：

- query-only 集合的 opaque identity、数量、稳定身份摘要和顺序摘要；
- 数据与 Replay 输入摘要；
- 固定的 baseline/candidate 角色；
- 允许变化的精确叶子 JSON Pointer、两侧预期值；
- Prompt、来源、预算、evaluator、随机种子、并发、超时、重试、约束、规范化
  与执行 profile 组成的共同契约；
- 预声明排除集合与原因。

允许路径只能位于 `/configuration/values/` 的具体叶子。放行父对象、缺失声明
处理、声明之外的配置变化或未知字段均失败。计划文件 SHA-256 同时进入两侧
`run_manifest_v1.comparison` 和 generation-zero `config.comparison`；后续提交代
持续受同一配置摘要约束。因此“预先声明”的可验证边界是运行产物已经绑定计划
哈希，而不是机器时间戳。

## 配对规则

每个 opaque query 必须在 baseline/candidate 中各出现一次。成功、失败、取消和
预声明排除都保留并逐项配对；来源终态也必须一致。单侧缺失、重复提交、失败后
过滤、后验排除、query 重排、非对称 resume 或只分析共同成功项都属于完整性违规。
两侧处于相同合法中间点时返回 `not_ready`，不会通过；处理污染或覆盖不对称始终
返回违规，不能通过删除失败 query 修复。

门禁不读取记录中的结果质量。输入记录若携带 gold、qrels、目标论文或质量指标
字段会被拒绝。报告只含脱敏 query identity、首个差异 JSON path 及两侧规范化摘要，
不含 query 文本或结果正文。

## CLI 与退出码

```bash
PYTHONPATH=src python scripts/check_experiment_pairing.py validate-plan \
  --plan <comparison-plan.json>

PYTHONPATH=src python scripts/check_experiment_pairing.py check \
  --plan <comparison-plan.json> \
  --baseline-manifest <baseline-run-manifest.json> \
  --candidate-manifest <candidate-run-manifest.json>

PYTHONPATH=src python scripts/check_experiment_pairing.py check-fixture
PYTHONPATH=src python scripts/check_experiment_pairing.py audit-registry
PYTHONPATH=src python scripts/check_experiment_pairing.py audit-frozen
```

退出码固定为：`0=passed`、`2=treatment_or_pairing_violation`、
`3=not_ready/not_eligible`、`4=usage_error`。当前 Record160 与 Full1000 为旧格式，
没有预绑定 comparison plan 和完整成对提交记录，只读资格审计如实返回
`not_eligible`，不得反推或补造。

新格式运行的 comparison plan 会登记进复现胶囊文件清单；胶囊仍只执行宿主代码。
当前 evidence registry 没有新式 pairing identity，因此只读审计返回
`not_eligible`，不会改写策略状态或既有结论。

## 与其他验证的边界

- `execution_determinism_v1`：同一输入跨执行方式是否产生相同语义输出；
- 本门禁：两侧是否只有预声明处理变量不同，且 query 覆盖完整对称；
- 显著性工具：在已通过配对完整性的结果上做统计推断；
- 人工 Precision：由盲化人工标签评估精度；
- 官方 scorer：由赛题官方定义成绩口径。

上述证据互不替代，本门禁输出不得描述为 Precision、Recall 或官方成绩。
