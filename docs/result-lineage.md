# 字段级结果血缘门禁

`result_lineage_v1` 为 connector 已映射的统一 `Paper`、生产去重簇和最终论文字段建立
离线可重建证据。它回答“某个字段来自哪条来源记录、为何采用或拒绝”，不同于
`run_manifest_v1` 的运行文件谱系，也不计算 Precision、Recall、F1 或官方成绩。

## 契约与观察点

生产 `deduplicate_papers_with_audit` 保持原有首见顺序、统一身份判断和 `_merge_papers`
选择规则，只增加默认关闭的 `lineage_sink`。`SearchService.run_search` 的可选
`result_lineage_callback` 在最终化后读取相同 raw candidates，并只选择实际返回论文的
既有血缘；不开启时没有额外字段、事件或持久化行为。

血缘中的 query 仅保存 `query:<sha256>` opaque identity。每条来源记录是 adapter 已映射
且通过 `Paper` Schema 的最小字段集合，其内容、来源、顺序和 SHA-256 被登记；不会保存
HTTP 请求头、认证信息或完整原始响应。每个最终结果记录：

- 全部来源记录引用、来源终态和统一身份合并依据；
- title、authors、year、venue、abstract、六类稳定标识、URL、sources 和 citation count
  的全部候选值、`null/empty/value` 状态、采用值和拒绝理由；
- `unified_identity_v1` 规范化和 `paper_field_merge_v1` 字段选择版本；
- 最终结果稳定身份、顺序和规范输出摘要；
- 同类稳定标识冲突造成的跨簇分离，以及部分来源失败但已有记录可贡献的状态。

当前字段选择完全复用既有规则：非占位/较长 title，作者数更多者，首个非空 year、venue、
标识和 URL，较长 abstract，最大 citation count，以及 sources 的首见稳定去重。规则并未
新增质量判断；冲突只被解释，不会改变既有采用值。

## 重建 invariant

独立验证器从已登记的统一来源记录再次调用同一生产 dedup 路径，并逐字段比较完整契约。
它拒绝来源 hash 漂移、补造字段、跨簇借值、错误 record ref、未登记规范化步骤、字段
状态混同、身份簇或结果顺序变化。失败仅输出 opaque query/result identity、字段、首个
差异路径及两侧摘要，不回显原始敏感内容。

新格式运行如持久化血缘，文件名固定为 `result_lineage.jsonl`，角色固定为
`result_lineage_v1`；该名称已进入原子提交产物白名单，并且必须像其他输出一样登记到
`run_manifest_v1.outputs`，接受大小、记录数和 SHA-256 校验。旧 manifest、旧提交代和
冻结产物不作改写。

## CLI 与退出码

```bash
PYTHONPATH=src python scripts/check_result_lineage.py check
PYTHONPATH=src python scripts/check_result_lineage.py check --fault field_injection
PYTHONPATH=src python scripts/check_result_lineage.py audit-frozen
PYTHONPATH=src pytest -q -m result_lineage_regression
```

| 退出码 | 含义 |
| ---: | --- |
| `0` | fixture 的来源、合并、字段重建与观察性等价全部通过 |
| `2` | 血缘、来源引用或重建 invariant 违规 |
| `3` | 产物缺少字段级候选/合并证据，不具备资格 |
| `4` | CLI、协议或离线输入使用错误 |

`--fault field_injection` 只在内存副本中注入不可追溯 title，用于稳定证明退出码 2；不会
修改 Snapshot 或 fixture。AutoScholarQuery Record160 和 Full1000 旧产物只有结果与
运行级元数据，没有字段候选和合并决策，因此只读资格检查固定返回 `not_eligible`，不得
反向推断或补造其血缘。

## 门禁边界

- `run_manifest_v1`：静态文件身份、配置绑定和运行父子谱系；
- `result_lineage_v1`：最终论文每个字段的来源证据和去重选择；
- `execution_determinism_v1`：同一 Replay 的跨执行方式一致性；
- `crash_consistency_v1`：写入代际、原子提交和恢复；
- `reproduction_capsule_v1`：自包含、跨目录离线复放；
- 内部 Benchmark/官方 scorer：检索质量评测。

这些门禁互补，任何一个通过都不能替代其他门禁或质量结论。
