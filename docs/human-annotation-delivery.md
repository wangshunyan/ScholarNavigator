# 双人盲化人工标注交付

`human_annotation_delivery_v1` 把既有人工 Precision 门禁绑定的 439 个当前包条目和 32 个旧包引用条目，确定性地合并为 471 项交付总体。它不创建标签，也不计算 Precision、agreement 或官方成绩。

## 交付边界

- annotator A/B 均看到完整 471 项，但顺序和 `item-…` alias 使用不同固定种子；两侧 alias 不相交。
- 标注者文件只包含 alias、query、title、abstract、year 和冻结 rubric。全局 opaque ID、策略、方向、排名、来源、分数、gold/qrels、case ID 与新旧包归属只存在于 operator-only 映射。
- 静态页面不加载外部资源，使用 `textContent` 展示元数据；进度仅保存在本地浏览器。完成覆盖后可锁定并导出，锁定摘要不匹配、公式前缀 notes、遗漏、重复、跨包 alias 或非法标签都会被拒绝。
- 回收器把 alias 严格恢复成 `human_precision_adjudication_v1` 所需的当前 439 项和旧包 32 项身份。只有真实双人覆盖及所有分歧裁决完成后，既有门禁才允许统计。

## 离线命令

```bash
PYTHONPATH=src python scripts/check_human_annotation_delivery.py verify-package
PYTHONPATH=src python scripts/check_human_annotation_delivery.py dry-run
PYTHONPATH=src python scripts/check_human_annotation_delivery.py audit-readiness
```

`dry-run` 的合成标签只存在于临时目录，验证完回收与裁决状态机后立即丢弃；报告中的 `statistics` 始终为 `null`。真实 readiness 当前固定返回退出码 3 和 `blocked_awaiting_real_annotators`。

退出码：0 表示交付链路就绪，2 表示包或回收完整性违规，3 表示仍等待真实双人标注，4 表示用法错误。
