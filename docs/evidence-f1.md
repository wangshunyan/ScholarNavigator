# 全文 Evidence F1 评测契约

`evidence-f1-v1` 是独立于论文相关性 Recall/F1 的段落级证据指标。它只比较
`FullTextEvidenceDocument` 产生的稳定 `evidence_id`，不参与排序，也不允许通过
标题模糊匹配、AutoScholarQuery gold 或 qrels 推断证据标签。

## 输入

两个 JSONL 文件都使用如下行格式：

```json
{"case_id":"demo_01","paper_id":"arxiv:2301.00001","evidence_ids":["paragraph:<content_sha256>:0"]}
```

`gold` 由独立标注者在已核验全文上填写；`predictions` 由受控全文解析结果导出。
每个 `(case_id, paper_id)` 必须恰好出现一次，证据 ID 不得重复；gold 每行至少要有
一个已标注段落。当前协议采用严格 ID 集合交集：

- micro Precision = 命中段落数 / 预测段落数；
- micro Recall = 命中段落数 / gold 段落数；
- F1 为 Precision 与 Recall 的调和平均；
- 同时报告每个 query-paper 对和 macro F1。

## 运行

```bash
PYTHONPATH=src python scripts/evaluate_evidence_f1.py \
  --gold /secure/evidence-gold.jsonl \
  --predictions /secure/evidence-predictions.jsonl \
  --output outputs/evidence_f1.json
```

没有人工 gold 时，命令返回退出码 `3`，输出 `pending_human_labels` 和空指标，而不是
伪造 0 分。输入闭环固定后，仍需在真实查询集合上完成人工标注，才能把 P1-04 标为
完成；Europe PMC 单篇 smoke 只能证明抓取、许可、定位和哈希链路可用。

所有结果均属于内部工程指标，不是赛事官方 scorer 成绩。
