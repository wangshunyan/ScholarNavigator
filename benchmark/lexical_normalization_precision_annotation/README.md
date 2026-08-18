# lexical_normalization_v1 盲化 Precision 标注包

本包含 200 个随机化 query-paper 样本。标注者只能查看 `public/blind_samples.jsonl` 与标注 Schema，不得查看 `private/`。

两位标注者应分别填写 `annotator_1.jsonl` 与 `annotator_2.jsonl`；完成前不得交流。仅对分歧项填写 `adjudication.jsonl`。完成后使用同一 CLI 的 `score` 子命令计算一致性和人工指标。

当前模板没有人工标签，因此不包含 Precision、误放率或相关性结论。
