# 公开证据透明日志

`evidence_transparency_log_v1` 为 readiness、standalone 审计包、软件发布候选和未来
clearance receipt 提供确定性追加日志。它不执行检索、不计算质量指标，也不认证发布者身份。

## 记录与 Merkle 规则

每条记录按连续 `sequence` 绑定发布身份、代码提交、声明摘要、三项正式阻断、freshness、
revocation、readiness、standalone、release candidate 和可选 clearance receipt。记录内容哈希
是将 `content_sha256` 置零后的规范 JSON SHA-256；`previous_record_sha256` 形成追加链。

Merkle 树采用域分离：

- 叶：`SHA-256(0x00 || canonical_record)`；
- 内部节点：`SHA-256(0x01 || left || right)`；
- 非二次幂树按小于树大小的最大二次幂切分；
- consistency proof 明确携带旧前缀叶哈希和新增叶哈希，不将排序当作一致性证明。

同一 release identity 不得对应不同内容。历史删除、重排、插入、旧 evidence 回滚、阻断隐藏、
活动撤销继续发布、日志分叉和 receipt 绑定漂移均 fail-closed。合法 supersession 必须保留旧
记录，并引用既有 release identity 与结构化 revocation event 哈希。

## 当前候选 checkpoint

当前 genesis 绑定协议源提交 `f764eb3c0849c53512f9326d7a83429e1c430a7b` 中不可变 Git blob，
避免生成中的 readiness/standalone 文件与日志形成自引用。它只有
`candidate_checkpoint_no_public_release` 状态：发布候选尚未合格，正式验证尚未完成，Full1000、
真实人工 Precision、官方 scorer/schema 三项阻断均保留。

哈希链、inclusion proof 和 consistency proof 只证明内容及前缀历史一致性，不证明发布者身份、
签名权或组织背书。

## 只读命令

```bash
PYTHONPATH=src python scripts/check_evidence_transparency.py verify-log
PYTHONPATH=src python scripts/check_evidence_transparency.py prove-inclusion --sequence 0
PYTHONPATH=src python scripts/check_evidence_transparency.py prove-consistency \
  --old-log benchmark/evidence_transparency_log_v1_log.json \
  --new-log benchmark/evidence_transparency_log_v1_log.json
PYTHONPATH=src python scripts/check_evidence_transparency.py audit-readiness
```

`audit-readiness` 在当前无公开 release checkpoint 时按契约返回退出码 3。退出码 0 只表示透明
控制或证明通过，2 表示日志/发布一致性违规，3 表示没有公开 release checkpoint，4 表示用法
错误。
