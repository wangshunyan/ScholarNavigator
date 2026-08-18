# 来源原始响应取证与解析重放

`provider_ingest_provenance_v1` 是新格式联网运行的可选观察性契约。它不改变查询、来源、
预算、解析器、排序或默认策略，也不计算检索质量。正式 capture 将每次 HTTP attempt 的精确
响应字节保存到内容寻址的只读归档，并把脱敏 envelope 与同一提交代的
`resource_ledger_v1`、事件、checkpoint generation 和 `run_manifest_v1` 绑定。

## 安全边界

- JSON envelope 仅保存 opaque run/query/attempt/operation identity、状态码、媒体类型、
  encoding/compression、字节长度和 SHA-256、分页游标摘要、parser 版本及数量终态。
- 不保存请求 URL、query 参数、请求头、凭据、`.env`、用户名或绝对路径。
- 原始响应始终是不可信数据，只能作为 parser 输入；归档成员不会执行，也不会进入 Prompt。
- bodyless 的 429、503、timeout、connection failure 和 adapter exception 使用明确终态，
  缺失 encoding、成本或供应商信息保持 `not_available`，不以 0 或猜测替代。

## 守恒与重放

门禁对每个 envelope 复核原始字节长度和 SHA-256，并调用四个生产 connector 当前使用的
record parser。重放必须保持 accepted record 的顺序、来源身份和规范输出摘要。每次请求满足
`parsed = accepted + rejected`；未知顶层 schema、非法记录和缺失身份具有明确 reason code。
分页按照 endpoint/parser 隔离，拒绝断链、游标循环、重复 envelope、跨 attempt 混用和未登记
资源操作。`run_manifest_v1` 仅接受同时登记 JSON bundle、raw tar 和权威资源账本的新运行；两项
取证文件也进入原子 completion generation 与复现胶囊。

## CLI

```bash
PYTHONPATH=src python scripts/check_provider_ingest_provenance.py check-fixtures
PYTHONPATH=src python scripts/check_provider_ingest_provenance.py verify \
  --bundle <provider_ingest_provenance.json> \
  --raw-archive <provider_ingest_raw.tar> \
  --resource-ledger <resource_ledger.json>
PYTHONPATH=src python scripts/check_provider_ingest_provenance.py replay-parser \
  --bundle <provider_ingest_provenance.json> \
  --raw-archive <provider_ingest_raw.tar>
PYTHONPATH=src python scripts/check_provider_ingest_provenance.py audit-frozen
```

退出码为 `0=passed`、`2=provenance_or_parser_violation`、
`3=not_eligible/missing_raw_response`、`4=usage_error`。合成矩阵不访问网络、不写 Snapshot，
其论文元数据仅为临时 fixture，不进入正式证据或质量统计。

## Full1000 与历史边界

`benchmark/full1000_provider_ingest_capture_addendum_v1.json` 只读绑定现有 Full1000 执行计划
文件与内嵌计划摘要，并要求未来全新运行启用本协议；它不改写原计划。Record160/162 没有
parser 前原始响应字节，因此资格固定为 `not_eligible`，不能从 Snapshot 反推历史 payload 或
静默解析损失。Full1000 未完成、真实人工 Precision 缺失和官方 scorer/schema 缺失三项正式
阻断不变，`formal_validation_complete=false`。
