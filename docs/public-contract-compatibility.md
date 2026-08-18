# 公共评测接口兼容性门禁

`public_contract_compatibility_v1` 冻结当前可复核的 FastAPI OpenAPI、前端消费类型、关键离线
CLI 和机器产物契约。它只治理工程兼容性，不定义官方 Schema、质量指标或官方成绩。

## 覆盖与归一化

- OpenAPI 路径、方法、参数、响应递归 Schema 和状态码；描述文字等非语义展示字段不进入摘要。
- `frontend/src/types/api.ts` 的逐字段类型、required/optional、nullable，以及 request producer / response consumer 的方向兼容性。FastAPI 默认输出字段以真实 `jsonable_encoder` fixture 证明会稳定出现。
- run provenance、readiness、formal clearance、人工标注交付和外部 scorer handoff CLI 的命令与参数；每个 CLI 还运行固定安全探针，冻结真实退出码、stderr 边界及递归机器 JSON Schema。
- `run_manifest_v1` 及 readiness、clearance、人工标注和 scorer handoff 机器 JSON 的递归结构、null/missing 区别、严格未知字段策略和版本。

基线不复制 query、论文正文、私有映射或凭据，也排除绝对路径、时间戳和描述文本顺序。

## 兼容性语义

- 删除或重命名字段、类型/可空性变化、枚举收窄、退出码变化、顺序语义变化均为 `breaking`。
- 对 `additionalProperties=false` 或其他严格消费者新增字段始终是 `breaking`，CLI 不提供全局放宽参数。
- 只有基线在精确 JSON Pointer 登记允许扩展、节点允许未知字段且新增字段确实 optional 时，才是 `additive_review_required`；它仍不能自动通过当前基线。
- 同一协议版本的 breaking 漂移直接失败。`supported_read_versions` 与 migration registry 明确登记旧读能力；版本升级缺少旧读验证或已知迁移器时失败，不能只刷新基线。
- JSON 输入拒绝重复键、NaN/Infinity 和非法 UTF-8。文档命令仅作为命令引用清单；本门禁不声称已验证文档中的示例输出，机器输出契约来自可执行 CLI 探针。

## 使用

```bash
PYTHONPATH=src python scripts/check_public_contract_compatibility.py snapshot
PYTHONPATH=src python scripts/check_public_contract_compatibility.py verify-current
PYTHONPATH=src python scripts/check_public_contract_compatibility.py compare \
  --from old.json --to new.json
PYTHONPATH=src python scripts/check_public_contract_compatibility.py audit-readiness
```

退出码固定为：`0=contracts_compatible`、`2=breaking_or_versioning_violation`、
`3=not_ready_missing_contract_baseline`、`4=usage_error`。输出始终为规范 JSON；门禁不联网、不调用
LLM、不写 Snapshot，也不读取 gold/qrels 或计算质量指标。
