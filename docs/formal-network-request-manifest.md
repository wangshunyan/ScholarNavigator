# Full1000 正式网络请求清单

`formal_network_request_manifest_v1` 在不访问网络的前提下，通过生产
query planning、query adapter、四源 connector 请求描述和现有缓存/Snapshot identity
路径，冻结未来 Full1000 执行的请求意图。它不发起请求、不写 Snapshot，也不表示任何
query 已完成。

## 契约和闭合边界

协议绑定固定提交、Full1000 execution plan、1000 条 opaque query identity 及顺序、
20 个 shard、生产 planning 结果、四源适配版本、请求预算和 retry 上限。当前清单闭合为：

- 1000 条 query，按稳定 ordinal 唯一分配到 20 个 shard；
- 2410 个预先冻结的 subquery；
- 每源 2410 个逻辑请求槽，共 9640 个；
- 19280 次理论 HTTP attempt 上限；
- 9600 个唯一 Snapshot/cache identity，40 个语义相同请求由多个 intent 引用。

每个 intent 只保存 query/subquery/request 的 opaque identity、规范化参数名和参数值
SHA-256、endpoint alias、method、timeout、retry、认证范围别名、content negotiation、
adapter/version、缓存键摘要与 Snapshot key。原始 query、URL、请求头、凭据、环境值和
绝对路径均不进入产物。

## 分页和缓存身份

初始请求可完全物化。只有 PubMed 当前生产路径包含依赖响应 `idlist` 的条件式 efetch；
清单仅登记父字段、参数生成规则、page/retry 预算及父请求身份，不构造 PMID、cursor 或
后续 URL。缓存门禁拒绝语义不同请求共用同一 key、同一请求出现多个 key、参数顺序漂移、
隐式默认参数变化及未登记的认证语义。

四源 connector 的 `ConnectorRequestSpec` 是生产 connector 与离线清单共享的观察性请求
描述；生产调用仍使用同一参数，不存在平行请求生成器。清单不会读取配置密钥，认证只以
固定 scope alias 表示。

## 历史 Snapshot 对照

现有 Record160/162 的 1093 个冻结 Snapshot key 只读对照结果为：816 个与新计划精确
匹配、277 个历史孤立 key、8784 个新计划唯一 key 尚无历史覆盖，冲突为 0。所有历史 key
都标记为 `historical_reference_only`，不能充当 Full1000 checkpoint、已完成 query 或
正式运行证据。

## 命令和退出码

```bash
PYTHONPATH=src python scripts/check_formal_network_request_manifest.py build \
  --output <empty-output-directory>
PYTHONPATH=src python scripts/check_formal_network_request_manifest.py verify \
  --bundle <output-directory>
PYTHONPATH=src python scripts/check_formal_network_request_manifest.py audit-snapshots
PYTHONPATH=src python scripts/check_formal_network_request_manifest.py audit-readiness
```

退出码为：

- `0=request_manifest_ready_network_blocked`
- `2=request_identity_or_plan_violation`
- `3=not_ready_missing_request_metadata`
- `4=usage_error`

launch addendum 要求未来授权前在目标提交重新生成并得到相同清单摘要；随后每个真实 attempt
还必须由 `provider_ingest_provenance_v1` 记录响应 envelope。请求清单只证明待采集输入
闭合，不证明网络可用、来源产出、相关性、Precision/Recall、Full1000 完成或官方成绩。
