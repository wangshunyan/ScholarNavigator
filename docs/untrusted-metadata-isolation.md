# 非可信学术元数据隔离门禁

`untrusted_metadata_isolation_v1` 是离线安全正确性门禁，不是检索质量指标、人工
Precision 或官方 scorer。论文标题、摘要、作者、venue、URL 和来源错误文本都来自外部
来源，始终按数据处理；它们不能取得 system/developer/tool 角色、触发工具、读取环境，
也不能改变查询、来源、预算、排序或下一条 query 的状态。

## 信任边界

- `Paper` 中的权威原值、统一身份和去重输入不被改写。
- relevance judgement 仅在 LLM 边界派生带
  `untrusted_academic_metadata_v1` / `untrusted_data` 标记的单个 user JSON envelope。
  消息必须恰好为 `system,user`，且每条仅有 `role/content`；元数据不能增加消息或工具参数。
- LLM JSON 响应只允许既有 judgement 字段。未知根字段或嵌套字段使整批回退到现有规则，
  不执行其中的工具请求。
- 公共 API 链接仅允许无凭据的 HTTP(S) URL；旧客户端仍在渲染前复核 scheme。
  Markdown 导出转义结构字符与控制/双向字符。
- 来源错误仅保留安全诊断码；包含伪指令、敏感资源名、绝对路径或控制内容时只公开原文
  SHA-256 的短身份，不回显原文。

## 规范化和血缘

预注册规则位于
`benchmark/untrusted_metadata_isolation_v1_protocol.json`。文本使用 Unicode NFKC；换行和
Tab 转为空格；C0/C1 与双向控制字符变成可见 Unicode escape；title/abstract/author/
venue/error/URL 分别执行字段级上限。截断、拒绝和转义都写入可选 hash-only 观察记录，
不保存恶意原文。

新格式运行可把 `untrusted_metadata_isolation.jsonl` 以
`untrusted_metadata_isolation_v1` 角色登记到 `run_manifest_v1`。字段血缘文档可选携带同一
观察摘要；复现胶囊按现有规则收录 manifest 已登记输出。未启用观察器时，结果、排序、
去重、事件和旧序列化保持原语义。

## CLI 与退出码

```bash
PYTHONPATH=src python scripts/check_untrusted_metadata_isolation.py check-fixture
PYTHONPATH=src python scripts/check_untrusted_metadata_isolation.py \
  check-fixture --fault role_escape
PYTHONPATH=src python scripts/check_untrusted_metadata_isolation.py \
  check-fixture --fault cross_query_pollution
PYTHONPATH=src python scripts/check_untrusted_metadata_isolation.py audit-frozen
```

- `0`: passed
- `2`: isolation_or_injection_violation
- `3`: not_eligible / unsupported_path
- `4`: usage_error

完整 fixture 使用本地 fake LLM，只检查生产消息构造、严格解析、API 映射、去重和字段血缘
路径；真实 LLM、网络、工具、Snapshot 写入和质量指标均为 0。违规报告只含脱敏身份、JSON
路径与两侧规范化摘要。历史 Record160/Full1000 没有本协议要求的原始来源字段、消息边界
及隔离观察记录，因此只读资格固定为 `not_eligible`，不得反推其历史抗注入能力。
