# LLM 评审后端能力资格门禁

`judge_backend_qualification_v1` 用脱敏历史调用证据和固定合成 canary 检查后端能否满足
严格结构化输出协议。它只验证协议能力，不判断论文相关性，不生成人工标签、LLM 相关性
标签、Precision/Recall 或官方成绩，也不会改变 evidence registry 或任何默认策略。

## 冻结输入与分析边界

协议位于 `benchmark/judge_backend_qualification_v1_protocol.json`，在真实探针前固定：

- v1/v1.1 protocol、manifest、calls 和 status 的路径及 SHA-256；
- 输入长度分桶、Unicode/Markdown/边界字符特征和并发窗口定义；
- 24 条合成 canary 的内容、顺序与 SHA-256；
- Prompt 版本与文件 SHA-256；
- 每候选调用上限、温度、超时、Token 上限和资格阈值。

历史分析只读取公开盲标元数据和已保存的脱敏调用状态、usage、HTTP 状态及 hash。响应正文
没有保存且不会尝试恢复，历史部分标签也不会读取。按 attempt、调度窗口、输入长度和文本
特征输出的结果只能解释为关联；不能把某个特征声明为 Schema failure、429 或 503 的原因。

## 候选发现与 canary

只有 `probe` 命令调用项目既有 `load_project_env`。候选来自运行时已配置的不同
provider/model 对；当前配置模型只有一个时只产生一个候选，没有替代候选时不会虚构后端。
持久化候选描述仅含 candidate ID、provider、完整 model ID、available/reason 和公开请求参数，
不含 key、完整 URL、endpoint、host、请求头或异常正文。

24 条 canary 全部是合成文本，覆盖短/长文本、空摘要、Unicode、Markdown、边界与控制字符。
它们不复制真实 query、论文、gold、qrels、case ID 或目标标题。每个候选按同一固定顺序执行
一次单项逻辑调用，最多 24 次；并发固定为 1，不允许追加临时 attempt。成功响应必须为唯一
顶层 `labels` 对象、唯一 item、精确键集合、合法枚举、原样 opaque item binding 和固定合成
evidence token。客户端使用 Pydantic `extra=forbid` 再校验，malformed 响应不能修复、猜测或
补造。调用文件只保存输入/响应 hash、脱敏失败分类、原生模式诊断和供应商明确返回的 usage，
不保存响应或合成标签。

## 资格门槛

候选必须 24/24 同时满足：Schema、item binding、原生 `structured_json`、恰好一次供应商
HTTP attempt、无 compatibility fallback、完整的 prompt/completion/total token usage，且无
provider failure。任何失败都使该候选不合格；不能删除 canary、只补失败项或追加 attempt。
供应商费用没有权威字段时固定为 `not_available`，不会估算。

通过者只生成后续全量运行的只读建议 manifest，绑定模型、Prompt、协议、并发、attempt 和
预算；门禁本身绝不启动 471 项评审。未通过或没有候选时不生成建议。

## CLI 与退出码

```bash
PYTHONPATH=src python scripts/check_judge_backend_qualification.py analyze-frozen
PYTHONPATH=src python scripts/check_judge_backend_qualification.py probe
PYTHONPATH=src python scripts/check_judge_backend_qualification.py qualify
PYTHONPATH=src python scripts/check_judge_backend_qualification.py verify
```

`analyze-frozen`、`qualify` 和 `verify` 完全离线；仅 `probe` 可调用配置的 LLM API。
四个命令都禁止学术 API、其他网络和 Snapshot 写入。退出码固定为：

- `0=qualified`
- `2=integrity_or_conformance_violation`
- `3=no_qualified_backend/not_eligible`
- `4=usage_error`

运行目录支持 hash 锁定的安全 resume：已经落盘的 canary 不会重复调用；protocol、Prompt、
canary、候选或请求参数漂移会拒绝混用。发布结果中的 `labels_file` 和 `statistics_file` 均为
`null`，并通过 manifest 对所有证据文件做大小与 SHA-256 校验。

## 结果解释

资格通过仅说明该后端在这组预注册合成输入上满足严格协议。它不能证明模型相关性判断正确，
不能代替双人盲标和裁决，也不等于人工 Precision 或官方 scorer。若后续需要运行完整评审，
必须以新任务显式批准建议 manifest，并继续遵守冻结覆盖门槛、盲化和不可选择性补采规则。

## 本次资格终态

运行时发现一个候选：`openai_compatible` / `deepseek-ai/deepseek-v4-flash`。固定 24 条
canary 全部按顺序执行：18 条为 provider failure（11 次 timeout、7 次 HTTP 503），6 条
取得原生 `structured_json` 响应并全部通过严格 Schema 与 item binding；其中 3 条恰好一次
HTTP attempt，另外 3 条发生 2/3 次既有 transient attempt，因此严格成功仅 3/24
（12.5%），provider/Schema 成功均为 6/24（25%）。候选未通过资格门槛。

供应商只为 6 条成功响应报告 usage：prompt 1,719、completion 438、total 2,157 tokens；
其余 18 条 usage 不可用，费用字段也为 `not_available`。没有 fallback、标签、相关性统计或
后续全量运行建议。发布 manifest 绑定的 `calls.jsonl`、`frozen_analysis.json`、
`qualification.json` SHA-256 分别为
`ba9ff65fd830b01c3fbd11dba4a7ec2b985c633e8d20c54a5a30e92eaa104d38`、
`dfb542b20a6fb7c6e8641bf83dc882fcedb7de814fcb6876b4f0be4f354615a6`、
`73429588771ad3c9714e76853903a65b8e3e249841d613fb33a83f6cae1c7ac6`。
