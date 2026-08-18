# Record160 盲化 LLM 相关性代理评审

`llm_relevance_judging_v1` 是冻结 Top-20 变更盲标包的内部 LLM proxy。
它不是人工 Precision、不是官方 scorer，也不会改变 `lexical_normalization_v1`
或任何默认开关。输入固定为当前包 439 项和既有包中被引用的 32 项，共 471
个 query-paper 变更关系对应的唯一可见条目。

## 盲化与协议

`benchmark/llm_relevance_judging_v1_protocol.json` 绑定两个公开盲标文件、私有映射、
冻结 rubric、Record160 case 诊断、查询连通分量和四个 prompt 文件的 SHA-256。
`prepare` 将公开 `sample_id` 替换为不可逆的 `item:<sha256>`，评审视图严格只有：

- `item_id`
- `query`
- `title`
- `abstract`
- `year`

策略/arm、换入换出方向、来源、排名、分数、gold/qrels、case/query ID、目标论文
和 query→gold 关系不进入评审消息。论文元数据作为 `untrusted_data` JSON envelope
传输，消息角色固定为 system/user；响应使用禁止未知字段的 Pydantic Schema。

## 双评、裁决与锁定

两轮独立评审均固定温度 0、每批 8 项、最多 4 个隔离客户端并发、每批最多两个有界逻辑
attempt。并发完成顺序不参与 batch identity 或落盘路径。每项标签只能为
`relevant`、`partially_relevant`、`not_relevant` 或
`insufficient_information`，证据说明最多 240 字符且不保存思维链。只有标签分歧项进入
第三轮匿名裁决；原始两轮结果保持不变。每个成功批次以输入/响应 hash 锁定，resume
跳过已锁定批次，不重复计费。失败、Schema 错误和遗漏 item 均以 opaque identity 明示，
不得当作 `not_relevant`。

所有批次闭合后才生成 `labels_lock.json` 并允许 `score` 读取私有 arm 映射。发布目录与
人工标签目录隔离；`calls.jsonl` 只含调用 hash、attempt、供应商明确返回的 usage、
诊断和脱敏失败码，不保存 Prompt/响应原文或凭据。供应商没有返回 usage/cost 时记录
`not_available`，绝不以 0 或估算值代替。

## 统计边界

正类固定为 `relevant + partially_relevant`。评分复用既有 change-only 配对统计与冻结
查询连通分量，使用 20,000 次固定种子的 query bootstrap 和 cluster bootstrap/sign-flip。
`insufficient_information` 不作为负例；涉及它的 query 从完整配对估计中显式排除并报告。
由于盲包只覆盖 Top-20 换入/换出，不含双方共有项，所以绝对 arm Precision@20 必须为
`null`；可报告的是变更项 Precision proxy 和每 query 的 Top-20 正类贡献差值。

## CLI 与退出码

```bash
PYTHONPATH=src python scripts/run_llm_relevance_judging.py prepare
PYTHONPATH=src python scripts/run_llm_relevance_judging.py judge --round independent_1
PYTHONPATH=src python scripts/run_llm_relevance_judging.py judge --round independent_2
PYTHONPATH=src python scripts/run_llm_relevance_judging.py adjudicate
PYTHONPATH=src python scripts/run_llm_relevance_judging.py verify
PYTHONPATH=src python scripts/run_llm_relevance_judging.py score \
  --publish-dir benchmark/llm_relevance_judging_v1_baseline
```

以上 CLI 当前默认使用加固协议 `llm_relevance_judging_v1_1` 和独立目录
`outputs/benchmark_runs/llm_relevance_judging_v1_1_record160`。如需只读核验历史 v1，必须
同时显式传入 v1 protocol 与历史 run 目录；两个 contract、opaque item ID、Prompt hash、
runtime binding 和 batch lock 不能混用。

只有 `judge`/`adjudicate` 通过项目既有 loader 加载运行配置，并只调用配置的 LLM API；
命令不输出 provider URL、密钥或认证头。所有阶段禁止学术 API、其他网络和 Snapshot 写入。
退出码固定为 `0=completed`、`2=integrity_or_schema_violation`、
`3=incomplete_or_llm_unavailable`、`4=usage_error`。

若任一批次在两个有界 attempt 后仍失败，必须停止后续解盲和统计。此时可用
`verify --publish-incomplete-dir <dir>` 发布仅含 opaque coverage、调用/响应 hash、usage
和脱敏失败码的审计证据；该目录不含部分标签、arm mapping 或指标。

测试只使用 fake LLM：

```bash
PYTHONPATH=src pytest -q -m llm_relevance_judging_regression
```

## 本次 Record160 运行终态

在模型 `deepseek-ai/deepseek-v4-flash`、judge/adjudicator prompt `1.0.0` 下，第一轮
59 个批次中 56 个通过 Schema，锁定 447/471 项；其余 3 个批次、24 项在各自两个
逻辑 attempt 后仍为 `schema_failure`。因此第二轮、裁决、解盲和统计均未启动，所有
Precision proxy、配对差值、CI 和 kappa 都不存在，不能从部分标签推断。

本轮共记录 65 个逻辑调用、73 个 provider HTTP attempt。供应商明确返回
223,458 prompt tokens、44,034 completion tokens、267,492 total tokens；供应商费用字段
不可用，保持 `not_available`。不完整证据位于
`benchmark/llm_relevance_judging_v1_record160_incomplete/`，其中 `calls.jsonl` 与
`status.json` 的 SHA-256 分别为
`b2538101f73dbb7e78ab6307d625781df4ff53f528dfffe182c1d3bc808467f6` 和
`468696d91612a0bb20a19b3e21aa1431ac176ef90795734660a77ab9e63833e4`。
该终态只证明评审链路尚未形成完整代理验证，不改变策略或 evidence registry 结论。

## v1.1 结构化加固

v1 脱敏调用审计显示 65 次逻辑调用均使用供应商已接受的 `json_object`，没有 fallback；
失败分类为 8 次 `llm_response_schema_invalid` 和 1 次
`llm_response_batch_mismatch`。由于响应正文按安全要求未保存，截断没有可验证证据，不能
作为已确认根因。v1.1 保留原生 JSON mode 和严格 Pydantic Schema，把固定 batch 从 8
降为 1，并在 Prompt 中冻结唯一顶层对象、精确键、枚举和单一 item identity。

每个单项统一最多两次逻辑 attempt、固定 1 秒退避、温度 0、最大并发 4；符合 Schema 的
响应立即 hash 锁定，失败项在同一次调用中用完相同 attempt 预算。resume 只跳过当前
v1.1 contract 下已锁定且 hash 可复核的响应；模型、Prompt、协议、输入或 runtime 摘要
漂移都会拒绝混用。额外文本、缺键、非法枚举、重复/遗漏 item 均不会被解析修复。

v1.1 仍要求 471 项从头完成第一轮，只有完整后才开始第二轮；只有双评和全部分歧裁决
完整后才可解盲评分。v1 历史证据保持只读并由 v1.1 protocol 的 predecessor hash 绑定。

### v1.1 实际终态

模型 `deepseek-ai/deepseek-v4-flash`、Prompt `1.1.0` 的完整第一轮覆盖 471/471：291 项
严格锁定，180 项终止失败。终止项中 106 项以 Schema failure 结束，74 项以 provider
failure 结束；689 次逻辑调用包含 291 次成功、233 次 Schema 无效、2 次 identity
mismatch、155 次 HTTP 429 和 8 次 HTTP 503。供应商报告 391,725 prompt tokens、
55,507 completion tokens、447,232 total tokens，费用字段仍为 `not_available`。

第一轮完整锁定门槛未满足，所以第二轮、裁决、解盲和统计均没有运行。发布目录
`benchmark/llm_relevance_judging_v1_1_record160_incomplete/` 只有脱敏调用 hash、usage、
失败分类和覆盖状态；没有部分 labels 或 statistics。`calls.jsonl`、`manifest.json`、
`status.json` 的 SHA-256 分别为
`b5561e05c8b188c31129ccab40ae576df43022fc1f05841096c378ad61fd8ec3`、
`949b38847337a27681c21e1bbf0e4b233ffe97aef281a70517a95ce9c8272aae`、
`ed836ceeda568d177c136effff4244369dc070ad60f896ab0ebdcf707d8c18c7`。
该终态不能产生人工 Precision、官方成绩或内部 LLM-proxy 效果结论。

## 后端资格复核

后续 `judge_backend_qualification_v1` 没有重跑 471 项，而是只读取上述 v1/v1.1 脱敏
证据并执行 24 条预注册合成 canary。当前唯一配置候选取得 6/24 provider/Schema 成功、
3/24 严格成功；其余 18 条是 11 次 timeout 和 7 次 HTTP 503，另有 3 条响应因额外 HTTP
attempt 不满足资格门槛。因此没有合格后端，也没有生成后续全量运行建议、标签或统计。
该结论只说明当前后端/协议组合未通过 conformance，不评价相关性质量。协议和证据见
[`docs/judge-backend-qualification.md`](judge-backend-qualification.md)。
