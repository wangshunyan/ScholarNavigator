# 离线执行确定性门禁

`execution_determinism_v1` 验证同一份本地 Replay 输入在不同执行方式下是否产生等价的
逐查询语义结果。它不会访问 gold，不计算 Recall/F1，不调用官方 scorer，也不证明检索质量。
协议冻结于 `benchmark/execution_determinism_v1_protocol.json`。

## 验证范围

门禁通过真实 `SearchService`、现有 query planning、fixture retriever、判断、排序、去重、
阶段诊断和事件回调执行以下六项 invariant：

1. 同配置连续重复执行；
2. 单查询逐条执行与批量执行；
3. 查询批次反序后按稳定 query identity 对齐；
4. 串行与固定 worker 数的受控并发；
5. 完整运行与合法 checkpoint/resume；
6. 取消一条查询后，同一执行器的下一条查询不继承取消状态或事件。

checkpoint 使用预期 query identity 顺序、配置摘要和已完成 identity 摘要；重复、未知、遗漏、
重排或配置漂移均被拒绝。默认 fixture 按本地 `retrieval_outputs.json` 文件顺序取前三个唯一
query，不读取 `search_cases` 或任何 gold 字段。

## 规范化规则

比较保留检索响应、论文身份、列表顺序、排序结果、阶段状态、事件名称/载荷/顺序及未知
字段。只有协议 `canonicalization.excluded_fields` 中逐路径登记的瞬态字段可被替换，包括：

- 本次调用的 run ID、PID、调用序号；
- 总耗时、阶段耗时、connector 耗时、rate-limit 等待和 budget elapsed；
- 事件载荷中的耗时或时间戳。

通配符只匹配一个明确路径段。列表绝不排序，字典只在最终 JSON 序列化时按 key 排序；未知
字段不会被递归忽略。报告同时给出每条排除规则在 baseline 中的实际匹配次数。

## CLI 与退出码

```bash
PYTHONPATH=src python scripts/check_execution_determinism.py check \
  --output outputs/benchmark_runs/execution_determinism_gate/report.json
```

可控故障仅用于验证门禁定位能力：

```bash
PYTHONPATH=src python scripts/check_execution_determinism.py check \
  --fault semantic_result_change
```

冻结 AutoScholarQuery 160/1000 产物只读资格审计：

```bash
PYTHONPATH=src python scripts/check_execution_determinism.py audit-frozen
```

退出码固定为：`0=passed`、`2=invariant_violation`、
`3=not_eligible/insufficient_fixture`、`4=usage_error`。违规 JSON 只输出 invariant、query
identity、首个差异 JSON path 及两侧规范化摘要，不包含环境变量或认证信息。

现有 160/1000 冻结运行缺少完整 `run_manifest_v1` 的 query 顺序、确定性参数、evaluator、
谱系和完成数声明，因而固定返回 `not_eligible`，不会补造元数据或宣称通过。

## 与其他验证的边界

- 快照回归一致：确认冻结 Snapshot、候选和内部指标没有漂移。
- 跨执行方式确定性：本门禁确认同一 Replay 输入不受共享状态、顺序、并发、resume 或取消污染。
- 官方质量评测：必须由官方数据与官方 scorer 单独完成；本门禁不产生竞赛成绩。

离线回归阶段可单独快速运行：

```bash
PYTHONPATH=src pytest -q -m execution_determinism_regression
```
