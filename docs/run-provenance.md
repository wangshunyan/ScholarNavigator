# 实验产物谱系与完整性门禁

`run_manifest_v1` 是 Benchmark/Replay 的内部、离线产物契约。它组合已有的
Snapshot 内容哈希、query-only manifest、Prompt manifest、evaluator 版本与 checkpoint
信息，不调用 SearchService、connector、LLM 或 evaluator，也不读取 gold。门禁只证明
产物来源和文件完整性；它不是官方 scorer，也不代表比赛成绩或检索效果通过。

## 契约

稳定字段如下；所有路径均相对仓库根目录，禁止绝对路径和 `..`：

- `manifest_kind=run_manifest_v1`、`schema_version=1`、稳定 `run_id`；
- `dataset`：名称、版本、输入文件大小/SHA-256，以及这些字段的身份摘要；
- `queries`：query-only JSONL、ID/文本字段、数量、与顺序无关的身份摘要和顺序摘要；
- `prompt`：Prompt manifest 文件身份、使用的 Prompt 版本及本次是否调用 Prompt；
- `configuration`：有序来源、显式预算、关键配置值和稳定摘要；
- `evaluator`：内部 evaluator 名称与指标版本；
- `determinism`：随机种子和排序、温度等确定性参数摘要；
- `progress`：`planned/running/partial/completed/failed`、预期/完成数量和记录文件；
- `lineage`：checkpoint ID、resume 序号及父 manifest 相对路径、文件哈希、run/checkpoint ID；
- `git`：生成时 commit、dirty 路径和稳定工作区摘要。既存
  `third_party/paper-qa` 子模块状态会被显式记录为 `allowed_dirty_paths`，其他 dirty 路径
  仍列入 `unexpected_dirty_paths`；不会读取 `.env`；
- `outputs`：每个关键输出的角色、格式、相对路径、大小、SHA-256 与 JSONL 记录数；
- `metadata_bindings`：运行 config 中 dataset、Prompt、sources、budgets、evaluator 字段到
  manifest 字段的 JSON Pointer 绑定；七类必需绑定不可省略；
- `score_scope=internal_not_official`。

启用字段级血缘的新运行还必须将 `result_lineage.jsonl` 以角色
`result_lineage_v1` 登记到 `outputs`；该文件和其他报告使用相同封闭清单、大小、记录数及
SHA-256 校验。血缘契约本身见 [`docs/result-lineage.md`](result-lineage.md)，旧运行不得
事后补造该输出。

启用资源账本的新运行还可登记 `resource_ledger_v1` binding：其输出文件必须同时以
`resource_ledger_v1` 角色出现在 `outputs`，并绑定 opaque run identity、账本 authority
manifest identity 与 `committed_generation_only` 权威边界。校验器会离线执行账本守恒
门禁，并核对 run identity、query 数量及文件哈希；兼容镜像或日志不能替代该 binding。
详见 [`docs/resource-accounting-integrity.md`](resource-accounting-integrity.md)。

JSON 采用键排序、UTF-8、固定缩进和末尾换行。摘要不包含时间戳、绝对路径、用户名或
机器临时目录。query 身份摘要对输入顺序不敏感，顺序摘要单独记录，因此内容变化和重排
都可定位。输出目录采用封闭清单：未登记文件、缺文件、大小或 SHA-256 变化都会失败；
manifest 若置于该目录，必须在生成前将其相对目录的路径写入
`output_inventory_excludes`，避免自引用哈希。

## 生成和校验

新运行先生成一个不含 gold 的 JSON spec，字段对应上述契约；`outputs` 必须已存在，
`metadata_bindings` 必须指向已登记的 JSON 配置输出。然后执行：

```bash
PYTHONPATH=src python scripts/check_run_provenance.py generate \
  --spec path/to/run_manifest_spec.json \
  --output path/to/run_manifest.json

PYTHONPATH=src python scripts/check_run_provenance.py validate \
  --manifest path/to/run_manifest.json \
  --output outputs/benchmark_runs/run_provenance_gate/report.json
```

resume 子运行必须将父 manifest 路径、SHA-256、run ID 和 checkpoint ID 全部绑定；门禁
递归校验父链、resume 序号、query/config 一致性和进度单调性，并拒绝缺父、哈希漂移、
谱系循环或进度倒退。普通校验安装 socket 护栏，机器结果中的 `execution` 必须保持
0 network、0 LLM、0 Snapshot write、`gold_fields_accessed=false`。

退出码固定为：

| 退出码 | 状态 | 含义 |
| --- | --- | --- |
| `0` | `passed` | v1 Schema、谱系、绑定、封闭输出清单和文件身份全部通过 |
| `2` | `invalid` | 完整性、Schema、谱系、计数或元数据不一致 |
| `3` | `legacy_metadata_incomplete` | 旧产物文件可只读核验，但缺少 v1 所需元数据，不能宣称通过 |
| `4` | `invalid` | CLI/spec/JSON 使用错误，未生成可验证 manifest |

最小离线回归入口为：

```bash
PYTHONPATH=src pytest -q -m run_provenance_regression
```

## 冻结 AutoScholarQuery 只读审计

旧 160 分析输入来自同一未完成 1000 条运行的 162 条 Record，其中 160 条进入主分析、
2 条无成功来源被单列。只读 profile 位于
`benchmark/run_provenance_legacy_profiles.json`，审计命令为：

```bash
PYTHONPATH=src python scripts/check_run_provenance.py audit-legacy \
  --profile benchmark/run_provenance_legacy_profiles.json \
  --output outputs/benchmark_runs/run_provenance_legacy_gate/audit.json
```

当前冻结文件哈希均可核验，观测 Record 数为 162：相对 162 条分析输入，文件数量闭合；
相对预期 1000 条，仍只有 162 条。两者都缺少完整 query 身份/顺序绑定、evaluator 版本、
确定性参数、输出清单、completed claim 与 checkpoint 谱系，因此都必须返回退出码 `3`。
审计不会修改旧结果、Snapshot 或 resume manifest，也不会推断缺失字段。
