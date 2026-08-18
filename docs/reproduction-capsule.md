# 离线可移植复现胶囊

`reproduction_capsule_v1` 用于封装一个已经满足新格式契约的离线 Replay/Benchmark
运行。它验证“换目录以后仍能仅凭已登记输入复放”，不计算 Precision、Recall、F1，也不
等同官方 scorer 或质量评测。

## 契约与信任边界

导出只接受同时具备以下证据的运行：

- 通过校验的 `run_manifest_v1`；
- `crash_consistency_v1` 的完整 `COMMITTED` generation 链和 `RUN_COMPLETED`；
- query-only 输入、数据身份、Prompt 注册表、配置/预算、evaluator 与确定性参数；
- `execution_determinism_v1` 本地 Replay 协议及其原始响应；
- 按 query 顺序提交的规范化结果、阶段终态和语义事件摘要。

manifest 若登记可选 `resource_ledger_v1`，其账本文件会和其他封闭 outputs 一样进入胶囊
清单并接受大小/SHA-256 校验；胶囊不会从日志或兼容报告重建缺失账本。

归档只包含数据。`entrypoint.kind` 固定为 `host_search_service_replay`，复放由当前宿主
checkout 中的 `SearchServiceFixtureBackend` 执行，`execute_archived_code=false`。胶囊
生成 commit 必须与宿主 commit 一致。Prompt 若实际使用，导出器会登记 Prompt manifest
引用的模板文件；缺文件即不具备自包含资格。

文件清单对每个成员登记角色、字节数和 SHA-256。归档采用未压缩 USTAR，成员按路径排序，
mtime/uid/gid 固定为 0，权限固定为 `0644`，路径固定为 NFC POSIX 相对路径。稳定摘要不
包含绝对路径、用户名、运行时间、日志、缓存、`.env`、凭据或 `third_party`。

## 安全导入

导入先完整验证，再写入同一临时父目录中的 staging 目录，最后原子替换；失败时 staging
会被删除。以下输入会被拒绝：

- 绝对路径、`..`、反斜线、非 NFC 路径；
- 符号链接、硬链接和其他特殊文件；
- 重复成员、Unicode/casefold 后冲突的路径；
- 未登记、缺失、大小或 SHA-256 不一致的成员；
- 压缩归档、超出文件数/单文件/总解包大小上限的归档；
- query 顺序、Prompt、预算、evaluator、Replay 协议或 generation 谱系漂移。

校验器不会调用归档内命令，也不会递归忽略未知字段。规范化只使用
`execution_determinism_v1` 预登记的字段级瞬态排除规则，论文排序和事件顺序保持原样。

## CLI 与退出码

```bash
PYTHONPATH=src python scripts/check_reproduction_capsule.py export \
  --source-root /path/to/new-format-run \
  --capsule /tmp/run.capsule.tar

PYTHONPATH=src python scripts/check_reproduction_capsule.py verify \
  --capsule /tmp/run.capsule.tar

PYTHONPATH=src python scripts/check_reproduction_capsule.py replay \
  --capsule /tmp/run.capsule.tar

PYTHONPATH=src python scripts/check_reproduction_capsule.py audit-frozen
```

输出为稳定、机器可读 JSON：

- `0=passed`
- `2=integrity_or_replay_mismatch`
- `3=not_eligible/non_self_contained`
- `4=usage_error`

失败只返回阶段、invariant、相对成员或 query identity、首个差异路径及规范化摘要，不回显
原始响应、请求头、凭据或机器路径。`--fault semantic_result_change` 仅用于离线验证退出码
2，不修改胶囊。

## 冻结基线与离线回归

AutoScholarQuery Record160 和 Full1000 是旧格式冻结产物：它们缺少完整 Replay 输入、
`run_manifest_v1` 和提交代际证据，因此当前只读资格结果为 `not_eligible`。不得反向补造
或修改这些产物。

快速门禁：

```bash
PYTHONPATH=src pytest -q -m reproduction_capsule_regression
```

测试胶囊只创建在 pytest 临时目录。静态哈希完整性、跨执行方式确定性、运行中崩溃恢复、
可移植离线复现和检索质量评测分别由独立门禁承担，互不替代。
