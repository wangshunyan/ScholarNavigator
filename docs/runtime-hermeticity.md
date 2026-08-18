# 离线运行环境封闭性门禁

`runtime_hermeticity_v1` 验证同一份本地 Replay 在环境变量、Python hash seed、工作目录、
HOME/TMPDIR、时区、locale 和线程相关环境变化下保持语义一致，并在业务执行边界阻断未登记
I/O。它不读取 gold，不计算 Precision/Recall/F1，不访问外部服务，也不替代官方 scorer。

## 协议与审计边界

版本化协议位于 `benchmark/runtime_hermeticity_v1_protocol.json`。业务入口复用
`execution_determinism_v1` 的真实 `SearchService` fixture Replay；依赖导入完成后、加载
业务协议与 Replay 输入前才启用审计钩子。这样解释器和依赖模块的正常加载不被误报，实际
业务内容访问仍被完整约束。

允许输入固定为两个精确文件及其 SHA-256：执行协议和本地 retrieval fixture。不能按仓库、
用户目录或任意父目录宽泛放行。业务阶段只允许读取这两个文件；输出只能落在该 profile 的
隔离输出目录。协议登记了可读取的非敏感环境 key，但正常 fixture Replay 当前不读取任何
环境变量。

门禁拦截并归因以下操作：

- socket connect、DNS 和其他网络尝试；
- `.env`、HOME 配置、敏感环境变量和任意未登记内容文件读取；
- 输出目录外写入、未登记缓存/临时残留和 Snapshot 写入；
- 业务阶段启动的未登记子进程；
- 哨兵秘密出现在 worker stdout、stderr、结果、事件或错误报告中。

测试只在临时目录创建明显的伪 `.env`、伪 HOME credential 和伪 key。门禁不读取项目
`.env`。违规资源只输出稳定脱敏身份，不回显路径、内容或环境值。

## 环境 profile 与语义比较

协议预登记七类 profile：最小环境、不同 `PYTHONHASHSEED`、不同 cwd/HOME/TMPDIR、
UTC/Asia-Shanghai 时区、C/UTF-8 locale、不同线程环境和含大量无关/敏感哨兵变量的污染
环境。系统不支持的 locale 返回 `profile_not_supported`，不能伪装通过。

每个 profile 比较完整规范化逐查询结果、排名、统一身份、阶段终态、语义事件以及
`result_lineage_v1` 摘要。规范化只复用 `execution_determinism_v1` 已登记的字段级瞬态
排除规则；列表顺序保留，未知字段参与比较。观察性血缘开启前后，生产结果 hash 必须一致。

## CLI 与退出码

```bash
PYTHONPATH=src python scripts/check_runtime_hermeticity.py check
PYTHONPATH=src python scripts/check_runtime_hermeticity.py check --fault network_attempt
PYTHONPATH=src python scripts/check_runtime_hermeticity.py audit-frozen
```

退出码固定为：

- `0=passed`
- `2=hermeticity_or_semantic_violation`
- `3=not_eligible/profile_not_supported`
- `4=usage_error`

可控 fault 仅验证门禁，不改变生产路径。AutoScholarQuery Record160/Full1000 缺少可复放
输入和声明式 I/O 契约，因此只读资格审计固定返回 `not_eligible`，不会补造证据。

离线回归阶段可独立执行：

```bash
PYTHONPATH=src pytest -q -m runtime_hermeticity_regression
```

同环境执行确定性、跨目录胶囊复现、运行时封闭性和检索质量评测是四个独立边界，彼此不
替代。
