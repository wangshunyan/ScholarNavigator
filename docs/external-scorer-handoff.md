# 外部官方 scorer 交接门禁

`external_scorer_handoff_v1` 验证未来官方 scorer 的输入交接、隔离执行、不可变性和输出校验
链路。当前官方 package、完整 Schema 和指标命名空间均未提供，因此所有未知字段保持
`unknown/not_provided`；本门禁不会猜测官方输入格式、公式或成绩。

## 契约与 canonical handoff

scorer package manifest 绑定 scorer 名称与版本、入口文件 SHA-256、输入/输出 Schema 摘要、
运行时、资源上限、允许 I/O、确定性要求和指标命名空间。canonical handoff 只包含稳定 query
identity、全局顺序、最终 result identity、rank 与权威字段摘要，并绑定对应 run manifest 和
commit generation SHA-256。缺少任一权威绑定的旧运行不能后验升级为正式输入。

Record160 的旧交付证据可证明 160 条内部结果的身份与顺序，但缺少新式 run manifest 和提交代；
它只能用于资格说明，不能作为正式提交。Full1000 未完成和官方 scorer 缺失继续是独立阻断。

## 隔离边界

scorer 在最小环境的受控 Python 子进程中执行。入口和 handoff 在执行前校验哈希，输入只读，
唯一允许输出位于临时目录。业务边界阻断 socket/DNS、`.env`/HOME 与未登记文件访问、未登记
子进程和输入修改；执行后再次校验输入哈希。输出必须严格满足已登记 Schema、完整覆盖 query、
只使用登记命名空间、仅含有限数值，并在两次执行间字节一致。malformed、部分或随机输出不会
被修复或补全。

## CLI 与退出码

```bash
PYTHONPATH=src python scripts/check_external_scorer_handoff.py run --synthetic-matrix
PYTHONPATH=src python scripts/check_external_scorer_handoff.py audit-readiness
```

CLI 还提供 `prepare` 和 `verify-package`。退出码为：

- `0`：`handoff_chain_verified`，仅表示合成交接链路通过；
- `2`：scorer、Schema、不可变性或隔离违规；
- `3`：缺少官方 scorer 或完整权威输入；
- `4`：命令使用错误。

合成 scorer 的计数字段属于 `synthetic_handoff` 协议一致性值，不是质量指标。任何门禁结果都
不是人工 Precision、内部 Recall/F1 或官方成绩，也不会解除 validation readiness 中的三项正式
阻断。
