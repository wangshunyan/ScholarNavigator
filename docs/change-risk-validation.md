# 变更风险分级与验证范围编排

`change_risk_validation_v1` 把 freshness 组件、公共契约、readiness 只读 gate 和 pytest
入口组合为确定性 validation plan。它的目标是减少与变更无关的机械验证，同时保持
fail-closed：未登记语义文件、跨组件重命名、测试证据缺失或最终 HEAD 漂移都不能被
人工一句“无影响”跳过。

## 风险与升级

- `low`：文档以及协议能证明无语义的注释/格式变更；
- `targeted`：已登记的孤立模块、CLI 或局部门禁，运行对应 pytest 与只读 gate；
- `high`：共享运行时、全局门禁、协议/证据基线、持久化与检索路径，必须完整 pytest；
- `frontend`：前端、API schema/mapper、前端依赖或构建配置，必须 lint 与 build。

所有计划固定包含双跑确定性、敏感扫描、`git diff --check` 和 HEAD/upstream 核验。
出现 high 组件、定向失败、证据缺失、未登记语义文件或被测提交与最终 HEAD 不一致时，
完整 pytest 成为必跑项。仅 frontend 风险触发前端 lint/build；跳过必须在 plan 中登记
稳定原因码。

## 命令

```bash
PYTHONPATH=src python scripts/check_validation_scope.py plan \
  --from BASE_COMMIT --to TARGET_COMMIT
PYTHONPATH=src python scripts/check_validation_scope.py plan-worktree
PYTHONPATH=src python scripts/check_validation_scope.py verify-execution \
  --plan PLAN.json --attestation ATTESTATION.json
PYTHONPATH=src python scripts/check_validation_scope.py audit-current
```

退出码：`0=validation_scope_satisfied`、`2=scope_or_execution_violation`、
`3=validation_incomplete`、`4=usage_error`。attestation 只证明登记命令、退出码和目标
源码身份一致，不证明质量指标或正式验证完成。
