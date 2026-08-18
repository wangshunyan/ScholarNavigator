# 正式验证职责分离与双人控制

`formal_validation_separation_of_duties_v1` 为未来正式验证定义角色、授权和审批链。它只保存
稳定 opaque identity、alias 到 principal binding 的不可逆摘要及角色绑定；姓名、邮箱、
用户名、主机账户、凭据和真实身份映射均不进入仓库。真实身份映射由外部治理流程维护，
当前状态明确为 `not_available`。

## 角色与禁止组合

协议覆盖计划封印者、运行授权者、执行操作者、人工包协调者、标注者 A/B、裁决者、
scorer 接收者、证据审计者、clearance 批准者、发布签名者和撤销管理员。标注者 A/B
必须互异且都不能裁决；执行者不能批准自己的 launch 或 clearance；发布签名者不能成为
撤销的唯一批准者。正式 clearance 至少绑定两个独立 principal 和两个批准域
（assurance 与 clearance governance），且批准者不能是全部上游证据的唯一生成者。

授权记录精确绑定协议、代码提交、artifact 摘要、action、role、alias、状态和前序哈希。
通配 identity、alias 重绑定、事后补签、跨提交复用、重复 action、撤销后继续执行和哈希链
断裂均 fail-closed。正式入口应调用共同的 `verify_entry_authorization`；test-only 合成流程
不得作为正式授权。

## 离线验证

```bash
PYTHONPATH=src python scripts/check_formal_validation_roles.py verify-policy
PYTHONPATH=src python scripts/check_formal_validation_roles.py simulate-ceremony
PYTHONPATH=src python scripts/check_formal_validation_roles.py audit-readiness
```

`verify-authorization` 接收 assignments、authorizations 和 events 三个 JSON 文件，完整验证
角色绑定与两条 append-only 链。退出码为：

- `0`：职责分离控制或合成仪式验证通过；
- `2`：角色、授权、审批、协议或输入完整性违规；
- `3`：工程控制已就绪，但缺少真实角色分配；
- `4`：命令使用错误。

当前合成矩阵只证明控制逻辑：它不是真实双人审批，不生成标签、指标或正式 receipt，不解除
Full1000、真实人工 Precision、官方 scorer/schema 三项阻断。
