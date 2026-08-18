# Full1000 备份集成员接收与激活

`formal_backup_set_member_intake_v1` 在既有独立备份目标资格和
`formal_backup_set_topology_v1` 之上增加逐槽位接收层。它不降低 bytes、inode、
quota、并发 writer、原子持久化或故障域标准，也不改变单目标默认行为。

## 槽位 kit 与信任边界

2、3、4 成员方案中的每个槽位都有独立的一次性 challenge，并绑定当前提交、
Full1000 计划摘要、备份集拓扑、允许的 shard 集合及该槽位的最坏情况容量门槛。
kit 内的单文件 verifier 只依赖 Python 标准库，可在无仓库环境使用
`python -I -S`。kit 不含 query、凭据、`.env`、主机名、用户名、绝对路径或环境值。

哈希只证明内容一致性，不认证设备所有权、操作者身份或声明来源。真实导入仍须提供
fresh、未撤销、完整故障域与恢复证据；`not_available` 一律 fail-closed。

## 登记、激活与失效

单个槽位的追加状态链为 `empty → qualified → reserved → set_activated`，并允许
进入终止状态 `revoked/invalid`。成员可分时导入，但激活前必须重新验证全部必需
槽位。相同设备、filesystem、quota pool、故障域或管理域不能重复计容；目录别名、
重新挂载或 opaque alias 不能规避检查。

只有全部槽位闭合时才生成 `backup_set_activation_receipt_v1`。receipt 绑定成员
attestation、shard 分配、压实容量模型、恢复命令和 launch addendum。任一成员容量
下降、身份替换、过期、撤销或恢复失败都会使整套激活失效；剩余成员不能继续正式
运行、aggregate 或声称可恢复。

## 命令与当前状态

```bash
python scripts/check_formal_backup_set_intake.py build-slot-kit \
  --members 4 --slot 0 --challenge <64-hex> --output <kit.zip>
python scripts/check_formal_backup_set_intake.py verify-member \
  --kit <kit.zip> --attestation <member.json> --observation-epoch <epoch>
python scripts/check_formal_backup_set_intake.py import-member-dry-run \
  --kit <kit.zip> --attestation <member.json> --observation-epoch <epoch>
python scripts/check_formal_backup_set_intake.py activate-set-dry-run \
  --members 4 --bundle <bundle.json> --observation-epoch <epoch>
python scripts/check_formal_backup_set_intake.py simulate-matrix
python scripts/check_formal_backup_set_intake.py audit-readiness
```

退出码固定为 `0=backup_set_activated`、`2=member_or_activation_violation`、
`3=not_ready_missing_real_members`、`4=usage_error`。当前没有真实合格成员，
2/3/4 成员方案分别缺少 2/3/4 个槽位，因此真实审计返回 exit 3。合成矩阵只证明
接收、恢复和失效控制，不解除 Full1000、真实人工 Precision 或官方 scorer 三项阻断。
