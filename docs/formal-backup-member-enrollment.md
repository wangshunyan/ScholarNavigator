# Full1000 真实备份成员离线入驻

`formal_backup_member_enrollment_v1` 将既有目标发现、显式登记、目标能力探测和
backup-set slot intake 串成一个离线流程。它不定义新的容量、inode、quota 或故障域
标准；每个 kit 内嵌的 `slot_contract` 直接来自
`formal_backup_set_member_intake_v1`。

流程固定为：验证 kit、由操作者显式提供目标目录、执行路径受限的本地探测、结合
结构化故障域与 quota 证据生成 attestation、验证 slot 绑定，最后输出
`member_candidate_ready_for_intake` 包。即使全部通过，也不会写入真实成员登记链、
占用 slot、激活备份集或解除 Full1000 阻断。

## 隐私与路径边界

Kit 只含协议摘要、一次性 challenge、slot 门槛和 Python 标准库 runtime。它不含
query、凭据、`.env`、用户名、主机名或预填路径。Runtime 只访问命令行显式给出的
单一目录，拒绝不存在路径和符号链接，不扫描 HOME、根目录或网络账户。输出只保存
opaque target identity、path binding hash、能力摘要和脱敏原因码；哈希只证明内容绑定，
不认证设备所有权或操作者身份。

## CLI

```bash
python scripts/check_formal_backup_member_enrollment.py build-kit --members 4 --slot 0 --challenge <64-hex> --output /tmp/member-kit.zip
python scripts/check_formal_backup_member_enrollment.py run-enrollment-dry-run --kit /tmp/member-kit.zip --target /operator/selected/target --evidence /operator/private/evidence.json --observation-epoch 10000 --output /tmp/member-package.json
python scripts/check_formal_backup_member_enrollment.py verify-member-package --kit /tmp/member-kit.zip --package /tmp/member-package.json --observation-epoch 10000
python scripts/check_formal_backup_member_enrollment.py simulate-matrix
python scripts/check_formal_backup_member_enrollment.py audit-readiness
```

退出码为 `0=member_package_ready`、`2=enrollment_or_attestation_violation`、
`3=no_real_enrolled_members`、`4=usage_error`。当前仓库没有真实入驻成员，因此
readiness 稳定返回 3；2/3/4 成员方案分别缺少 2/3/4 个 slot。
