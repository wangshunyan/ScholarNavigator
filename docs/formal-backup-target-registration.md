# 备份目标登记与本地探测

`formal_backup_target_registration_v1` 为备份成员发现器提供唯一的显式本机路径入口。登记文件是操作者私有输入，不属于仓库产物；每项仅含 opaque alias、绝对本地路径、固定用途、固定探测范围和 opaque 操作者身份。仓库证据、CLI 输出和日志只保存路径绑定哈希，不保存或回显绝对路径、用户名、主机名或环境值。

门禁只访问登记项对应的精确目录，不递归扫描 HOME、根目录、网络账户、云账户或其他位置。路径不存在、符号链接/别名、重复路径、重复设备或文件系统、路径替换和撤销后使用都会失败。探测复用 `formal_backup_target_attestation_v1` 的原子替换、file/dir fsync、advisory lock、并发 writer、写入校验删除及路径能力检查，临时探测文件必须清理。

预检成功只生成 `registered_candidate`。配额或独立故障域无法从本地证明时保持 `not_available`，候选仍必须依次通过 `formal_backup_target_attestation_v1` 和 `formal_backup_set_member_intake_v1`；登记不会占用 slot、激活备份集或解除 Full1000 阻断。

私有登记文件示例（必须保存在仓库外或未跟踪位置）：

```json
{
  "registration": "formal_backup_target_private_registration_v1",
  "schema_version": "1",
  "source_commit": "29fc5556c0b6af65a96673b170cbbcae50735e06",
  "protocol_sha256": "<frozen protocol sha256>",
  "targets": [{
    "alias": "backup-target-local-1",
    "path": "<absolute operator-private path>",
    "purpose": "full1000_backup_member_candidate",
    "allowed_probe_scope": "exact_registered_directory_only",
    "operator_identity": "<64 lowercase hexadecimal opaque identity>"
  }],
  "revoked_aliases": []
}
```

只读命令：

```bash
python scripts/check_formal_backup_target_registration.py register-dry-run --registration <private.json>
python scripts/check_formal_backup_target_registration.py verify-registration --registration <private.json> --manifest <manifest.json>
python scripts/check_formal_backup_target_registration.py probe-target --registration <private.json> --alias backup-target-local-1
python scripts/check_formal_backup_target_registration.py simulate-profiles
python scripts/check_formal_backup_target_registration.py audit-readiness
```

退出码为 `0=registered_candidates_ready`、`2=registration_or_probe_violation`、`3=no_real_registered_candidates`、`4=usage_error`。当前仓库不含私有登记文件，因此真实审计返回 3；该工程状态不证明备份目标合格、Full1000 完成或正式验证完成。
