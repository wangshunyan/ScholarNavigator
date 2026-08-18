# Full1000 多卷分片存储

`formal_multivolume_storage_v1` 是 `formal_run_storage_governance_v1` 的只读
拓扑扩展。它不修改 Full1000 原计划、预算或检索输出，而是把 20 个 shard 按稳定
volume identity 排序后 round-robin 分配到多个文件系统。旧单卷
`713,501,442,048` bytes 的要求被分解为逐卷
`assigned_shards × 35,030,827,008 + aggregate（仅首卷）+ 10 GiB reserve`；
总预算没有降低。

每个 shard 的 pending、提交代、资源账本、原始响应和操作审计链必须共处同一文件系统，
因此提交仍只需要同文件系统原子 replace，不依赖跨文件系统 rename。resume 必须保持
原 shard→volume 映射；迁移只能经过“已验证备份→空目标恢复→哈希核验→新 host/storage
attestation”，直接移动目录会被拒绝。

容量门禁逐卷检查可用 bytes、inode、显式 quota、安全余量和并发 writer 上限。
总容量足够不能弥补单卷碎片、未知 quota 或 inode 不足。每个主卷还必须绑定已验证容量
且处于独立故障域的备份卷。aggregate 只保存各卷最终 selected attempt 的
manifest/hash 引用，不复制或改写 shard 历史。

```bash
PYTHONPATH=src python scripts/check_formal_multivolume_storage.py \
  build-topology --profiles VOLUMES.json --output TOPOLOGY.json
PYTHONPATH=src python scripts/check_formal_multivolume_storage.py \
  verify-capacity --profiles VOLUMES.json --topology TOPOLOGY.json
PYTHONPATH=src python scripts/check_formal_multivolume_storage.py simulate-run
PYTHONPATH=src python scripts/check_formal_multivolume_storage.py audit-readiness
```

退出码为 `0=multivolume_storage_ready`、
`2=topology_or_storage_violation`、
`3=not_ready_missing_qualified_volumes`、`4=usage_error`。
当前主机只观测到当前工作文件系统，缺少显式 quota、合格附加卷和独立备份故障域，
所以真实 readiness 保持 exit 3。目录、稀疏空间或同文件系统路径不会被计作独立卷。
该工程门禁不启动正式运行，也不证明 Precision、Recall 或官方成绩。
