# 可移执行节点资格包

`portable_execution_site_attestation_v1` 复用
`formal_execution_host_attestation_v1`、`formal_multivolume_storage_v1`、
`formal_run_storage_governance_v1` 与 `full1000_launch_control_v1` 的既有标准，
让候选 Full1000 主机在不克隆仓库、不联网且不安装项目依赖的情况下完成文件系统取证。
它不建立较宽松的第二套主机标准。

资格包是确定性 ZIP，固定成员顺序、时间、权限和无压缩编码。包内 `probe.py` 与
`verify.py` 只依赖 Python 标准库，可用 `python -I -S` 执行；`site_contract.json`
绑定 Full1000 计划、当前提交、20 个 shard、两主卷/两备份卷拓扑、逐卷容量/inode/writer
要求及一次性 challenge。包不包含 query、项目代码依赖、`.env`、凭据或绝对路径。

```bash
PYTHONPATH=src python scripts/check_portable_execution_site.py build-kit \
  --challenge <64-hex-one-time-challenge> \
  --issued-epoch <integer-epoch> \
  --output /offline-transfer/site-kit.zip
PYTHONPATH=src python scripts/check_portable_execution_site.py verify-kit \
  --kit /offline-transfer/site-kit.zip
```

候选节点对每个槽位显式传入路径，但路径只用于当次探测，不写入证明。quota 与主备故障域
必须由 challenge 绑定的 operator observation 提供；未知值 fail-closed。probe 还验证
same-filesystem atomic replace、file/dir fsync、advisory lock、权限、Unicode/大小写、
writer 数及非空恢复拒绝。目录别名或相同文件系统不能冒充独立 volume。

```bash
python -I -S probe.py probe \
  --contract site_contract.json \
  --volume primary-00=<path> --volume primary-01=<path> \
  --volume backup-00=<path> --volume backup-01=<path> \
  --site-evidence site-evidence.json \
  --observation-epoch <integer-epoch> \
  --output execution-site-attestation.json
python -I -S verify.py verify \
  --contract site_contract.json \
  --attestation execution-site-attestation.json
```

仓库侧导入会重新验证包、证明自哈希、计划/提交/拓扑绑定、逐卷资格与 freshness，并在
append-only challenge 账本中拒绝重放。只有非合成、未过期且未使用的合格证明才生成
`portable_execution_site_import_receipt_v1`；未来 launch authorization 必须引用该回执。
哈希证明内容一致性，不认证主机或操作者身份。

```bash
PYTHONPATH=src python scripts/check_portable_execution_site.py \
  import-attestation --kit <kit.zip> --attestation <attestation.json> \
  --ledger <challenge-ledger.json> --current-epoch <integer-epoch>
PYTHONPATH=src python scripts/check_portable_execution_site.py simulate-site
PYTHONPATH=src python scripts/check_portable_execution_site.py audit-readiness
```

退出码为 `0=execution_site_qualified`、
`2=attestation_or_import_violation`、
`3=not_ready_no_qualified_external_site`、`4=usage_error`。当前仓库没有真实外部节点证明，
真实审计固定返回 3，不启动 Full1000，也不解除 Full1000、人工 Precision 或官方 scorer
三项阻断。
