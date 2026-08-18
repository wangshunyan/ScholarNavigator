# 发布身份签名与信任根门禁

`release_authenticity_signing_v1` 使用系统 OpenSSH `ssh-keygen -Y sign/verify`
和 Ed25519 为发布候选的规范 envelope 提供离线签名验证。它不自创密码算法，也不把
内容哈希、透明日志 Merkle root 或测试密钥误称为组织身份认证。

## 签名边界

待签名 envelope 精确绑定 artifact 类型/版本及 SHA-256、透明日志 root/sequence、
代码提交、readiness 状态、三项正式阻断、算法、key identity、namespace、
`test_only` 与签名上下文。规范 JSON 固定 UTF-8、键序、缩进和尾换行；时间戳、
绝对路径、用户名和主机身份不进入签名内容。

可签对象为透明日志 checkpoint、standalone 审计包、软件发布候选和未来 clearance
receipt。当前对象仍是候选，不是正式发布或官方成绩。

## 私钥与信任根

私钥只通过操作者提供的路径交给受控 `ssh-keygen` 进程。程序不读取、复制、打印、
归档或提交私钥；仓库中的
`benchmark/release_authenticity_signing_v1_trust_root.json` 仅保存公钥状态结构。
当前清单没有真实公钥，因此真实审计固定为
`not_ready_missing_real_trust_anchor_or_signer`。

信任根状态为 `active → rotated/revoked`。轮换声明必须由旧 active key 签署，或使用
预登记的离线恢复规则；历史签名按签发 sequence 继续可验证，rotated/revoked key
不得签发新对象。测试密钥始终带 `test_only=true`，不能满足真实发布条件。

## CLI

```bash
PYTHONPATH=src python scripts/check_release_authenticity.py audit-readiness
```

退出码：

- `0`：签名与信任根控制通过；
- `2`：签名、绑定或信任根违规；
- `3`：缺少真实信任锚、签名者或系统签名工具；
- `4`：用法错误。

`generate-test-key`、`sign-dry-run` 与 `verify` 只用于临时目录合成演练。真实私钥路径
不应写入脚本、日志、协议、readiness 或证据包。

## 声明边界

签名证明“持有登记私钥的签名者签署了这些规范字节”，透明日志证明内容及历史
一致性；在真实组织信任锚由操作者独立配置前，两者均不证明发布者组织身份。
Full1000、真实人工 Precision 和官方 scorer/schema 三项正式阻断保持不变。
