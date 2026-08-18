# Official scorer package intake

`official_scorer_package_intake_v1` 是未来赛事方 scorer 材料的离线接收门禁。它复用
`external_scorer_handoff_v1` 的 canonical handoff 与隔离执行器，不建立第二套评分链，
也不定义或推测官方 Schema、指标、方向或成绩。

## 接收契约

离线 kit 以一次性 challenge 绑定 Full1000 计划、canonical handoff、预注册、
quarantine、clearance、public contract 和代码提交。kit 只依赖 Python 标准库，可在
`python -I -S` 和无仓库目录运行。候选包只允许包含：

- scorer 名称、版本、入口和完整文件清单；
- 输入/输出 JSON Schema；
- 指标 namespace、数值类型、方向和缺失语义；
- runtime、资源上限、允许 I/O、确定性声明；
- 每个文件的相对路径、大小和 SHA-256，以及结构化来源证据类型。

缺失材料必须保持 `unknown` 或 `not_provided`。接收器拒绝路径穿越、绝对路径、链接、
重复成员、压缩率/大小/文件数超限、重复 JSON key、NaN/Infinity、入口或 inventory
哈希漂移、非严格 Schema、额外指标、跨 challenge/提交/协议复用、撤销包和重复导入。

SHA-256 只证明接收内容未变，不认证赛事方身份。没有独立来源真实性证据时，状态固定为
`unverified_origin`，不能进入正式 clearance。

## 无成绩 conformance

候选入口只在既有 scorer sandbox 中，以脱敏合成 canonical input 运行两次。检查范围是
启动、输入不可变、Schema/覆盖、双跑确定性、零网络、零 HOME/`.env` 读取、零未登记
子进程及受限输出。合成输出只用于协议状态转换，不是 Record160/Full1000 指标或官方成绩。

```bash
PYTHONPATH=src python scripts/check_official_scorer_intake.py \
  conformance-dry-run --matrix
PYTHONPATH=src python scripts/check_official_scorer_intake.py audit-readiness
```

退出码为 `0=official_scorer_package_qualified`、
`2=package_schema_or_sandbox_violation`、
`3=not_ready_missing_verified_official_package`、`4=usage_error`。当前真实审计必须返回
3，因为官方 package、输入/输出 Schema、指标 namespace 和来源真实性证据均未提供。
