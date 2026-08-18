# 检查点与运行产物崩溃一致性门禁

`crash_consistency_v1` 验证 Benchmark 在持久化任意阶段中断时，只把完整提交的状态交给
resume。它不访问网络、LLM、gold 或 Snapshot，不计算检索质量，也不替代官方 scorer。
协议固定于 `benchmark/crash_consistency_v1_protocol.json`。

## 生产提交边界

新 Benchmark 运行在原输出目录下增加 `.run_commits/generations/`。每个代际只保存本次
增量及以下权威状态：

- `delta.json`：初始化、单查询 upsert 或运行完成增量；
- `checkpoint.json`：游标、记录身份摘要、事件数和运行状态；
- `events.jsonl`：本代语义事件，序号跨代连续；
- `run_manifest.json`：run/config/query 集、完成数、报告清单和父代；
- `generation_manifest.json`：本代文件大小与 SHA-256 封闭清单；
- `RUN_COMPLETED`：仅完整运行具有的完成声明；
- `COMMITTED`：最后写入的代际提交标记。

写入过程在 `generations/` 同目录创建 pending 目录。文件写完后执行 `flush`、文件
`fsync` 和目录 `fsync`，再用同文件系统的目录 `os.replace` 发布代际；最后原子写入并
同步 `COMMITTED`。没有 `COMMITTED` 的目录、pending 目录和文件级临时项都不属于已
提交状态。resume 按父代链验证清单、记录、事件、checkpoint 和 manifest 后选择最新
有效代；最新代损坏时回退到上一有效代，代际编号不会复用。

顶层 `config.json`、`results.jsonl`、`metrics.json` 等保留为兼容镜像，但不再是 resume
的权威输入。镜像只由已提交代际生成；镜像截断或落后不会污染权威检查点。旧运行没有
代际证据时仍可按原逻辑 resume，并从恢复时刻建立新链，但不能反向证明旧内容具备崩溃
一致性。

同一运行目录使用操作系统 advisory writer lock；第二个 writer 会被拒绝，不能覆盖首个
writer，进程退出时锁由内核释放，持久锁文件本身不代表占有状态。
清理只删除命名为 `.generation-*.pending-*` 的未提交目录，不删除已提交或无法证明为
临时的历史目录。

## 故障矩阵与判定

门禁通过可控 writer/filesystem seam 覆盖：临时目录创建前后、截断写、flush/fsync
失败、checkpoint 后中断、manifest 后但完成标记前中断、replace 前后、ENOSPC、权限
错误和提交标记后中断。另行验证正常多代、损坏最新代回退、临时项清理和并发 writer。
测试只使用临时目录，不 kill 进程、不填满磁盘、不使用真实睡眠或概率竞态。

核心 invariant 为：

1. torn JSON、部分记录、混合代际不被读取；
2. 失败提交不修改上一已提交代；
3. resume 不重复或遗漏已提交查询；
4. checkpoint、事件、manifest、报告和完成标记属于同一代；
5. `completed` 必须记录数闭合并具有一致的 `RUN_COMPLETED`；
6. 未提交清理与并发 writer 均保持已提交历史不变；
7. 机器报告只给规范化摘要，不回显绝对路径、认证信息或原始敏感 payload。

## CLI 与退出码

```bash
PYTHONPATH=src python scripts/check_crash_consistency.py check
PYTHONPATH=src python scripts/check_crash_consistency.py check \
  --fault non_atomic_writer
PYTHONPATH=src python scripts/check_crash_consistency.py audit-frozen
PYTHONPATH=src pytest -q -m crash_consistency_regression
```

| 退出码 | 状态 | 含义 |
| ---: | --- | --- |
| 0 | `passed` | 全部提交/恢复 invariant 通过 |
| 2 | `invariant_violation` | 定位到 fault point、代际和首个违规路径 |
| 3 | `not_eligible` | 旧产物缺少原子代际/提交证据 |
| 4 | `usage_error` | 协议、参数或输入不可安全解析 |

`--fault non_atomic_writer` 只在临时夹具中模拟原地破坏已提交代，必须稳定返回 2；它不会
进入生产写入器。AutoScholarQuery 冻结 160/1000 产物只有逐文件哈希和 legacy
checkpoint 信息，没有提交代、目录同步或原子边界证据，因此只读检查固定返回 3，不修改
或补造历史产物。

## 与其他门禁的边界

- `run_manifest_v1`：校验静态文件哈希、配置绑定和 checkpoint/resume 谱系；
- `execution_determinism_v1`：校验同一 Replay 跨执行方式的语义一致性；
- `crash_consistency_v1`：校验运行中持久化的原子提交与恢复；
- Snapshot/current_rules 回归和内部指标：校验冻结结果与检索质量。

四者证据互补，均不是官方比赛计分器。
