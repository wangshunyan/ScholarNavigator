# 离线发布候选复现门禁

`release_candidate_reproducibility_v1` 从固定 Git 提交构造软件工程发布候选。它不包含
Benchmark 运行结果、query/论文正文、质量指标或官方提交物，也不会读取 `.env`、未跟踪文件和
`third_party`。

## 契约与构建边界

版本化契约固定源码文件清单及 SHA-256、`SOURCE_DATE_EPOCH`、Python/Node 工具链、依赖声明与
锁文件、允许环境变量、构建命令和归档元数据。源码通过 `git show <commit>:<path>` 写入两个
独立临时树；本地工作树和既存子模块状态不会进入发布包。构建输出包括 Python wheel、源码归档、
Next/webpack 前端静态归档、SBOM、发布 manifest 和外层确定性归档，二进制只存在于临时目录，
不提交仓库。

Python 依赖闭包来自当前离线安装元数据并另存受控 lock；根声明中未精确锁定的依赖会被显式
计数并作为供应链违规。Node 依赖来自 `package-lock.json`。许可证只采用声明值，无法证明时保留 `unknown`，不联网
补全。开发依赖可以参与前端构建，但不得作为前端静态运行文件进入归档。

```bash
PYTHONPATH=src python scripts/check_release_candidate.py build --output /tmp/spar-release
PYTHONPATH=src python scripts/check_release_candidate.py verify --release /tmp/spar-release/first/release
PYTHONPATH=src python scripts/check_release_candidate.py compare \
  --first /tmp/spar-release/first/release --second /tmp/spar-release/second/release
PYTHONPATH=src python scripts/check_release_candidate.py audit-readiness \
  --evidence benchmark/release_candidate_reproducibility_v1_evidence/current.json
```

退出码为 `0=reproducible_release_ready`、`2=build_or_supply_chain_violation`、
`3=not_ready_missing_offline_dependency_or_input`、`4=usage_error`。门禁不会下载缺失依赖。
若权威撤销账本存在活动事件，所有 build/verify/compare/audit 入口都会在读取发布资格前
fail-closed；历史发布包即使文件哈希完整，也不能继续声明对应证据有效。

## 当前资格

历史 a743 构建中，wheel、源码归档和 SBOM 逐字节一致，但前端 webpack 静态归档随源码父路径
漂移；原始失败证据保持不变。后续 `frontend_reproducible_build_v1` 通过 canonical staging
消除了该路径依赖，当前两份前端归档的 56 个成员逐字节一致。由于 11 条 Python 根声明仍未
精确锁定，完整发布候选继续为 `not_qualified`、退出码为 2。门禁不忽略或规范化实际内容差异，
也不把 Turbopack 沙箱限制当成产品失败。详见
[`docs/frontend-reproducible-build.md`](frontend-reproducible-build.md)。
