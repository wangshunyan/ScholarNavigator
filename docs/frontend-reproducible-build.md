# 前端跨路径可重复构建

`frontend_reproducible_build_v1` 复用软件发布门禁的两棵独立源码树和离线 webpack 构建，专门
验证 Next 前端产物是否受源码父路径、`HOME`、`TMPDIR`、cwd 或空缓存影响。比较采用归档内
每个原始文件的完整字节，不忽略、删除或事后重写差异。

## 可证根因与修复

历史构建的固定 Build ID 相同，未生成 source map，产物也不含源码绝对路径字面量；但 56 个
成员中有 38 个不同。同一 bundle 的模块内容和稳定 module ID 保持一致，而 webpack chunk ID、
chunk 文件名、HTML/RSC 引用及 `.nft.json` trace 会随源码父目录变化。将两份独立 materialized
源码依次复制到同一个、由源码清单摘要确定的 `/tmp` staging identity 后，差异归零。这证明
路径敏感点位于 webpack context 下的 chunk assignment，不是 Build ID、环境变量或 source map。

修复仅作用于发布构建：每次从独立 Git 源码树复制到受锁保护的 canonical staging root，使用
不同隔离 `HOME`/`TMPDIR` 和空 `.next` cache 执行 `next build --webpack`，随后清理 staging。
生产 Next 配置、页面代码、API 类型、检索与排序逻辑均不改变。

```bash
PYTHONPATH=src python scripts/check_frontend_reproducible_build.py run \
  --output /tmp/spar-frontend-build --report /tmp/spar-frontend-report.json
PYTHONPATH=src python scripts/check_frontend_reproducible_build.py verify
PYTHONPATH=src python scripts/check_frontend_reproducible_build.py audit-readiness
```

退出码为 `0=qualified`、`2=reproducibility_or_runtime_violation`、
`3=not_qualified_upstream_limitation`、`4=usage_error`。

## 资格边界

当前固定提交的两份前端归档及其 56 个成员逐字节一致，并通过根路由、静态资源引用、RSC
hydration 和 TypeScript API 契约检查。该结论只证明前端发布构建可重复；完整软件发布候选仍因
11 条 Python 根依赖未精确锁定而保持 `build_or_supply_chain_violation`。它不包含质量指标，
也不解除 Full1000、人工 Precision 或官方 scorer 阻断。
