# 验证证据新鲜度与变更影响

`validation_evidence_freshness_v1` 是只读的证据依赖与影响传播门禁。它不运行 Benchmark、
evaluator、LLM 或检索，也不修改历史证据；它只判断已发布证据和声明所依赖的代码、协议、
配置、数据身份、前端转换、CLI 与统计实现是否仍与冻结基线一致。

## 依赖和状态

依赖契约以精确文件组成版本化组件，不允许用整个仓库作为依赖。每份机器证据登记 artifact
SHA-256、基线提交、组件集合和 `evidence_basis_digest`；门禁和声明分别登记组件、引用证据及
声明源文档。依赖缺失、重命名、环、未登记生产文件变化或基线早于实现都会失败。

状态含义：

- `fresh`：证据基础与冻结摘要一致；
- `stale`：语义依赖、artifact 或上游证据发生变化，需要按最小门禁集合重跑；
- `blocked`：证据本身是已声明的外部输入阻断，且依赖仍新鲜；
- `not_applicable`：契约明确说明不适用。

Python 文件以去除注释和 docstring 后的 AST 比较，JSON 以解析后的规范结构比较；因此注册
文件的纯注释变化不会制造大范围失效。测试目录只有在契约明确的 `test_only` 豁免下不传播。
重命名已登记依赖始终视为语义变化，未登记生产文件则 fail closed。

## CLI

```bash
PYTHONPATH=src python scripts/check_validation_freshness.py verify-current
PYTHONPATH=src python scripts/check_validation_freshness.py impact --from <commit> --to <commit>
PYTHONPATH=src python scripts/check_validation_freshness.py impact-worktree
PYTHONPATH=src python scripts/check_validation_freshness.py audit-release
```

退出码：0 表示 `fresh_with_declared_blockers`，2 表示过期或依赖违规，3 表示缺少基线，4 表示
用法错误。门禁输出的是重跑计划，不会重算或替换历史指标，也不会解除 Full1000、真实人工
Precision 和官方 scorer/schema 三项正式阻断。
