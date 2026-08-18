# Top-20 交付保真检查清单

`top20_delivery_fidelity_v1` 只验证内部 `current_rules` 最终结果在已有产品出口中的数量、
身份、顺序和字段保真；它不读取 gold/qrels，不计算 Precision、Recall 或 F1，也不是官方
scorer 或官方提交格式。

## 权威边界

- 唯一权威输入是生产阶段 `final_returned`：完整 rerank 后先截取最多 20 条，再只保留
  `highly_relevant` 与 `partially_relevant`。
- “Top-20”是上限而非补齐目标；0–19 条均为合法结果，禁止复制或补造论文。
- 每条结果携带 `result_identity` 作为统一论文身份和前端 key，并携带
  `authority_digest` 绑定公共展示变换前的完整内部 `RankedPaper`。
- `year` 的未知状态用 JSON `null` 表示，不再以 `0` 冒充年份；空字符串、缺失字段、
  `null` 和前端占位符不可互相反写。
- 危险 URL 可在公共/前端展示边界失去可点击性，但论文不能被删除，原始权威字段仍由
  `authority_digest` 绑定。

## 现有出口

| 出口 | v1 资格 | 检查 |
| --- | --- | --- |
| FastAPI 公共响应 | 支持 | Pydantic Schema、身份、字段、顺序、数量 |
| 前端 JSON 下载 | 支持 | UTF-8 JSON round-trip |
| 批处理 JSONL | 支持 | 每行独立 JSON、LF、顺序与字段 round-trip |
| 前端展示 | 支持 | 直接消费 API 数组；React key 固定为 `result_identity` |
| CSV/表格 | `unsupported_export` | 仓库没有生产结果 CSV/表格出口，不伪造兼容性 |
| Record160 复现胶囊 | `not_eligible` | 旧运行缺少 `run_manifest_v1` 和提交代，不后验补造 |

CSV 公式前缀和 RFC4180 只作为未来出口的门禁夹具测试；这不构成一个已实现的 CSV
出口。机器可读形状示例位于
`benchmark/top20_delivery_contract_v1_sample.json`，仅说明内部契约，不推测比赛 schema。

## 运行与退出码

```bash
PYTHONPATH=src python scripts/check_top20_delivery_fidelity.py run \
  --contract benchmark/top20_delivery_contract_v1.json \
  --output /tmp/top20-delivery-fidelity
PYTHONPATH=src python scripts/check_top20_delivery_fidelity.py verify \
  --output /tmp/top20-delivery-fidelity
```

- `0`: 所有声明的出口均通过。
- `2`: 重建、字段、身份、顺序、分页或 round-trip 违规。
- `3`: 权威输入不具资格，或至少一个现有契约出口明确不支持/旧格式不具资格。
- `4`: 命令或输入用法错误。

当前 Record160 审计预期返回 `3`，因为 CSV 不存在且旧运行不能导出新式复现胶囊；这不
影响 API、JSON、JSONL 与前端四个支持出口的逐项保真结论，也不能被改写成全出口通过。

## 比赛交付前只读检查

1. 固定目标运行、Git commit、合同文件和输入哈希。
2. 确认 `final_returned` 数量、身份和顺序闭合，且每条不超过 20。
3. 只选择比赛官方明确发布的提交 schema；本门禁不提供或推断该 schema。
4. 对实际采用的出口运行 round-trip，并确认 `result_identity`、`authority_digest`、字段和
   顺序一致。
5. 明确记录 `unsupported_export` / `not_eligible`，不得用临时转换宣称生产兼容。
6. 不把交付保真结论描述为检索相关性、人工 Precision 或官方成绩。
