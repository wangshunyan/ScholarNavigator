# AutoScholarQuery 部分完成偏差审计

`completion_bias_audit_v1` 是严格离线、检索前特征限定的样本选择偏差审计。它将冻结的
AutoScholarQuery 1000 条有序清单、Record162 成员、Record160 主分析集合和既有两个排除项
按稳定身份精确闭合；任何重复、缺失或无法精确映射都会返回 `not_eligible`，不会使用文本近似
匹配补齐成员。

## 特征与统计边界

协议在读取分析结果前冻结。允许特征仅包括 query 的字符/token 长度、词汇与标点结构、
布尔/引号/年份/Unicode 模式、稳定顺序位置，以及既有查询连通分量。原始数据身份只用于精确
连接，输出立即替换为不可逆 opaque identity；gold、qrels、case ID、目标论文、检索结果、
来源产出和质量指标均不得加载或作为特征。

审计固定比较 Record160 与其余 840 条、Record162 与未记录 838 条，并对两个既有排除项仅作
描述。连续特征报告分布、标准化差异和 KS 距离，类别特征报告频数和 Jensen-Shannon 距离；
推断使用冻结连通分量作为 bootstrap 和置换单元并执行 Holm 校正。固定交叉验证的可区分性
诊断只度量这些预注册特征能否区分成员身份，不是因果分析，也不推断未完成查询的检索表现。

## 运行与验证

```bash
PYTHONPATH=src python scripts/check_completion_bias.py run \
  --protocol benchmark/completion_bias_audit_v1_protocol.json \
  --output benchmark/completion_bias_audit_v1_release

PYTHONPATH=src python scripts/check_completion_bias.py verify \
  --protocol benchmark/completion_bias_audit_v1_protocol.json \
  --output benchmark/completion_bias_audit_v1_release
```

退出码为 `0=completed`、`2=identity_or_analysis_violation`、
`3=not_eligible/insufficient_query_metadata`、`4=usage_error`。正常路径强制
0 网络、0 LLM、0 Snapshot 读写、0 gold/qrels 和 0 质量指标。

## 外推边界

Record160 的覆盖、来源、排序、约束和交付结论只适用于冻结的 160 条主分析查询。即使没有
检测到某项特征差异，也不能据此宣称它代表完整 1000 条；Full1000 未完成阻断保持不变。
审计产物只提供脱敏聚合、opaque 分层计数和确定性 manifest，不改写任何冻结运行或历史证据。
