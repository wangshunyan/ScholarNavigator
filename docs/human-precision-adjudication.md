# 盲化人工 Precision 标签与裁决门禁

`human_precision_adjudication_v1` 是冻结盲标包的离线接收与完整性门禁。它不生成
人工标签，也不推断相关性；只有两轮独立标注、必要裁决和既有包引用标签全部闭合后，
才调用仓库已有的 change-only Precision 统计实现。

## 冻结协议

协议位于 `benchmark/human_precision_adjudication_v1_protocol.json`，绑定：

- Record160 全量 Top-20 变更包的版本、文件树 SHA-256、439 个 opaque item 的集合摘要；
- 既有 200 项盲标包及当前包引用的 32 个 opaque item；
- 冻结 rubric 的四个标签；
- `deduplicated_gold_identity_v2` 身份版本与
  `full_swap_precision_annotation_v1` 统计版本；
- 两位独立匿名标注者、仅分歧项可裁决、当前不支持 confidence、排除项固定为 0；
- 内部人工 Precision 范围，明确不是官方 scorer 成绩。

包摘要使用 `sorted_relative_path_size_sha256_v1`，按包内全部文件的相对路径、大小和
SHA-256 排序后计算；报告中的 opaque item 身份使用 `sha256_utf8_v1`。校验不会修改盲标包，
也不会把私有 mapping 内容写入错误报告。

## 输入契约与盲化边界

独立标注文件是单个 JSON 对象，顶层绑定协议与包摘要；标注者身份使用
`anon-` 前缀的 opaque identifier，不接收姓名或邮箱。`labels` 中只能出现
`item_id`、`label` 和可选 `notes`。裁决文件只允许 `item_id`、`final_label` 和可选
`rationale`。未知字段会被拒绝，因此 gold、qrels、case ID、策略名、来源、排名、
分数或额外论文身份不能随标签进入门禁。

既有包引用标签必须同时携带两位原始判断、最终标签和
`annotator_agreement/adjudicated` 解析状态；分歧时还要求独立裁决者。门禁报告保存各输入
文件 SHA-256 和逐 item 的 opaque identity SHA-256、两位原始标签、最终决议及解析方式，
不会回显原始 item ID 或标注正文。

## 状态与退出码

| 状态 | 退出码 | 含义 |
| --- | ---: | --- |
| `validated` | 0 | 覆盖、独立性、裁决和既有包引用全部闭合，可生成内部统计 |
| `invalid` | 2 | Schema、盲化、身份、哈希、覆盖或裁决完整性违规 |
| `awaiting_labels` | 3 | 独立标注或既有包已裁决标签尚未齐备 |
| `adjudication_required` | 3 | 两轮标注齐备，但仍有未裁决分歧 |
| `not_eligible` | 3 | 冻结 rubric 或包身份不完整，不能安全接收标签 |
| usage error | 4 | 命令或协议输入不可用 |

`insufficient_information` 是 rubric 中的人工标签，不等同于排除。当前协议不允许新增
排除理由。状态未达到 `validated` 时，`statistics` 必须为 `null`；不会输出零值占位。

## 使用

只读检查当前真实包（目前无人工标签，预期退出码 3）：

```bash
PYTHONPATH=src python scripts/check_human_precision_labels.py
```

标签齐备后：

```bash
PYTHONPATH=src python scripts/check_human_precision_labels.py \
  --annotator-one /secure/path/independent_1.json \
  --annotator-two /secure/path/independent_2.json \
  --adjudication /secure/path/adjudication.json \
  --prior-resolved /secure/path/prior_resolved.json \
  --output /secure/path/validated_report.json
```

提交的标签文件应保存在受控位置，不属于仓库冻结包或 evidence registry。快速离线门禁为：

```bash
PYTHONPATH=src pytest -q -m human_precision_adjudication_regression
```

标注包完整性只表示接收输入已准备好；人工裁决结果、内部 Precision 与官方 scorer
成绩是三个不同概念。当前真实包没有人工标签，因此没有 Precision、误放率或 kappa
结论。
