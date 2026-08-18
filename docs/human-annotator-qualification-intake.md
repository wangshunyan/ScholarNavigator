# Human annotator qualification intake

`human_annotator_qualification_intake_v1` 为未来两位独立标注者及一位裁决者提供离线
资格接收链。它不分发真实 471 项盲标包，不生成论文相关性标签、Precision 或官方成绩。

## 离线资格包

三种角色分别使用一次性 challenge 生成 kit：`annotator_a`、`annotator_b` 和
`adjudicator`。kit 只依赖 Python 标准库，可在无仓库环境用 `python -I -S` 校验。
其校准内容是 6 个与真实论文无关的合成案例，不含真实 query、论文、全局 opaque item
identity、arm、策略、来源、排名、gold/qrels、case ID 或私有映射。

资格提交只允许：

- `prn_` 前缀的匿名 principal identity 和稳定 principal commitment；
- 申请角色、challenge、协议、提交和 rubric 绑定；
- 结构化独立性、利益冲突、数据处理及本人提交声明；
- 合成校准标签、受限 notes、锁定摘要和资格哈希。

姓名、邮箱、单位、用户名、主机身份和凭据均不允许进入提交。哈希证明内容绑定，不认证
自然人身份；真实身份映射属于外部治理输入。

## 角色与导入边界

A、B 和裁决者必须是三个不同 principal，也必须具有三个不同 commitment，防止更换 alias
绕过冲突。A/B 不得担任裁决者，裁决者不得提交原始判断，包协调者不得代签。challenge
只能消费一次，资格不得事后补签、跨协议/提交复用、撤销后复用或过期后继续使用。

三角色全部 qualified 后只生成 append-only `ready_for_real_assignment` proposal。它不会
自动分发真实包，也不会解除人工 Precision 阻断；真实双人完整标注、合法裁决及既有
`human_precision_adjudication_v1` 验证仍是后续前置条件。

```bash
PYTHONPATH=src python scripts/check_human_annotator_qualification.py \
  build-kit --role annotator_a --challenge <sha256> --output <kit.zip>
PYTHONPATH=src python scripts/check_human_annotator_qualification.py simulate-matrix
PYTHONPATH=src python scripts/check_human_annotator_qualification.py audit-readiness
```

退出码为 `0=annotator_roles_qualified`、`2=qualification_or_role_violation`、
`3=not_ready_missing_real_qualified_principals`、`4=usage_error`。当前真实审计返回 3：
真实 annotator-A、annotator-B 和 adjudicator 资格均未提供，三项正式验证阻断不变。
