# 正式验证预注册封印

`formal_validation_preregistration_v1` 在任何真实 Full1000、双人标签或官方 scorer
证据进入仓库前，冻结正式分析的样本、执行配置、停止条件、统计方法和声明边界。封印只证明
登记内容及依赖文件未变化，不证明发布者身份，也不生成 Precision、Recall 或官方成绩。

## 冻结范围

- Full1000 必须按冻结的 1000 条稳定身份与顺序从头执行；Record160/162 不能作为 checkpoint。
- 默认策略为 `current_rules`，四个来源、预算、Prompt 摘要、随机种子、evaluator 固定，
  `deterministic_tiebreak_v2` 与其他实验能力关闭。
- 人工阶段要求 A/B 两位独立匿名标注者完整覆盖 471 项，标签枚举和分歧裁决沿用
  `human_precision_adjudication_v1`；覆盖不完整时停止评分，不以零或不相关代替。
- change-only 配对以冻结查询连通分量为重采样单位，固定 20,000 次 bootstrap、
  双侧 cluster sign-flip、95% 区间、Holm–Bonferroni 校正与 6 位报告精度。
- 官方 scorer 的 package、Schema、指标命名空间与方向保持 `unknown/not_provided`，
  在权威材料到达前不得推测。

失败、取消、部分完成和不可评审项必须保留明确终态；禁止只分析成功项、删除失败项、
后验新增指标或切换主次终点。任何 Record160 结论不得外推至未完成的 840 条查询。

## 修订与证据边界

状态机为 `sealed → amended_before_evidence → invalid_post_evidence_change`。真实证据 intake
后，样本、阈值、统计方法、排除规则、默认策略或声明边界发生语义变化，会使对应正式声明
失效。仅 `/documentation/` 下且机器证明语义摘要不变的勘误可以维持 `sealed`。
预注册必须早于证据 intake，执行必须早于解盲或评分。

## 只读命令

```bash
PYTHONPATH=src python scripts/check_formal_validation_preregistration.py verify
PYTHONPATH=src python scripts/check_formal_validation_preregistration.py simulate-amendment
PYTHONPATH=src python scripts/check_formal_validation_preregistration.py audit-readiness
```

`verify` 成功返回 0；当前 `audit-readiness` 必须返回 3，因为 Full1000、真实人工
Precision 与官方 scorer/schema 三项外部证据仍缺失。退出码 2 表示封印、依赖或后验
修改违规，4 表示命令用法错误。
