# 正式验证就绪证据包

`validation_readiness_bundle_v1` 是只读的证据发布与声明追溯门禁。它把仓库已经跟踪的协议、
聚合审计结果和阻断记录绑定为一个确定性目录，但不运行 Benchmark、Replay、LLM、检索 API
或 evaluator，也不生成新的质量指标。

## 契约边界

契约固定以下内容：

- 实现所基于的 Git commit，以及生成器、CLI、契约和文档的代码树摘要；
- 数据、Snapshot、冻结 Record 的已有摘要，不复制原始 query、论文正文或私有映射；
- 每份机器证据的仓库相对路径、大小、SHA-256、协议版本和依赖；
- README、架构、评测和比赛要求中的声明状态与机器证据引用；
- `current_rules` 默认状态、实验 tie-break 默认关闭状态和关键门禁退出码；
- Full1000、人工 Precision、官方 scorer/schema 三项不可替代的正式阻断。

包还登记 `completion_bias_audit_v1` 的只读证据。该证据精确闭合 Full1000、Record162 和
Record160，并将 Record160 的覆盖、来源、排序、约束与交付声明限制在冻结 160 条总体；它不
解除 Full1000 阻断，也不推断其余查询的检索表现。

`external_scorer_handoff_v1` 作为独立工程声明登记：严格合成 package 已验证 canonical
handoff、隔离子进程、输入不可变和双次确定性，但真实 readiness 仍以退出码 3 保持 blocked。
该声明不提供官方 Schema、指标或成绩，也不解除 Full1000 与官方 scorer 两项阻断。

`human_annotation_delivery_v1` 也仅登记为工程链路就绪：两套 471 项盲化包、operator-only
恢复映射和合成回收/裁决演练已经离线验证，但真实标注数仍为 0，统计保持 `null`。因此
`human_precision_missing` 阻断和 `formal_validation_complete=false` 均保持不变。

`validation_evidence_freshness_v1` 为当前声明、证据和只读门禁登记精确语义依赖与 basis digest。
readiness 一键验证会先确认当前基线仍新鲜；变更影响报告只给出受影响对象和最小重跑集合，不会
自动刷新证据或改写历史结论。新鲜度通过同样不解除 Full1000、人工 Precision 或官方 scorer
阻断。

`full1000_execution_readiness_v1` 只登记“执行计划就绪”工程声明：1000 条 query-only 输入、
分片、attempt、resume、资源理论上限和 1000-query 本地 fake dry-run 已确定性闭合。网络状态
仍为 `network_not_checked`，旧 Record160/162 不具 checkpoint 权威性；因此
`full1000_incomplete` 阻断和 `formal_validation_complete=false` 不变。

`formal_validation_clearance_v1` 将三项正式阻断的解除条件固化为机器状态机和 receipt 签发
门禁。当前 Full1000、人工 Precision 与官方 scorer 分别缺少完整真实外部证据，真实审计返回
exit 3；合成状态转换只存在于临时测试目录，不能签发正式完成凭证。因此 blocker 数量仍为 3，
`formal_validation_complete=false` 不变。

`formal_evidence_quarantine_v1` 已将未来人工标签、裁决和官方 scorer 输出限制为
evaluation/reporting/clearance 消费，并将 intake 后的检索、Prompt、预算、排序、默认策略或
数据身份变化传播为 `stale_for_claim`。这只证明隔离控制链就绪；当前不存在真实正式证据，
readiness 审计返回 exit 3，不改变三项正式阻断或历史 evidence 结论。

`public_contract_compatibility_v1` 将 FastAPI OpenAPI、前端消费类型、关键离线 CLI 以及
run/readiness/clearance/人工标注/scorer handoff 机器产物冻结为规范化语义基线。当前只读门禁
验证同版本无 breaking 漂移；它不定义官方 Schema 或质量指标，也不解除 Full1000、真实人工
Precision 和官方 scorer/schema 三项阻断。

`release_candidate_reproducibility_v1` 登记为发布工程门禁：固定源码、工具链、依赖闭包、SBOM
和双树构建均可离线复核。前端 webpack 的历史跨路径漂移已由独立资格门禁修复并逐成员验证；
完整候选仍因 Python 根依赖未精确锁定而为 `not_qualified`。该工程状态不会替代或解除任何正式
验证阻断。

`evidence_transparency_log_v1` 登记为公开证据历史防分叉工程门禁。当前 checkpoint 只含
candidate-only genesis，绑定固定提交中的 readiness、standalone、release candidate、
clearance、freshness 与 revocation Git blob；公开 release 数为 0。该状态证明追加链、Merkle
inclusion/consistency proof 和阻断披露控制可复核，但不认证发布者身份，不改变发布候选资格，
也不解除 Full1000、真实人工 Precision 或官方 scorer/schema 三项阻断。

`formal_run_storage_governance_v1` 仅登记“正式运行存储控制就绪”工程声明。冻结计划的
HTTP attempt、提交代和 shard 上限已映射为响应、代、shard、运行和备份链配额，注入式
1000-query 压力演练验证低空间回滚、超大响应 fail-closed、保留窗口和单 writer 约束。
真实主盘与备份盘的可用空间、inode 和文件系统配额仍未提供，因此 readiness 保持
`not_ready_capacity_unverified`，不会解除 Full1000 未完成阻断。

`formal_execution_host_attestation_v1` 仅登记“正式执行主机资格门禁就绪”工程声明。
合成 profile 证明能力判定、封印失效和启动绑定会 fail closed；当前真实主机封印仍因
主盘容量不足、配额及备份故障域观测缺失而返回 exit 3。该状态不会解除 Full1000、
人工 Precision 或官方 scorer/schema 阻断。

`formal_provider_health_supervisor_v1` 仅登记“长周期来源健康监督与安全暂停就绪”
工程声明。1000-query fake 故障矩阵验证持续 429/503/timeout、单源/多源退化、无进展、
预算燃烧、容量/取证故障、在途排空与恢复；真实来源健康尚未观测，readiness 返回
`external_provider_health_not_observed`。该状态不会解除 Full1000、人工 Precision 或
官方 scorer/schema 阻断，也不产生质量指标。

`formal_scheduler_fairness_v1` 仅登记“Full1000 并发调度公平性与背压控制就绪”工程声明。
1000-query 注入式负载矩阵验证慢源、慢 shard、retry/page 风暴、worker 缩减、暂停/恢复和
取消下的有限服务、并发/attempt 守恒与完整终态覆盖；当前正式运行尚未启动，readiness 返回
`external_run_not_started`。该状态不会解除 Full1000、人工 Precision 或官方 scorer/schema
阻断，也不是检索质量或真实性能结论。

`formal_provider_pacing_v1` 仅登记“Full1000 来源容量与确定性节流控制就绪”工程声明。
1000-query/9,640-intent 合成矩阵验证来源独立 token bucket、分钟窗口、并发、首次请求
公平屏障、Retry-After、bounded retry、pause/resume 和账本守恒；真实四源容量声明仍全部
为 `not_available`，`audit-readiness` 返回 exit 3。该状态不解除 Full1000、人工
Precision 或官方 scorer/schema 阻断，也不代表真实性能或质量。

`provider_capacity_declaration_intake_v1` 仅登记“来源容量声明离线接收链路就绪”工程声明。
标准库 kit、结构化 schema、一次性 challenge、append-only 导入账本和请求集合守恒已由
合成声明验证；真实 OpenAlex、arXiv、Semantic Scholar、PubMed 声明仍全部缺失，
`audit-readiness` 稳定返回 exit 3。该状态不会授权启动 Full1000，不会改变三项正式阻断
或 `formal_validation_complete=false`。

`release_authenticity_signing_v1` 仅登记“发布身份签名控制就绪”工程声明。它证明
OpenSSH Ed25519 规范 envelope、测试密钥隔离、信任根轮换/撤销与离线验证链路；
当前没有真实发布信任锚或签名者，真实审计保持 exit 3。该声明不把候选 checkpoint
标记为正式发布，也不改变三项正式阻断或 `formal_validation_complete=false`。

`formal_validation_separation_of_duties_v1` 仅登记“正式验证职责分离控制就绪”工程声明。
12 个角色、禁止组合、opaque principal/alias 绑定、append-only 授权与操作链以及双批准域
已由合成仪式验证；当前没有真实角色分配，readiness 稳定返回 exit 3。合成身份不能授权
正式 launch、人工裁决、clearance、签名或撤销，也不改变三项正式阻断或
`formal_validation_complete=false`。

`change_risk_validation_v1` 登记“开发验证范围已机器化编排”工程声明。它对 freshness
组件、公共契约、readiness gate 和测试入口建立 fail-closed 风险映射，能够自动要求定向
pytest、相关只读 gate、确定性、安全扫描及仓库检查，并仅在 high/异常条件下升级完整
pytest、仅在 frontend/API/build 变更时要求前端构建。该工程控制不生成质量指标，也不改变
三项正式阻断。

`formal_backup_target_registration_v1` 登记“备份目标显式登记与本地预检入口就绪”
工程声明。私有登记文件不进入仓库，门禁只访问操作者列出的精确目录，并复用既有
target attestation 的原子写、fsync、锁和清理探针；输出仅保留脱敏 identity 与路径
绑定哈希。当前没有真实私有登记文件，readiness 返回 exit 3；候选也仍需通过 quota、
独立故障域、target attestation 和 member intake，不解除 Full1000 或其他正式阻断。

`formal_backup_member_enrollment_v1` 登记“真实备份成员端到端离线入驻链路就绪”
工程声明。2/3/4 成员的逐 slot kit 复用既有目标探测与 member intake 契约，且能在
无仓库 `python -I -S` 环境输出待 intake 候选包。当前真实入驻成员仍为 0，各方案
分别缺少 2/3/4 个 slot，因此不改变 Full1000 或其余正式验证阻断。

声明状态只有 `verified`、`internal_only`、`blocked` 和 `not_applicable`。`verified` 仅用于
工程能力，`internal_only` 仅用于内部冻结验证或诊断；正式验证要求在缺失外部输入时必须是
`blocked`。覆盖、稳定性、来源漏斗、LLM proxy 或交付保真都不能代替这些阻断。

## 生成与一键验证

```bash
PYTHONPATH=src python scripts/check_validation_readiness.py generate \
  --contract benchmark/validation_readiness_bundle_v1_contract.json \
  --bundle benchmark/validation_readiness_bundle_v1_release
```

```bash
PYTHONPATH=src python scripts/check_validation_readiness.py verify \
  --contract benchmark/validation_readiness_bundle_v1_contract.json \
  --bundle benchmark/validation_readiness_bundle_v1_release
```

`verify` 只调用登记过的既有 `verify/check` 入口，校验历史哈希、协议依赖、跨证据计数、声明
边界、默认关闭项和既存嵌套工作树状态。它不调用任何会补采、付费、联网或写 Snapshot 的
`run/generate` 路径。

一键验证同时读取 `evidence_revocation_response_v1` 的权威空账本。活动撤销事件会使
readiness 生成与验证返回阻断状态，并列出下游声明和最小重跑 gate；删除或改写历史事件不能
恢复发布资格。

退出码：

- `0`：`ready_with_declared_blockers`，包完整且阻断已明确保留；
- `2`：证据哈希、字段、声明、默认状态或发布包内容违规；
- `3`：必需证据缺失，无法建立就绪包；
- `4`：命令使用错误。

## 与其他证明的区别

本门禁只证明“声明能否追溯到未篡改的内部证据，以及限制是否完整披露”。它不证明检索
相关性，不是人工 Precision，不是官方 scorer，也不生成官方提交。历史证据由原门禁继续
负责，证据包只登记其原始哈希，不重写历史文件或结论。

`formal_multivolume_storage_v1` 登记“Full1000 多卷分片存储控制就绪”工程声明。
合成 1000-query 演练证明逐卷配额、原子边界、resume、受控迁移与只引用 aggregate
可以闭合；当前真实主机仍因缺少显式 quota、合格附加卷及独立备份故障域返回 exit 3。
这不解除 Full1000 未完成、真实人工 Precision 或官方 scorer/schema 三项阻断。

`formal_shard_streaming_retention_v1` 登记“Full1000 分片流式归档与本地释放
控制就绪”工程声明。当前主盘在预注册窗口 1/2/4 下均满足确定性容量上限，但
合格备份目标的容量、inode、quota 和独立故障域仍不可证明，因此 readiness 保持
exit 3，`formal_validation_complete=false` 且三项正式阻断不变。

`portable_execution_site_attestation_v1` 登记“可移执行节点资格包与封印导入控制就绪”
工程声明。标准库资格包、一次性 challenge、逐卷 fail-closed 探测和启动回执已由两个
无仓库环境及合成节点矩阵验证；当前没有真实外部节点证明，readiness 返回 exit 3。
该状态不解除 Full1000 未完成、真实人工 Precision 或官方 scorer/schema 三项阻断。

`formal_network_request_manifest_v1` 登记“Full1000 正式请求输入清单就绪”工程声明。
生产请求构造路径已离线闭合 1000 条 query、2410 个 subquery、9640 个逻辑来源请求和
19280 次理论 HTTP attempt 上限，并将历史 1093 个 Snapshot key 明确隔离为
`historical_reference_only`。网络仍未检查且正式运行未启动，因此 Full1000 未完成、
真实人工 Precision、官方 scorer/schema 三项阻断和
`formal_validation_complete=false` 均保持不变。

`formal_backup_target_attestation_v1` 登记“独立备份目标资格与故障域导入控制就绪”
工程声明。当前真实状态仍为 `not_ready_no_qualified_backup_target`：容量、inode、
quota 及独立故障域尚无 fresh 真实封印，因此流式保留、灾难恢复、host/site 和
launch 复审均未通过，Full1000 未完成阻断保持不变。

`formal_backup_compaction_v1` 登记“Full1000 内容寻址增量备份压实控制就绪”工程
声明。新最坏情况门槛为 1,028,812,963,840 bytes 和 108,137 inode，所有压缩、
稀疏文件、未来重复率及未来清理抵扣均为 0。合成 1000-query 演练验证旧 root
保留、中断回退、恢复后零重复请求、账本守恒和 aggregate 等价；当前仍缺满足容量、
inode、quota 与独立故障域要求的真实备份目标，readiness 返回 exit 3，三项正式
阻断和 `formal_validation_complete=false` 不变。

`formal_backup_set_topology_v1` 登记“Full1000 多目标备份集容量拆分控制就绪”
工程证明：2/3/4 member 的确定性分配、逐成员最坏情况容量、quota pool 防重复
计容、完整集合恢复和 1000-query 零重复请求演练均可离线复核。当前没有真实
合格 member set，readiness 仍为 `not_ready_missing_qualified_members`，三项
正式验证阻断及 `formal_validation_complete=false` 不变。

`formal_backup_set_member_intake_v1` 登记“Full1000 真实备份集成员离线接收与
激活控制就绪”工程声明。逐槽位 kit、一次性 challenge、append-only 登记链、
防设备/quota/故障域重复计容及完整恢复通过合成演练；当前 2/3/4 成员方案仍分别
缺少 2/3/4 个真实槽位，`audit-readiness` 返回
`not_ready_missing_real_members`。该工程状态不授权 launch，也不改变三项正式阻断。

`official_scorer_package_intake_v1` 登记“官方 scorer 离线接收与沙箱 conformance
控制就绪”工程声明。11 项合成矩阵只证明 package/Schema/namespace 接收、安全导入、
防重放和既有沙箱复用；当前没有真实官方 package、I/O Schema、指标 namespace 或可验证
来源，readiness 返回 exit 3。Full1000、人工 Precision、官方 scorer/schema 三项正式
阻断和 `formal_validation_complete=false` 均保持不变。

`human_annotator_qualification_intake_v1` 登记“真实人工角色资格离线接收控制就绪”
工程声明。合成校准与冲突矩阵不接触真实盲标内容；当前 annotator-A、annotator-B 和
adjudicator 的真实匿名资格均缺失，因此 readiness 返回 exit 3。角色提案不会自动分发
真实包，人工 Precision 阻断及另外两项正式阻断保持不变。

`human_annotation_assignment_activation_v1` 登记“真实人工标注任务分配与盲包签发
控制就绪”工程声明。A/B 包、裁决者 rubric 包、一次性 receipt 和 append-only 状态链均
由合成资格演练验证；真实 A/B/adjudicator 资格与三份确认仍缺失，因此 readiness 返回
exit 3，真实包未签发、标签导入未启用，三项正式阻断和
`formal_validation_complete=false` 保持不变。

`human_annotation_submission_intake_v1` 登记“真实人工标注提交接收与裁决队列控制
就绪”工程声明。14 项合成矩阵验证完整覆盖、锁定摘要、角色绑定、撤销/重签发和
全部分歧队列；当前真实三角色资格、签发确认及 A/B 锁定提交仍缺失，因此 readiness
返回 exit 3，不生成标签、统计或裁决结果。Full1000、人工 Precision、官方
scorer/schema 三项正式阻断和 `formal_validation_complete=false` 保持不变。
## 人工裁决激活

`human_adjudication_activation_v1` 已用 13 个纯合成场景验证分歧包签发、裁决者
绑定、完整结果接收、撤销/篡改拒绝及既有 change-only scorer 解锁。该工程能力不
代表真实人工标签已到位；真实 readiness 仍缺 A/B/裁决者资格、三方 acknowledgement、
A/B 锁定提交、真实分歧队列和裁决提交，三项正式阻断不变。

## 外部阻断行动包

`formal_external_blocker_action_pack_v1` 登记“三条外部接收链已形成唯一行动交接”
工程声明。当前三条链均无未登记工程缺口，但真实备份成员、完整真实人工证据和经
验证的官方 scorer 材料仍缺失，因此 `audit-readiness` 返回 exit 3，开发冻结状态为
`external_action_required`。三项正式阻断及 `formal_validation_complete=false` 不变。
