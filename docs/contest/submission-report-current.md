# ScholarNavigator 赛题三说明书（当前证据版草稿）

> 这是与当前仓库证据一致的参赛材料底稿。团队信息、官方评测结果和平台要求仍需在提交前补充；本文不把内部工程指标写成赛事官方成绩。

## 1. 项目定位

ScholarNavigator 面向“科研场景下复杂学术查询的智能论文搜索与推荐”。用户可以在一句自然语言中同时提出主题、方法、数据集、时间范围、会议/期刊、排除条件和证据要求。系统将查询解析为结构化约束，生成受预算控制的检索计划，返回论文、匹配理由、证据边界、引用关系和成本诊断。

## 2. 技术流程

1. 规则查询理解保留原始查询，并抽取时间、方法、数据集、领域、论文类型、venue、必含词和排除词。
2. 规划器在固定预算内生成原查询及受控扩展查询；LLM 查询理解和判断默认关闭，Provider 不可用时保留规则路径。
3. 本地 BM25 与 BGE 语义向量检索分别召回候选，使用稳定 arXiv ID 去重并以 RRF 融合；候选池受预算限制。
4. Judgement 检查硬约束并给出相关性类别；排序理由、证据来源和缺失字段随结果返回。
5. 仅在显式配置且通过资源门禁时使用 Qwen3-Reranker；当前成对证据未支持默认启用，因此默认关闭。
6. 结果页展示查询理解、阶段进度、证据定位、引用图、导出内容和 API/Token/延迟/失败诊断。

## 3. 数据与安全边界

当前可审计的 v3 脱敏语料包含 569,432 条唯一 arXiv 记录，SHA-256 为 `7a385c87250ff438f5748cc49ee683acf1edd01d2f12432d17fe60e83908a31a`。标题、摘要、作者、年份完整度为 100%；venue 为 12.152%，DOI 为 17.897%。语料使用稳定 arXiv ID 精确关联，不使用标题模糊匹配，也不使用 AutoScholarQuery gold/qrels 生成在线语料、索引、Prompt 或排序输入。

公开 GitHub source-only 包不包含服务器原始运行目录、模型、Faiss/BM25 索引、`.env`、密钥或临时输出。gold/qrels 只允许在检索完成后的离线 evaluator 与误差诊断中使用。

## 4. 当前可核验的 200 条内部工程证据

两组 v3 Hybrid 运行固定同一查询顺序、语料、索引、预算和 profile，均完成 200/200，失败 0。指标口径为内部离线工程指标，不是赛事官方 scorer。

| 配置 | Recall@20 | Precision@20 | F1@20 | MRR | 平均延迟 | 成功率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid baseline | 0.12850 | 0.01200 | 0.02108 | 0.03974 | 2.558 s | 1.000 |
| Hybrid + Qwen3 Reranker | 0.12797 | 0.01225 | 0.02137 | 0.03649 | 7.580 s | 1.000 |

严格配对 bootstrap 的 Reranker 差值为：Recall@20 `-0.00054`（95% CI `[-0.00987, 0.00838]`），F1@20 `+0.00029`（95% CI `[-0.00219, 0.00273]`），F1@10 `-0.00475`（95% CI `[-0.00875,-0.00143]`）。没有指标满足“均值为正且置信区间下界大于 0”的启用门槛，因此默认方案保持 Hybrid、关闭 Reranker。上述运行资产保存在本地被忽略的服务器脱敏证据目录，不进入 GitHub 发布包。

## 5. 演示与复现

评委可用 `docs/contest/demo-script.md` 和 `docs/contest/demo-queries.md` 演示复杂查询、阶段进度、证据定位、引用图、Markdown/JSON 导出、成本统计和失败降级。当前 clean-clone smoke 已验证：health/config=200、离线 BM25 返回 5 条、网络请求 0、LLM disabled、gold/qrels 未加载；source-only 包约 37 MB，低于 200 MB 源码包上限。

推荐现场查询：

- `找 2021 年以后关于扩散模型用于医学图像分割的论文，要求结果里说明数据集和评价指标。`
- `Find papers after 2022 that use retrieval-augmented generation for multi-hop question answering and report results on HotpotQA.`
- `Search for graph neural network papers on molecular property prediction that evaluate on MoleculeNet and exclude review papers.`

## 6. 创新点与已知限制

创新点是复杂约束的可解释解析、BM25+语义的受控混合召回、稳定身份去重、证据边界展示和从检索到成本的可审计闭环。当前限制也必须在答辩中主动说明：正式 venue/DOI 全字段门禁尚未通过；没有可核验的官方 scorer 结果；LLM Provider 默认关闭；受许可控制的全文获取与定位链路已实现，但开放全文覆盖和独立 Evidence F1 尚未完成；独立撤稿/重复风险来源尚未提供；服务器实验与本地的一致性只以脱敏 evidence bundle 证明，不能由 Git 同步推断。

## 7. 提交前必做事项

- 获得合法、可哈希、字段完整的元数据输入后，重新构建索引并完成 200 条资格门禁。
- 若 200 条资格通过，再按同一代码/输入/资源账本启动 1000 条正式运行并接入官方 scorer。
- 补齐正式运行、GPU/模型指纹、资源账本、许可证与开放全文覆盖证据。
- 填写团队信息、硬件/软件版本、数据许可、成本和最终官方指标；保留当前“内部指标/非官方成绩”分栏。
