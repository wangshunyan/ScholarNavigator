# ScholarNavigator 项目说明书草稿

> 提交前填写方括号中的团队信息，并仅使用 `outputs/benchmark_runs/` 的真实完整运行结果。

## 项目概述

**项目名称：** ScholarNavigator  
**赛题：** 企业赛题三，科研场景下复杂学术查询的智能论文搜索与推荐  
**团队：** [团队名称]  
**成员与分工：** [成员及分工]  

ScholarNavigator 面向带有研究领域、方法、数据集、时间、会议和排除条件的复杂学术查询，提供查询理解、多子查询检索、跨源去重、相关性判断、排序和结构化归纳。系统将本地 PaSa 标题 BM25 索引与公开 arXiv 摘要语义向量索引组合，在控制 API 调用、Token 和延迟的同时输出论文列表、证据说明、引用关系图和可导出结果。

## 问题与应用场景

科研人员常以自然语言描述复杂约束，而不是直接给出标准关键词。系统需要识别这些约束，生成可执行的检索子查询，在多个来源中获得候选论文，并解释每篇论文与用户需求的匹配依据。

适用场景包括文献调研、研究选题、方法/数据集追踪、跨领域论文发现和论文推荐。

## 系统方案

```mermaid
flowchart LR
    Q["复杂自然语言查询"] --> U["规则查询理解与约束提取"]
    U --> P["子查询规划与安全改写"]
    P --> L["PaSa 本地 SQLite FTS5 BM25"]
    P --> A["公开 arXiv 摘要 BGE 向量检索"]
    L --> F["RRF 融合"]
    A --> F
    F --> D["跨源去重与候选预算"]
    D --> J["相关性判断与硬约束保护"]
    J --> R["综合排序"]
    R --> O["论文列表、证据、引用图、导出与成本诊断"]
```

### 核心设计

1. **复杂查询理解**：识别方法、数据集、领域、时间、venue、论文类型和排除条件，生成受预算约束的子查询。
2. **混合检索**：PaSa 本地标题库提供低延迟 BM25 召回；公开 arXiv 标题+摘要子集提供 BGE 语义向量召回；RRF 融合降低单一路径漏召风险。
3. **标题库适配**：本地查询侧过滤自然语言填充词，避免礼貌语和泛化词主导标题检索；短查询与专名不被过滤为空。
4. **可解释排序**：返回匹配术语、标题/摘要/元数据证据、相关性类别和约束判定。
5. **成本可审计**：记录每次 API、重试、缓存、Token、延迟和失败，Benchmark 额外输出资源账本。

## 数据集与评测

主评测使用 PaSa/AutoScholarQuery。仓库中的公开文件包含 1000 条查询和 2403 个 arXiv gold 标识。PaSa 官方 `id2paper.json` 已转换为 569,432 篇本地标题语料；正式语义语料由 Cornell/arXiv 官方元数据按规范化 arXiv ID 精确关联得到同规模 title+abstract 语料。关联不使用标题匹配，也不使用 `AutoScholarQuery_test.jsonl`、gold 或 qrels 构建索引。旧 31,136 条标题匹配摘要子集仅保留为 legacy 证据，不进入正式成绩。

指标：F1@20、Precision@20、Recall@20、MRR、nDCG、成功率、API 调用数、Token、延迟和失败率。

## 实验设置

| 配置 | 检索源 | 查询适配 | 判断与排序 |
| --- | --- | --- | --- |
| local baseline | local BM25 | adaptive | current_rules |
| hybrid candidate | local_hybrid: BM25 + BGE semantic + RRF | adaptive | current_rules |

完整实验命令与恢复方式见 `docs/contest/experiment-protocol.md`。

## 已完成的内部工程证据

下表只列出已完成、可追溯的 P0/Faiss 1000 条内部运行。内部 F1/Recall 不等同赛事官方 scorer，也不代表隐藏测试集、官方排名或获奖结果。

| 配置 | 运行目录 | F1@20 | P@20 | R@20 | MRR | 平均 API | 平均 Token | 平均延迟 | 成功率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rules | `contest_full_rules_v1` | 0.01087 | 0.00620 | 0.06195 | 0.04071 | 0.0 | 0.0 | 0.719 s | 1.000 |
| Dense | `contest_full_dense_v1` | 0.02155 | 0.01225 | 0.13508 | 0.09171 | 0.0 | 0.0 | 0.968 s | 1.000 |
| Dense + reranker | `contest_full_dense_reranker_v4` | 0.02442 | 0.01390 | 0.15010 | 0.09406 | 0.0 | 0.0 | 3.909 s | 1.000 |

阶段诊断显示，reranker v4 的初始候选 Recall 为 0.29675，Judgement 后 Recall 为 0.19454，最终 Recall@20 为 0.15010。已检索 gold 中有 197 个在 Judgement 阶段被过滤、105 个在 Top-20 外，故软 Judgement 作为独立、默认关闭的受控候选进行资格验证。`contest_qual200_dense_reranker_soft_v2` 已通过 200 条配对资格门禁；其完整 1000 条运行未完成前不得在此表添加任何 soft Judgement 指标。

## 创新点与边界

- 本地 PaSa 标题 BM25 与按 arXiv ID 精确关联的摘要 BGE 向量检索的受控混合召回。
- 从查询理解、检索调用到结构化结果的端到端可观测性，便于复现实验与成本分析。
- 结果级证据链、引用图与导出，支持科研人员复核推荐理由。
- Qwen3-Reranker-0.6B 使用固定官方判定模板、2048 token 上限、batch=8 与显式 GPU 隔离；soft Judgement 仅改变一个预先声明的阈值并通过配对 bootstrap 决策。

当前限制：旧 `local_hybrid` 标题匹配语料只能作为 legacy 对照；LLM v5-v16 均为诊断，尚无零 fallback、完整审计通过的 1000 条 LLM 结果。soft Judgement 完整运行尚未完成，不能提前声明质量提升。Linux/Python 3.12 离线 wheelhouse 验证与 `record160` 历史冻结证据仍是最终 release tag 的阻塞项。

## 演示与复现

使用 VSCode 任务 `ScholarNavigator: run app` 启动系统。录制 3 到 5 条 `docs/contest/demo-queries.md` 中的复杂查询，展示输入、检索进度、结果证据、引用图、导出和运行成本。
