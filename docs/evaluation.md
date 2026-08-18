# 评测说明

## 统一口径

离线评测与批量评测共同使用 `scholar_agent.evaluation` 中的结果选择、论文匹配和指标实现。

默认结果策略为 `highly_and_partial`：先取 `highly_relevant`，再取 `partially_relevant`，各类别内按原始 rank 稳定排序。`highly_only` 只取高度相关论文；弱相关、不相关和证据不足论文均不进入正式列表。

论文匹配提取双方全部稳定标识符，任意交集即匹配。支持 DOI、arXiv（忽略版本号，并识别 arXiv DOI）、OpenAlex、Semantic Scholar 和 PubMed 的常见 URL 或前缀。只有双方均无稳定标识符时才按规范化标题和年份匹配。重复预测只计一次，一条预测最多匹配一个 gold。

核心排名指标为 F1@K，同时输出 Precision@K、Recall@K、MRR 和 nDCG@K；默认 K 为 5、10、20。Precision@K 的分母固定为 K。

### 官方计分器对齐门槛

2026-07-21 对用户提供的三页官方赛题 DOCX、仓库当前规范、历史评分说明以及
本地留存的 AstaBench/PaSa 官方参考仓库进行了纯离线审计。赛题 DOCX 只规定
F1 Score、运行效率和结构化回复分别占自动评分的 70%/20%/10%，没有提供
F1 精确公式、K、输入格式、论文身份与重复处理、未知文档、不可评估 gold 或
宏/微平均规则，也没有附带 scorer。AstaBench 和 PaSa 在文档中仅被列为参考
数据集/系统；两者现有计分实现语义也不同，不能任选其一充当赛题官方 scorer。

因此当前统一 F1@K 只能继续作为项目内部冻结指标，尚不能声明可直接用于赛题
成绩判断。本轮没有实现 scorer adapter，也没有运行 AutoScholarQuery/SciFact
对齐；获得版本化官方 scorer 或完整官方计分规范前，不得推测差异为零。来源
哈希、缺失语义和被排除的参考实现记录在
`benchmark/official_scorer_alignment_blocked.json`。

跨来源论文身份由 `scholar_agent.core.identity` 统一解析，并同时用于候选去重、API 结构化 ID 和离线 gold 匹配。DOI、arXiv（忽略版本号）、OpenAlex、Semantic Scholar 与 PubMed 的规范化稳定标识相交时直接判定同一论文；没有共同稳定标识时，只有规范标题、年份和至少一名共同作者同时一致才合并。稳定标识冲突始终保持分离。规则不使用 gold、查询文本或相似度阈值；去重函数为每次合并保留规则、共享标识和冲突标识审计证据。

聚合报告同时包含：

- `success_only_metrics`：仅统计成功且有有效 gold 的案例。
- `end_to_end_metrics`：失败、超时、取消、结果缺失或非法但有 gold 的案例按零分计入。
- `case_statistics`：记录总数、成功、失败、结果缺失、gold 缺失及对应比率。
- `efficiency`：直接读取 SearchService/API 的真实 CostReport，输出平均总 API、检索 API、引用 API、重试、错误、缓存命中、限流等待、LLM 调用和 LLM Token；不再通过 `source_stats` 条数推算请求数。

没有有效 gold 的 batch case 记录在 `missing_gold_cases`，不进入两套指标分母；gold 存在但 batch 缺失的案例记录在 `missing_result_cases`，并以零分进入端到端指标。

### 受约束 LLM 改写的离线因果审计

`scripts/audit_llm_constrained_rewrite.py` 只读取成对的 Benchmark Replay 与
Retrieval Snapshot，不初始化 SearchService、connector 或 LLM。审计按
`(case, source)` 重建原始查询、被替换派生查询和已接受改写的实际列表；只有
三者已执行请求均成功，且基线/实验中的原始查询终态与候选身份顺序一致时，
才计算独立候选、独立 gold 与反事实结果。fallback、校验拒绝、失败请求、
原始列表漂移均单独标记为不可归因。

反事实以基线去重后候选为冻结底座：先移除仅由被替换查询贡献的候选，再加入
改写列表，继续使用统一身份、`current_rules` judgement/reranker、候选预算、
正式结果过滤与 Top-20 指标。审计同时提供逐源单因素反事实；它们是诊断结果，
不得描述为线上可实现成绩。输出不含时间戳，两次相同输入应产生字节一致的
`case_audit.jsonl` 与 `aggregate.json`。

### 候选排序信号可分性审计

`scripts/audit_candidate_ranking_signals.py` 只读 SciFact local_bm25 与
AutoScholarQuery dev/val 的冻结 Replay。候选池固定为统一身份去重、全局候选
预算之后且相关性过滤之前的 `initial_reranked` 阶段；工具先从冻结 judgement
重建并校验综合分分量，再读取 gold。它分别按冻结综合排名、最佳 reciprocal
source rank、支持列表数和支持来源数排序，后三者仅用稳定候选身份作并列裁决，
不拟合权重或搜索组合。

报告同时保留全量可观察结果和严格子集；严格子集要求所有已执行检索请求成功，
且每个候选的 source、adapted query 与正源内排名 provenance 完整。Top-20/50/100
只表示冻结候选池内的 gold 捕获，不等同于经过相关性过滤后的产品 Recall/F1；
未匹配候选只能称为 benchmark 非 gold，不能当作可靠人工负例。2026-07-21 的
固定审计中，最佳 reciprocal rank 提升 SciFact Top-20 捕获但降低 Auto dev，
支持列表数和来源数则在 SciFact 退化；没有 provenance 单信号在三个集合严格
子集的所有深度均不退化，因此本轮证据不支持进入生产排序实验。

## 离线消融组

| 分组 | 查询演化 | RefChain |
| --- | --- | --- |
| `baseline` | 关闭 | 关闭 |
| `query_evolution_only` | 开启 | 关闭 |
| `refchain_only` | 关闭 | 开启 |
| `query_evolution_plus_refchain` | 开启 | 开启 |

真实四组消融固定使用 AutoScholarQuery 原始顺序开发集 10 条与独立验证集 5 条，来源为 arXiv/OpenAlex，使用 adaptive、balanced、Top-20、关闭 LLM，并限制为两轮、150 候选和 120 秒。阶段诊断事后统计 seed、查询或引用、去重候选、gold、Judgement/Top-K 丢失、API、重试、缓存和模块耗时；gold 不进入在线决策。

2026-07-19 的开发集 baseline 完成，候选 Recall、Recall@20、F1@5/10/20 分别为 0.150、0.125、0.0222/0.0143/0.0179，平均 API 3.1、平均延迟 32.81 秒。OpenAlex 0 次成功，TLS timeout 重试后出现 HTTP 429，随后 9 个 case 进入 cooldown；按实验协议停止其余开发组和全部验证组。当前没有可比较的模块增量，不得据此声明 Query Evolution、RefChain 或组合有效。

## Judgement 小样本校准

规则校准只在冻结的 arXiv Retrieval Snapshot 候选上重算 Judgement 与既有 Reranker，不重新查询、不读取 gold 生成规则。开发集固定为 offset 0、limit 10；运行前冻结 128 个通用参数组合，以 F1@20、Recall@20、gold Judgement false negative、Precision@20、MRR、默认参数距离和配置哈希依次稳定选型。配置冻结后，验证集 offset 10、limit 5 只运行一次。

开发集选出的 `calibrated_rules_v1` 将高度相关阈值由 0.72 调为 0.68、标题主题权重由 0.12 调为 0.10。开发集 F1@20、Precision@20、Recall@20 和 gold false negative 均不变，MRR 为 0.027692→0.045000，平均返回量为 11.4→10.2；独立验证集全部排名指标和 gold false negative 均不变，平均返回量为 11.2→9.4。候选召回完全一致，Replay 的 HTTP、重试和网络等待均为 0。该结果只通过“无回归、返回量受控”的候选门槛，不证明 F1 或 false negative 已改善，产品默认保持 `current_rules`。

复现命令：

```bash
PYTHONPATH=src python scripts/calibrate_judgement.py \
  --dataset auto_scholar_query --offset 0 --limit 10 \
  --validation-offset 10 --validation-limit 5 \
  --snapshot-dir outputs/benchmark_snapshots/autoscholar_qe_gap_dev10_20260720 \
  --validation-snapshot-dir outputs/benchmark_snapshots/autoscholar_qe_gap_val5_20260720 \
  --output outputs/benchmark_runs/judgement_calibration --resume
```

输出包含 manifest、冻结配置、开发网格、阈值敏感性、开发/验证对比和候选级诊断。Benchmark 非 gold 候选只标为“非 gold 候选”，不能视为真实负例。

### 固定保留集复核

2026-07-20 又在未参与选参的 AutoScholarQuery `offset=20, limit=30` 上进行一次固定复核。实验使用 arXiv、`current_rules` 初始规划、adaptive、关闭 Query Evolution/RefChain/LLM、balanced、Top-20 和 `highly_and_partial`；两种 Judgement 复用同一 Retrieval Snapshot，候选指纹与 candidate Recall 完全一致，回放 HTTP、重试和网络等待均为 0。

65 个 gold 中只有 1 个进入初始候选，candidate Recall 为 0.008333。`current_rules` 的 F1@5/10/20 为 0.007407/0.004762/0.002778，Recall@20 为 0.008333、MRR 为 0.016667、nDCG@20 为 0.008210；`calibrated_rules_v1` 将唯一召回的 gold 判为弱相关，各项最终排名指标均为 0，gold Judgement false negative 为 1/1。30 条查询的配对 bootstrap 使用固定 seed 20260720 和 5000 次重采样，F1@20 差值为 -0.002778，95% 区间 `[-0.008333, 0]`；Recall@20、MRR、nDCG@20 的区间也都包含 0。该保留集不支持校准配置具有稳定优势，产品默认继续使用 `current_rules`；64/65 gold 在 Judgement 前已缺失，下一瓶颈是 Retrieval。

复现命令：

```bash
PYTHONPATH=src python scripts/run_judgement_holdout.py \
  --snapshot-dir outputs/benchmark_snapshots/autoscholar_holdout30_20260720 \
  --output outputs/benchmark_runs/holdout30_baseline
```

产物包含 `comparison.json`、`comparison.md`、`per_query_diagnostics.jsonl`、`error_summary.json` 和 `snapshot_coverage.json`。查询切片只使用规则解析出的结构、方法、数据集、必要词、论文类型和查询长度，不使用 gold 定义分组；本次样本与极低候选召回不支持总体显著性或真实性能声明。

同一 holdout30 的事后 Retrieval 审计按 arXiv ID 冻结 65 个 gold 的标题、摘要、作者、年份、分类与 DOI，并对实际 adapted query、exact title、规范化标题和标题核心词分别 Record/Replay。ID 与 exact-title 可用率均为 65/65；当前查询扩到 Top-20/50/100 只找回 1/5/7，规范化标题找回 58/65，核心词在 Top-20/50/100 找回 60/64/64。64 个原始未召回 gold 的原因分布为：查询未匹配 25、过度约束 23、词汇错配 10、21–50 名截断 4、51–100 名低排位 2；adapter 术语丢失、元数据不匹配和 gold 级来源不可用均为 0。主导原因族是查询构造（48/64），只扩大候选深度最多额外恢复 6 个。

审计快照的 274 个必需键包含 269 个成功和 5 个冻结失败，记录 286 次请求、12 次重试；纯离线 Replay 命中 274 个键，HTTP、重试和网络等待均为 0。oracle 只用于 SearchService 完成后的诊断，不进入生产查询。复现时依次运行 `scripts/audit_retrieval_recall.py --mode plan`、`--mode record-missing` 和 `--mode replay`；最终产物位于 `outputs/benchmark_runs/retrieval_recall_audit/`。

## 数据格式

离线 fixture 的 `search_cases.jsonl` 每行包含 `query_id`、`query`、`gold_papers` 和 `top_k_values`。批量 qrels 每行包含 `case_id` 与 `relevant_papers`。gold 论文可提供上述任一稳定标识符；没有稳定标识符时必须同时提供标题和年份。`relevance_grade` 大于 0 表示相关，缺省为 1。

## 公开 Benchmark

当前接入 `benchmark/AutoScholarQuery_test.jsonl`：1000 条 test 查询、2403 条 gold 论文，所有 gold 均带 arXiv ID；原数据没有分级相关性，适配器按其二元 gold 关系使用 `relevance_grade=1`。查询、标题、标识符和源文件顺序均保持不变。仓库中的 LitSearch 目录目前只有代码，没有本地 query/corpus 数据，因此未建立 LitSearch Adapter。

### BEIR SciFact 泛化适配

`beir_scifact` 适配器只接受 BEIR 官方 TU Darmstadt `scifact.zip` 的解压目录或归档，固定使用 `qrels/test.tsv`，并按 query ID 的 SHA-256 升序稳定抽取 50 条 query。每条正相关关系保留 qrel grade；当前 SciFact test qrels 的正相关 grade 为 1，适配器仍将 grade 写入 gold metadata，便于离线敏感性复核。语料的 `_id` 同时写入 gold 的顶层 `s2orc_corpus_id` 与审计 metadata，并由统一身份实现规范化为 `s2orc:<corpus-id>`；Semantic Scholar connector 显式请求官方 `corpusId` 字段并保留来源返回的精确值，字段采集版本为 `search-v2`。该标识只允许精确 ID 匹配，候选缺失 Corpus ID 时不会通过标题推断。缺失语料映射会在加载阶段失败，不会被计为检索未命中。官方来源、固定 BEIR commit、归档校验值和抽样口径记录在 `benchmark/beir_scifact_manifest.json`；原始归档只保存在被忽略的本地输入目录中。

SciFact 的 evaluator 另使用版本化 `benchmark/beir_scifact_s2_crosswalk.json`：采集器只向 Semantic Scholar 官方 Graph API 发出 `CorpusId` 精确查询，并只保留响应中的 `paperId`、`corpusId` 与 `externalIds` 稳定标识。crosswalk 由 `scripts/build_scifact_crosswalk.py` 依次执行 `preflight`、串行 `record-missing` 和严格 `replay` 构建；它只丰富 `EvalGoldPaper`，不进入查询、Prompt、检索、排序、候选补全或生产 API。统一 `identity.py` 对 DOI、arXiv、PMID、Semantic Scholar ID 和 S2ORC Corpus ID 做精确规范化；任一同类标识冲突均阻止匹配，标题不会为带 Corpus ID 的记录兜底。官方精确查询的 `unavailable`/`failed` 终态在 gold 诊断中单列为 `identity_crosswalk_*`，不进入 Retrieval miss 或 Recall/F1 分母；旧数据不含 crosswalk metadata 时仍保持原有身份口径。

SciFact BM25 上界审计位于 `scripts/audit_scifact_bm25.py` 和 `scholar_agent.evaluation.scifact_bm25_audit`，不接入 `SearchService`。索引固定覆盖官方 5,183 篇 corpus，文档文本为未加权的 `title + abstract`；查询只使用 manifest 中 50 条原始 query text。实现固定为 `rank_bm25==0.2.2` 的 `BM25Okapi` 默认参数（`k1=1.5`、`b=0.75`、`epsilon=0.25`），tokenizer 仅做 Unicode `\w+` 与 casefold，不使用停用词、词干、扩展或 gold 信息。20/50/100/200 指标均从同一次完整语料排序的前缀计算，以 Corpus ID 精确评测；外部基线交集使用冻结 Replay 的 `initial_retrieval` 候选和同一统一身份规则。配置、输入哈希、汇总与产物哈希记录在 `benchmark/beir_scifact_bm25_audit_manifest.json`。该结果仅表示语料已知时的离线词法召回上界，不是产品可实现成绩。

`local_bm25` 将相同词法语义作为默认关闭的正式 connector 接入统一 SearchService，但不会读取上述审计的 qrels、crosswalk 或结果。Benchmark 必须显式传入 JSONL 语料路径、`document_id/title/abstract` 字段及 document-ID 身份类型；SciFact 固定把官方 `_id` 原样映射为 S2ORC Corpus ID、摘要字段映射为 `text`。PaSa 语义语料只能由带 arXiv ID 的 Cornell/arXiv 元数据按精确 ID 构建，不能使用标题匹配，也不能使用 AutoScholarQuery gold。索引缓存与 Snapshot connector version 同时绑定语料 SHA-256、字段配置及 `k1=1.5、b=0.75、epsilon=0.25`，查询只来自 `current_rules` 的正常 subquery/adapter 路径。该来源与四个产品默认来源共享 200 篇全局候选预算、统一身份去重、Judgement、排序、过滤和 Top-20，不因本地语料增加预算。

`scripts/audit_local_bm25_budget.py` 只读使用上述官方语料、SciFact 固定 50 条 manifest、冻结 `current_rules` 计划和既有 local-BM25 Snapshot，离线重建每个已执行 safe-original 子查询的完整 Top-200 排名。审计协议在读取 gold 指标前冻结于 `benchmark/beir_scifact_local_bm25_budget_audit_manifest.json`：现有 adapter 为每列表 Top-20 后按计划顺序统一身份去重；原始查询上界只取原始 query 的 Top-200；全部子查询上界把各列表 Top-200 按冻结顺序合并、去重后截断 200。三者都使用相同 tokenizer、BM25 参数和稳定 Corpus ID 并列裁决，gold 只在候选池完成后精确匹配；命令不导入 SearchService、不调用 connector/LLM、不写 Snapshot。

42 条可评估 gold 关系全部具有逐列表首次排名、adapter 配额、去重位置及冻结阶段位置。现有每列表 Top-20 在 20/50/100/200 前缀均命中 32 条（macro Candidate Recall 0.7561）；原始查询和全部子查询合并在 20/50/100 均命中 32 条，在 200 命中 35 条（macro 0.8293）。新增 3 条在原始查询列表分别位于 120、150、194，均归因于每列表 adapter 配额；跨查询身份去重、local 来源池截断、全局 200 候选预算和身份匹配各造成 0 条该差距。其余 7 条在所有可观察 Top-200 中仍未命中，属于查询词面失配的可观察证据。

原始查询 Top-200 生成 10,000 个候选且无跨查询重复；全部子查询 Top-200 生成 20,000 个候选，身份去重后 12,253 个，再由每 case 的 local 审计上限裁剪 2,253 个，却没有超过原始查询单列表的新增 gold。因此只值得后续单独评测“原始查询优先加深”的默认关闭方案，不支持全面加深所有派生查询，也不改变生产预算。两轮纯离线产物逐字节一致，`aggregate.json`、`case_audit.jsonl`、`gold_chains.jsonl` SHA-256 分别为 `5209ef2b237d07c6315fb768b49471a618f8b7d7501088332fdfafac6ebcfaed`、`10a9c35ef3b37b17140753f2d457382881c52015ef6beaf661137ae6e64a014e`、`47a6c547042cc35410cb035b5c218827ae71dd435296d39be6be1f96818438b2`；Snapshot tree 前后保持 `2a3b9efdfcec361bf0bdc99d2b30d06fddffb596f9885e70e46b350fdda08d13`。这些只表示固定候选池的离线上界，不是产品成绩或官方成绩。

`local_bm25_original_deepening` v1 是 benchmark-only、默认关闭的配对实验，不修改生产 connector/adapter。它只把 `purpose=original_query` 且 `adaptation_strategy=safe_original` 的 local-BM25 列表从 20 加深到 200；派生本地列表和四源 Snapshot 原样复用。本地候选仍先按统一身份去重并截断 200，随后与外部候选进入冻结的来源均衡全局 200 候选预算、current-rules Judgement、排序、相关性过滤和 Top-20。固定协议见 `benchmark/beir_scifact_local_bm25_original_deepening_manifest.json`，gold 只在两个候选池完成评分与排名后进入诊断。

SciFact 50 条的 50/50 外部响应列表及派生本地列表逐 case 一致。Candidate Recall 从 0.7561（32 条关系、30 篇唯一 gold）升至 0.7805（33/31），逐 query 为改善/持平/退化 1/40/0；Recall@20 和 F1@20 均保持 0.4634/0.04625，41 条可评估 query 全部持平，最终仍返回 20 条关系、19 篇唯一 gold。三条预审计深位 gold 中，原列表第 120 名进入全局池第 146 位，但 Judgement 为 irrelevant、分数 0、最终第 198 名；第 150、194 名均进入本地 200 池，却被全局 200 候选的来源均衡裁剪。策略因此没有把任何深位 gold 转化到最终结果，继续默认关闭。

每轮共审计 9,000 条原始查询深位候选：全部进入本地来源池，7,079 条进入全局池，其中 6,891 条相对 baseline 为新增身份，123 条进入最终返回。两次纯离线 Replay 的四份产物逐字节一致，`aggregate.json`、`case_comparison.jsonl`、`deep_candidates.jsonl`、`gold_conversion.jsonl` SHA-256 分别为 `3acad6ac113c44ef4e5c4f548f2381b2a59c2a2b4848f718d7c004efd6007325`、`65254a165a2fd063a2abc6a28a75b3f081c6abd69ef5f32f3c33a0d456c79864`、`1a91edae9e69b1efdfc653a848587aae3b5fba093c8a356e04d29ca0ef1f1d16`、`6de3837284315e209094127b0c3296dc32254028880463cbc6343403a1739ca1`；Snapshot tree 前后未变，执行期 0 网络、0 LLM、0 Snapshot 写入。汇总见 `benchmark/beir_scifact_local_bm25_original_deepening_result.json`，完整产物位于 `outputs/benchmark_runs/local_bm25_original_deepening_b7b8694_final_r{1,2}/`。

固定 SciFact 50 条配对中，四源与五源组复用了逐 key 完全相同的 262 个冻结外部 required keys；五源组只增加 101 个本地检索 key。42 条 evaluator 可评估 gold 关系对应 39 篇全局唯一论文：候选命中从 5 条关系/5 篇唯一论文增至 32/30（宏平均 Candidate Recall `0.1220→0.7561`），其中本地来源独立新增 27 条关系/25 篇唯一论文；最终返回从 5/5 增至 20/19，宏平均 Recall@20 `0.1220→0.4634`、F1@20 `0.0116→0.0462`，可评估 query 的 Recall/F1 胜负均为 `14/27/0`。本地列表共返回 2,020 个候选，统一身份去重后保留 1,192 个；32 条本地命中 gold 关系中 20 条进入最终返回，6 条被判 weak、6 条被判 irrelevant 且排在 Top-20 外。最终安全 JSON/gzip 缓存实现的冷索引构建约 1.03 秒、缓存加载约 0.54 秒，101 次本地查询记录延迟合计约 2.49 秒；两次 Replay 都是 0 HTTP、0 Snapshot 写入且核心指标、逐 query 指标和逐 gold 诊断一致。由于冻结外部响应包含 2 个失败终态（OpenAlex/Semantic Scholar 各 1 个 required key），四源绝对值仍是外部可用性下界；但两组外部 key/响应完全配对，因此观察到的净增可归因于本地语料来源。该成绩表示“已提供 SciFact 官方语料”的封闭语料检索，不是开放网络成绩，connector 继续默认关闭。

SciFact 跨源可索引性审计位于 `scripts/audit_scifact_source_index.py` 和 `scholar_agent.evaluation.scifact_source_index_audit`，同样与产品检索路径隔离。它只对 crosswalk 状态为 success 的 gold 使用来源支持的稳定标识：arXiv `id_list`、OpenAlex 单 work lookup、Semantic Scholar paper stable-ID lookup、PubMed PMID EFetch；不发送标题或 query，不使用模糊匹配。四源结果通过统一 `identity.py` 验证，同类稳定标识冲突保持不匹配。`record-missing` 串行执行并为每个请求保存不可变 Snapshot；`replay` 严格校验 request/key/content hash，外部失败与 not-found 分开。总体互斥归因优先保留正常查询命中，其次为精确可定位但查询未召回；只有全部适用源明确 not-found 才归为未定位，任何失败均保持不可判定。该 oracle 仅量化来源收录与查询命中缺口，不向查询、Prompt、候选或生产 API 注入 gold。

SciFact 的 current-rules 查询深度审计位于 `scripts/audit_scifact_query_depth.py` 和 `scholar_agent.evaluation.scifact_query_depth_audit`。请求计划只读取冻结 Benchmark 的 `initial_retrieval.retrieval_calls`，保留全部 source-adapted query 及原顺序，不含 case 或 gold 字段。arXiv/OpenAlex 单次请求 200 条，Semantic Scholar/PubMed 固定读取 offset 0/100 两页；20/50/100/200 曲线只取同一有序结果的前缀。Record 对每页使用可终止进程隔离、父级来源限流和连续两次 429 熔断；Replay 严格校验全部 request/key，只在返回结果后以统一精确稳定标识匹配 gold。外部失败、部分分页和 source outage 均保持不可判定，不能转为深度 200 miss。候选曲线同时给出去重后未裁剪池与冻结 200 篇全局预算后的指标，最终 Top-20 继续使用现有 judgement/reranking。固定输入、采集终态、成本及结果哈希记录在 `benchmark/beir_scifact_query_depth_audit_manifest.json`。

复现命令：

```bash
PYTHONPATH=src python scripts/audit_scifact_bm25.py \
  --dataset-path outputs/benchmark_inputs/beir_scifact/scifact.zip \
  --sample-manifest benchmark/beir_scifact_sample_manifest.json \
  --crosswalk benchmark/beir_scifact_s2_crosswalk.json \
  --external-run-dir outputs/benchmark_runs/beir_scifact_crosswalk_replay_commit_47c2168/current_rules_50_replay_r1 \
  --output-dir outputs/benchmark_runs/beir_scifact_bm25_audit_47c2168/run1
```

检查与运行示例（需先取得官方归档）：

```bash
PYTHONPATH=src python scripts/inspect_benchmark.py \
  --dataset beir_scifact \
  --path outputs/benchmark_inputs/beir_scifact/scifact.zip

PYTHONPATH=src python scripts/run_benchmark.py \
  --dataset beir_scifact \
  --dataset-path outputs/benchmark_inputs/beir_scifact/scifact.zip \
  --dataset-split test --run-id beir_scifact_current_rules_50 \
  --retrieval-mode record-missing \
  --snapshot-dir outputs/benchmark_snapshots/beir_scifact_current_rules_50
```

数据检查不访问外网：

```bash
PYTHONPATH=src python scripts/inspect_benchmark.py \
  --dataset auto_scholar_query
```

可恢复的真实运行示例：

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --dataset auto_scholar_query \
  --limit 10 \
  --output-root outputs/benchmark_runs \
  --run-id autoscholar_smoke \
  --run-profile balanced \
  --sources openalex,arxiv,semantic_scholar \
  --query-adapter-policy adaptive \
  --result-policy highly_and_partial \
  --top-k 20
```

每次运行独立写入 `config.json`、`dataset_report.json`、`results.jsonl`、`metrics.json`、`failures.jsonl` 和 `summary.md`。`--resume` 跳过成功案例、重试失败案例，并在配置签名一致时重新汇总全部结果。

真实响应快照支持 `live`、`record`、`record-missing`、`plan` 和 `replay`。`plan` 完全离线执行真实流水线并输出动态缺键及依赖，采集器串行补齐后再次规划到固定点；`--retry-failed-snapshots` 才会重试已冻结失败。结构合法的 success 或 failed 条目都算覆盖，但分别统计；Replay 缺键、Schema 不兼容或内容哈希不一致时失败，不访问外网：

纯 `replay` 按冻结组中的请求 key 重放每个已记录终态，不读取或更新 live connector 的进程级/运行级 cooldown。一个失败快照仍以原错误返回并计为失败，但不会抑制后续已有成功快照；组内必需 key 缺少文件时明确失败，未进入冻结组的历史未执行路径记为 `snapshot_key_not_recorded`，不会伪装成缺失请求。`record` 与 `record-missing` 继续使用 live 限流、cooldown 和熔断语义。

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --dataset auto_scholar_query --offset 0 --limit 10 \
  --sources arxiv,openalex --query-adapter-policy adaptive \
  --run-profile balanced --top-k 20 --max-workers 1 \
  --retrieval-mode replay \
  --snapshot-dir outputs/benchmark_snapshots/autoscholar_qe_refchain_dev10 \
  --run-id qe_refchain_replay_baseline --diagnostics

PYTHONPATH=src python scripts/inspect_benchmark_snapshot.py \
  --snapshot-dir outputs/benchmark_snapshots/autoscholar_qe_refchain_dev10
```

动态组使用 `prepare_benchmark_ablation_snapshots.py` 的保守集中上限执行“规划→采集→再规划”，计划与四组覆盖汇总位于快照目录的 `plans/`。`replay-ready` 不等于外部请求全部成功；只有纯离线 Replay 完成后才标记 `replay-verified` 并生成模块结论。

`cost_report` 在 Replay 中只表示本次执行成本，HTTP、重试和网络等待均为 0；`snapshot_cost_report` 与 `metrics.json.snapshot_costs` 另列快照所记录的 live 请求、重试、错误、限流等待和延迟。两套数字不得混加或把记录成本描述为 Replay 实际成本。

2026-07-20 固定开发集前 10 条的四组快照均达到 `replay-ready` 并完成纯离线 Replay，实际 HTTP、重试和网络等待均为 0。各组必需键的 success/failed 分别为：baseline 27/3、Query Evolution 52/2、RefChain 27/6、组合组 52/8；failed 条目属于可复现覆盖，不表示成功检索。四组 Recall@20 均为 0.1250，F1@20 均为 0.0179。Query Evolution 把候选 Recall 从 0.1500 提高到 0.1875，并新增 197 个唯一候选，但没有新增 gold，结论为 `new_candidates_but_no_gold`；RefChain 的引用响应均为冻结的 OpenAlex 失败，未产生新候选，结论为 `no_action_generated` 和 `source_failure_dominated`。所有结论同时标记 `insufficient_sample` 与 `small_sample_diagnostic_only`，不得外推为模块有效性或正式性能。

覆盖缺口策略使用 arXiv-only 冻结快照对比 `off`、`seed_expansion`、`coverage_gap`。开发集前 10 条中，三组 F1@20/Recall@20 均为 0.0179/0.1250；旧策略生成 24 条查询、53 次记录请求、197 个新增唯一候选，无效候选占比 0.7462，新策略为 10 条、41 次、232 个，占比 0.6584。独立 offset 10 的 5 条验证中，三组均为 0.0190/0.2000；旧策略为 12 条查询、27 次请求、135 个新增唯一候选，无效候选占比 0.8657，新策略为 4 条、18 次、78 个，占比 0.5938。两组策略在开发和验证子集都没有新增 gold，因此每新增 gold 的 API 与延迟为 `null`，产品开关保持默认关闭。

`run_benchmark.py --query-evolution-policy` 记录策略并选择独立快照组；`analyze_query_evolution_policies.py` 输出三组汇总、逐查询诊断和中文摘要。质量门使用事前约束信号，gold 只在全部运行完成后参与上述对比。

初始查询规划另以 arXiv-only、adaptive、balanced、Top-20、关闭 Query Evolution/RefChain/LLM 的冻结快照比较 `current_rules` 与 `facet_balanced` v1.2。开发前 10 条中，两者候选 Recall/F1@20/Recall@20 同为 0.1500/0.0179/0.1250；新策略平均记录请求为 2.6（旧策略 2.7），重复率为 0.2538（旧策略 0.4167）。独立 offset 10 的 5 条验证中，两者为 0.2000/0.0190/0.2000；新策略平均记录请求为 2.6（旧策略 2.4），唯一 gold 均为 1。由于未新增 gold 且成本增加，验收不通过，默认保持 `current_rules`。

`controlled_relaxation` v1.4 的规则只在 AutoScholarQuery `offset=50, limit=20` 开发集确定，随后冻结，并在 `offset=70, limit=20` 独立验证集运行一次。配置仍为 arXiv-only、adaptive、balanced、Top-20、`highly_and_partial`，关闭 Query Evolution、RefChain 与 LLM；两种策略使用隔离快照，最终 Replay 的 HTTP、重试和网络等待均为 0。开发集上旧/新策略的候选 Recall、唯一 gold、F1@5/10/20、P@20、R@20、MRR、nDCG@20 均分别为 0.1750、4、0/0.0083/0.0093、0.0050、0.0750、0.0113、0.0229；平均记录 API 为 2.40/2.45，唯一候选为 626/682，重复率为 0.3479/0.2745。验证集两组的候选 Recall、唯一 gold、F1@5/10/20、P@20、R@20 均为 0.1500、4、0.0143/0.0167/0.0139、0.0075、0.1000；旧/新的 MRR 为 0.0219/0.0183，nDCG@20 为 0.0356/0.0329，平均记录 API 均为 2.80，唯一候选为 689/762，重复率为 0.3848/0.3196。补充查询在验证集新增 221 个独占候选但新增 gold 为 0，因此未通过“至少新增一个 gold”的首要门槛，默认继续使用 `current_rules`。分析产物位于 `outputs/benchmark_runs/controlled_relaxation_analysis/`；该 20+20 小样本不代表完整 Benchmark。

来源互补性实验固定 `current_rules`、adaptive、balanced、Top-20，关闭 Query Evolution、RefChain 与 LLM，在开发集 `offset=90, limit=20` 冻结流程后，只对独立验证集 `offset=110, limit=20` 评估一次。验证集 arXiv-only 与 arXiv+OpenAlex 的候选 Recall、F1@20、R@20、MRR、nDCG@20 完全相同，分别为 0.0792、0.0133、0.0792、0.0205、0.0330，均找回 3 个唯一 gold；OpenAlex 没有成功响应或独占 gold，其最终执行路径记录错误率为 1.0。双源没有新增 gold，未形成 `high_recall` 候选，产品默认不变。最终 Replay 的 HTTP、重试和网络等待均为 0；五项产物位于 `outputs/benchmark_runs/source_complementarity/`。该 20+20 小样本只能说明当前失败环境下没有互补性证据，不能推出 OpenAlex 长期无贡献。

`disjunctive_facets` v1.5 在来源无关的规划层保留原查询，再生成最多一条 4–8 个可靠 topic/method/dataset/task 词的 `any` 查询及一条可选的“topic + 最佳分面”查询；显式 must-have 始终在 OR 组外保持硬约束。规则在 AutoScholarQuery `offset=130, limit=20` 开发集冻结，并在 `offset=150, limit=20` 独立验证集只回放一次。验证集相对 `current_rules` 的候选 Recall 为 0.0750→0.1000、唯一 gold 为 2→3，F1@20 与 R@20 分别保持 0.0093 和 0.0750，平均记录 API 为 2.40→2.50，弱相关与无关候选占比为 0.5377→0.5254；全部预设门槛通过。OR 子查询产生 153 个独占候选但事后独占 gold 为 0，新增 gold 来自整体规划组合，故策略仍保持实验状态且产品默认继续使用 `current_rules`。四次最终 Replay 的 HTTP、重试和网络等待均为 0；分析产物位于 `outputs/benchmark_runs/disjunctive_facets_analysis/`，20+20 小样本不代表完整 Benchmark。

预注册的 AutoScholarQuery `offset=170, limit=40` 保留集在冷却后按原协议续跑完成。三轮动态计划最终补齐 `current_rules` 107 个键与 `disjunctive_facets` 99 个键；两组 Replay 均为 40/40 成功，执行期 HTTP、重试和网络等待为 0。相对基线，析取策略的候选 Recall 为 0.1104→0.1229，唯一 gold 均为 8，但 F1@20、Recall@20、MRR、nDCG@20 分别由 0.0154、0.1021、0.0516、0.0542 降至 0.0131、0.0896、0.0266、0.0389。OR 查询产生 276 个独占候选和 1 个事后独占 gold，但整体未净增 gold；预设门槛未通过，策略继续仅作实验选项，不进入 high_recall profile，也不据此保留集调参。固定种子成对 bootstrap 与逐查询诊断位于 `outputs/benchmark_runs/disjunctive_holdout40/`。

`current_plus_disjunctive` v1.6 不替换旧规划：先按原顺序执行全部 `current_rules` 查询，只在高置信词和剩余候选预算允许时追加一条 OR，旧候选优先。AutoScholarQuery 开发集 `offset=210, limit=20` 中，候选 Recall、唯一 gold、F1@5/10/20、R@20 均保持 0.1250、4、0.0595/0.0341/0.0184、0.1250；OR 新增 107 个独占候选但无新增 gold，MRR 为 0.1167→0.1083。规则冻结后在独立验证集 `offset=230, limit=20` 只评估一次：候选 Recall、唯一 gold、F1@5/10/20、R@20、MRR、nDCG@20 均分别保持 0.0625、2、0.0278/0.0162/0.0089、0.0625、0.0350、0.0391；全部 2 个基线 gold 均保留，OR 增加 105 个独占候选但净新增 gold 为 0。平均记录 API 为 2.45→3.40，弱相关与无关率为 0.6860→0.6930；候选未通过“至少新增 1 个 gold”，停止继续扩展 OR，默认保持 `current_rules`。四组最终 Replay 的 HTTP、重试和网络等待均为 0，产物位于 `outputs/benchmark_runs/current_plus_disjunctive_analysis/final_analysis/`。

`facet_union` v1.7 完整保留基线，再按 dataset、method、task、topic 的稳定优先级追加至多一条独立分面查询。开发集 `offset=250, limit=20` 的候选 Recall 和唯一 gold 均保持 0.1625 和 6；分面查询新增 387 个独占候选但独占 gold 为 0。冻结后在验证集 `offset=270, limit=20` 只评估一次：候选 Recall、唯一 gold、F1@5/10/20、R@20 均保持 0.1625、5、0.0587/0.0337/0.0228、0.1625，基线 5 个 gold 全部保留，但 MRR 为 0.0979→0.0965、nDCG@20 为 0.0933→0.0923；391 个独占候选没有新增 gold。平均记录 API 为 2.40→3.55，弱相关与无关率为 0.5562→0.6596。预设验收未通过，`facet_union` 仅保留为实验策略，产品默认继续使用 `current_rules`；规则式 OR、放宽和分面组合规划至此冻结，后续转向语义规划或其他语义检索。四组最终 Replay 的 HTTP、重试和网络等待均为 0，产物位于 `outputs/benchmark_runs/facet_union_analysis/final_analysis/`。

`concept_projection` v1.8.1 是默认关闭的固定预算实验：它只按原文顺序选取规则式 Query Analysis 已有的 must-have/topic 片段，按大小写和边界标点去重，排除否定、格式、时间与数量约束，并替换最低优先级派生查询。SciFact 50 条、AutoScholarQuery 开发 0/10 与验证 10/5 的每个配对 case 都保持相同规划查询数且原查询始终首位；投影分别实际替换 20/10/5 条。正式 Replay 中，SciFact 的候选 Recall/Recall@20/F1@20/唯一 gold 在旧新策略间同为 0.1220/0.1220/0.0116/5，验证集同为 0.2000/0.2000/0.0190/1。开发集候选 Recall 与唯一 gold 同为 0.1500/3，但 Recall@20 从 0.1250 降至 0.0250、F1@20 从 0.0179 降至 0.0083；唯一退化查询的已召回 gold 从第 13 位降到第 24 位，该查询在双方来源调用均完整的配对子集中。SciFact/开发/验证双方完整子集分别为 49/50、8/10、4/5，其余结果保留为外部失败下界；全部两轮 Replay 的 HTTP、重试和网络等待均为 0，逐 gold 产物字节一致。策略没有跨数据集净增且开发集退化，继续默认关闭；正式产物位于 `outputs/benchmark_runs/concept_projection_0a0feb6_v3/`。

`--ranking-policy rrf_fusion` 是默认关闭的纯排序消融。它不改变查询、请求、候选预算、Judgement、过滤或指标；对每个实际 `(source, adapted_query)` 响应列表使用固定 `k=60`、等权 RRF，同列表重复论文只贡献最佳名次，现有综合分只作同分裁决。正式比较必须对基线与实验的排序前候选及 Judgement 状态逐 case 做完全一致性校验；列表 provenance 缺失或非法时该集合不可评测，不得推造名次。

固定 SciFact 50 条、AutoScholarQuery 开发 0/10 与验证 10/5 的纯 Replay 消融中，两组排序前候选身份、数量和 Judgement 状态逐 case 完全一致，双 Replay 的核心指标与逐 gold 诊断一致，执行期 HTTP、重试、网络等待和快照写入均为 0。开发集候选 Recall/唯一候选 gold 均为 0.1500/3，但 Recall@20 与 F1@20 从 0.1250/0.0179 降至 0.0250/0.0083，逐查询为改善 0、持平 9、退化 1；验证集候选 Recall/唯一候选 gold、Recall@20、F1@20 均保持 0.2000/1、0.2000、0.0190，5 条全部持平。SciFact 的候选 Recall/唯一候选 gold 均为 0.1220/5，但 Recall@20 与 F1@20 从 0.1220/0.0116 降至 0.0976/0.0093；41 条身份可评估查询中改善 0、持平 40、退化 1，另 9 条保持身份不可评估。RRF 未在三个集合均不退化，也未形成稳定净增，因此继续默认关闭；正式产物位于 `outputs/benchmark_runs/rrf_b0c9636_formal/`。

`llm_semantic` 评测固定使用同一 arXiv-only、adaptive、balanced、Top-20 配置，开发集为 offset 0 的 10 条，独立验证集为 offset 10 的 5 条；每例最多一次温度为 0 的 LLM 调用和两条补充查询。LLM 规划快照与检索快照独立冻结，动态计划先补 LLM 键再发现检索键；最终比较必须同时使用 LLM replay 与 retrieval replay，执行期两类网络请求、重试和网络等待均为 0。当前进程的 LLM provider 为 disabled，因此只重放了 `current_rules` 基线，未运行或伪造 LLM 指标。开发/验证基线的候选 Recall、F1@20、Recall@20 分别为 0.1500/0.0179/0.1250 和 0.2000/0.0190/0.2000；分析产物位于 `outputs/benchmark_runs/llm_query_planning_analysis/`，验收状态为未执行。

`llm_constrained_rewrite` 的正式配对口径固定为 SciFact 50 条、AutoScholarQuery development offset 0/10 与 validation offset 10/5。基线和实验组均使用产品默认来源、adaptive、balanced、Top-20，并关闭 Query Evolution、RefChain、concept projection、RRF 及其他 LLM 能力；实验组每例至多一次温度为 0 的严格 JSON 调用，原始查询保持首位，唯一改写只替换最低优先级派生查询，因此子查询数和计划请求预算不变。LLM Record 仅报告真实请求、Token、延迟和失败，正式召回/F1 只取同一 LLM 与 retrieval 快照的 Replay；Schema 失败、本地拒绝、fallback 和来源失败必须单列，fallback 不得计作改写收益。只有三个集合均不退化且至少两个集合的最终 Recall@20 或 F1@20 严格提升，才可建议继续评估；否则策略保持默认关闭。

`scripts/audit_current_rules_subqueries.py` 对冻结的 `current_rules` Replay 做纯离线子查询边际审计。它按 planning 的优先级以及 `(case, source, subquery)` 重建实际 Snapshot 列表，使用统一身份去重，再在 gold 仅进入 evaluator 后计算独立候选、独立 gold、首次命中和计划顺序边际。`failed`、缺失 Snapshot 与 `not_started` 分别记录；失败或缺失列表不会被记作零贡献。反事实固定使用现有去重、候选预算、规则 Judgement、排序、过滤和 Top-20：一组只保留 `original_query`，另一组每次全局移除一种通用派生 purpose。共享 Snapshot key 只有在所有 owner 都被移除时才计请求节省。输出不调用 SearchService、connector 或 LLM，并将网络请求、LLM 请求和 Snapshot 写入固定记录为 0。

`scripts/audit_current_rules_sources.py` 在同一冻结输入上按 query 计划顺序和来源顺序重建各来源候选，并分别计算单源保留与 leave-one-out。全量结果明确标记为已观测 Snapshot 下界；严格来源子集只纳入该 case 中至少一个实际来源请求成功、且该来源没有 failed、缺失或终态不一致的记录。未启动适配请求不产生候选或请求成本，但整项来源都未启动时不进入严格子集。独立候选/gold 只有在同 case 四源均成功时才具有严格归因资格；否则只输出 `observed` 值，不作为安全删除证据。请求、重试、延迟和限流节省按唯一 Snapshot key 聚合，离线执行继续固定为 0 网络、0 LLM、0 Snapshot 写入。

`scripts/audit_cross_strategy_union.py` 对已冻结的 `current_rules`、`concept_projection`、`llm_constrained_rewrite`、Query Evolution 与 `llm_semantic` 产物做跨策略候选联合上界审计。每个实际执行的 `(strategy, source, adapted_query)` 必须追溯到 Snapshot key 和终态；没有完整数据集产物的策略在输入 manifest 中显式排除，LLM fallback/rejected case 回退到同 case 的 `current_rules` 候选且不计策略贡献。全部成功候选先按统一论文身份合并，再分别报告当前规则排序的真实可转化 Recall@20/F1@20 与 gold 优先的候选池 oracle；后者只表示候选集合上界，不是可实现成绩。跨策略相同 `(source, query)` 的响应顺序或终态不一致会使 case 离开严格可比子集，防止把外部响应漂移计作查询策略收益。最小查询集合使用 gold 的时间点仅限离线审计，在已执行查询列表上做确定性精确 set cover，不接入生产规划。该命令不导入 SearchService 或 connector，并固定记录 0 网络、0 LLM、0 Snapshot 写入。

固定产物审计中，SciFact 没有完整的 Query Evolution/`llm_semantic` Replay，故只纳入另外三种策略；Auto 开发/验证均纳入五种。全部可观察下界中，SciFact 的候选 Recall/候选 gold 从 0.1220/5 提高到 0.1463/6，但当前排序后的 Recall@20/F1@20 仍为 0.1220/0.0116，gold 优先 oracle 才达到 0.1463/0.0139；Auto 开发候选 Recall/gold 保持 0.1500/3，当前排序却从 0.1250/0.0179 降到 0.0250/0.0083；Auto 验证候选 Recall/gold 从 0.2000/1 提高到 0.3000/2，且当前排序转化为 0.3000/0.0372。严格可比 case 仅为开发 1/10、验证 1/5、SciFact 2/50，其中只有验证集观察到新增 gold；其余受 LLM fallback、来源失败或重复查询响应漂移限制。审计 oracle 覆盖这三个并集分别最少需要 2/2、2/2、6/12 个查询/记录请求，但不得据此修改线上计划。两次产物字节一致，逐 case 与 aggregate SHA-256 分别为 `8d4b5353c7ea3c285308d4a5a2ed8d37ed98cefe4fa29f8766399f87fc85b459`、`85341d30b68959777a43109746d7df32b322b39595a44ec99822ed9b015155ab`。现有同类查询改写没有形成跨集合稳定净增，后续应转向新的召回范式；SciFact 的 1 个未转化 gold 同时说明排序转化仍是独立瓶颈。

`prf_v1` 是默认关闭的固定预算伪相关反馈实验。它与 Query Evolution 不等价：后者在完整初始检索后根据 judgement/coverage gap 开启额外演化轮次，PRF 则先只执行原始查询，以现有规则排序后的前 5 篇唯一候选为 seed，从标题与摘要抽取跨至少 2 篇 seed 出现的 unigram/bigram，并以固定倒数排名折扣选取最多 6 项；第二查询替换最低优先级派生查询，计划子查询数、每来源 adapter 上限和全局请求预算均不增加。重复 seed 使用统一论文身份合并；数字、年份、URL、论文标识、停用词及原查询词全部排除。首轮全失败、没有 seed/反馈词或没有可替换派生查询时完整回退 `current_rules`。

固定 SciFact 50 条、AutoScholarQuery development 0/10 与 validation 10/5 的正式 Replay 中，PRF 分别应用于 47/50、10/10、5/5 个 case，SciFact 另有 3 个 `no_derived_query_to_replace` 回退。全部可观察下界的候选 Recall/Recall@20/F1@20/唯一 gold：SciFact 旧新均为 0.1220/0.1220/0.0116/5；开发集为 0.1500/0.1250/0.0179/3 → 0.0500/0.0250/0.0083/2；验证集为 0.2000/0.2000/0.0190/1 → 0/0/0/0。逐查询改善/持平/退化分别为 0/50/0、0/9/1、0/4/1。来源终态双方均完整的严格子集只有 38/50、5/10、3/5，三个子集的候选 Recall、Recall@20 与 F1@20 均逐查询持平，因此全量两次退化不能归因于 PRF，而是来源终态不一致的下界差异。

PRF 查询新增的独占候选分别为 537/115/61，但三个集合的独占 gold 均为 0；SciFact 的 PRF 查询虽重复命中 2 个既有 gold，也没有形成边际召回。快照记录成本旧→新分别为：SciFact 请求 419→339、重试 29→2、错误 14→2、延迟 599.43→481.88 秒；开发请求 120→104、重试 12→4、错误 5→4、延迟 206.66→159.46 秒；验证请求 38→53、重试 4→6、错误均为 2、延迟 63.46→83.35 秒。实际调用数会因 adaptive adapter 的等价/低保留跳过和来源 cooldown 在固定上限内变化，不能把单次外部可用性差异解释为 PRF 效率收益。两轮 Replay 均为 0 HTTP、0 快照写入、0 missing；逐 gold 与 PRF 逐 case 审计字节一致，分析产物三文件 SHA-256 分别为 `051c54d1fed9f07814e91704badf40e07bf28efb0a3fce69f603e0f5ac005727`、`1ded9b10507e60762144ba8df9eebbfc9feaa0518e20f26e55f1838a2e66cb27`、`d37affe99b580de97d611ca256686a079e78c70e300b8b8c2fa856412e48602b`。策略未在三个集合均不退化，也没有任何最终 Recall/F1 净增，故保持默认关闭。

`semantic_seed_expansion` 是默认关闭的 Semantic Scholar 推荐候选扩展。正式配对必须先验证基线与实验的 `initial_retrieval`、`initial_deduplicated` 和 `initial_reranked` 候选及顺序逐 case 一致；Snapshot 组把实验的初始 retrieval key 冻结为同查询策略 baseline 的 required-key 集合，目录内其他既有 key 不得进入实验首轮。对没有 Paper ID 的首轮候选，实验只使用来源已有的 DOI、arXiv ID、PMID 或 S2ORC Corpus ID 构造一次官方 paper-batch 请求；返回映射必须经统一身份规则确认共享精确稳定标识且无同类冲突，不能使用标题等软证据，也不补写普通检索候选。直接与解析 seed 合并后仍只按首轮排名取最多 3 个，并执行至多 1 个、limit 100 的官方批量推荐请求。推荐候选使用统一身份去重和既有候选预算、Judgement、Reranker；重复候选、无 seed、冲突映射、失败或 fallback 不计为收益。阶段诊断分别记录解析候选/请求/成功/冲突/缺失、直接与解析 seed、两个 Snapshot key、推荐原始/唯一/新增候选、首轮与扩展后 Candidate Recall、独立 gold、首轮 gold 丢失和 Top-20 转化。正式指标只取 Replay；解析与推荐的 Record 成本归入 reference 请求，来源失败另报告，并同时给出排除失败请求的严格可比子集。

加入官方精确 ID 解析后的固定 SciFact 50 条、AutoScholarQuery development 0/10 与 validation 10/5 正式 Replay，三个初始阶段分别达到 50/50、10/10、5/5 完全一致。相对旧扩展每组只有 1 个 case/3 个 seed，新版具有 seed 的 case/最终 seed 分别增至 47/141、9/27、5/15，其中解析 seed 为 140/25/12；批量解析候选中分别有 1748/367/149 个通过精确稳定标识与冲突校验，另有 25/11/3 个同类标识冲突保持拒绝。推荐产生 3999/799/498 个新增唯一候选，但三个集合的独立 gold 均为 0。

全部可观察结果中，SciFact 的候选 Recall 与候选唯一 gold 保持 0.1220/5，Recall@20、F1@20 与最终返回 gold 从 0.1220、0.0116、5 降至 0.0976、0.0093、4；Auto 开发保持候选 0.1500/3，但最终三项从 0.1250、0.0179、2 降至 0.0250、0.0083、1；验证集保持 0.2000、0.2000、0.0190、1。F1@20 逐查询改善/持平/退化为 0/40/1、0/9/1、0/5/0：SciFact case 146 的既有 gold 由 rank 6 降至 57，开发 case 4 由 rank 13 降至 38，均被挤出 Top-20。排除解析或推荐 source failure 后的严格可比集为 32/41、8/10、5/5 个可评测 case，前两组仍退化、验证持平。新增 reference 实际请求/重试/错误/记录延迟分别为 133/36/10/359.33 秒、27/8/2/72.63 秒、12/2/0/35.97 秒。两次正式 Replay 都是 0 HTTP、0 Snapshot 写入、0 missing，逐 gold SHA-256 分别稳定为 `1b5c6dd68eb9645a9061b07dabe493ea505dad2040dfc0b1ff97436ca26ce045`、`f2899df23aede582b6f49432a5f7fb1fb265271c99e7b272acd812a1faa832da`、`fbb11f251bb1baf8ad1a853c40a6aaf36cb6aa5e246997859be4675160d71c3a`。解析显著扩大 seed 覆盖，却没有新增 gold，且在两个集合造成排序挤出，故策略继续默认关闭。

`scripts/audit_local_bm25_conversion.py` 对冻结的 SciFact 四源 baseline 与五源 `local_bm25` Replay 做纯离线候选转化审计。命令先逐 case 验证两组 query planning 及四个外部来源的执行终态、Snapshot key 完全一致，再从五源阶段快照追踪每条 local 候选 gold。由于 Retrieval Snapshot 不保存 BM25 原始分数、阶段快照不保存完整 reranker breakdown，审计只读加载同一官方语料，以固定 tokenizer 和 `k1=1.5、b=0.75、epsilon=0.25` 在内存重算 BM25 分数；随后用 Snapshot 中的完整论文和冻结 Judgement 重建 reranker 分量，并要求每篇候选的身份、rank、category 与 final score 和 Replay 完全一致。gold 只在上述重建完成后参与匹配。该路径不导入 SearchService、不调用 connector/LLM，也不写 Snapshot。

固定 50 条 SciFact 中，42 条可评估 gold 关系有 32 条进入候选池（Candidate Recall 0.7561，30 篇全局唯一论文）。32 条均具有完整的本地列表排名、BM25 分数、去重后位置、Judgement 分量、综合评分分量和最终排名；其中 20 条成功返回，12 条的互斥终态均为相关性过滤（6 weakly relevant、6 irrelevant），身份合并丢失、候选预算裁剪和“标签可返回但纯排序超出 Top-20”均为 0。当前 Recall@20/F1@20 为 0.4634/0.04625；保持当前排序但跳过相关性过滤后为 0.6098/0.06018，恢复 6 条 gold。仅按 `local_bm25` 最佳源内排名取 Top-20 为 0.7561/0.07412，覆盖全部 32 条候选 gold；gold-first 候选池 oracle 数值相同，但只表示候选集合上界，不是可实现成绩。另 6 条 irrelevant gold 在当前排名 23–47 位，因此仅跳过过滤仍无法转化；这属于过滤标签与类别优先排序的耦合，而不是独立的 eligible-candidate 排序损失。下一步诊断应优先校准相关性过滤，再评估排序，不据此直接修改生产阈值或权重。

两次审计均为 0 HTTP、0 LLM、0 Snapshot 写入，输入 Snapshot 目录内容哈希前后保持 `ce58a60bd387a9f020985f449313975dcfe9bb344b06c7367308b3150b1ce624`。两次 `aggregate.json`、`case_audit.jsonl`、`gold_chains.jsonl` 分别字节一致，SHA-256 为 `cdf2d2ee25360cf44edb4409bc88709802ba546c972a799de23a48da955e5ff5`、`e94bee0d98ac257a156baf732c6b9cf3dba5d5d1cff2157cee8092a0b5aae339`、`096f893343461367fb4a8fb5d956fcf070269ac894cabbad20aef7615828388f`；固定口径见 `benchmark/beir_scifact_local_bm25_conversion_audit.json`，正式产物位于 `outputs/benchmark_runs/local_bm25_conversion_d385334_r{1,2}/`。

`scripts/audit_relevance_filter.py` 在上述 SciFact Replay 以及现有 AutoScholarQuery dev/val Replay 上重建冻结候选、Judgement 和排序，逐候选输出 topic、must-have、method、dataset、domain 的输入词与命中词、评分分量、警告、类别门和排名链。SciFact 只把具有 `local_bm25` 来源证据的 32 条 gold 作为主审计对象；同查询中排名更高且不匹配任何 gold 的 340 条候选作为冻结非 gold 对照。根因分类采用固定优先级，区分查询解析缺失、字段缺失、保守词形/标点/缩写失配、约束惩罚、固定阈值、类别优先级和其他原因；这些标签仅用于离线诊断，不反馈到规则。

SciFact 的 32 条候选 gold 中 12 条被过滤，误杀率为 37.5%；互斥根因为保守词形/标点/缩写失配 6、约束惩罚 2、固定阈值 2、类别优先级 2，查询解析缺失、字段缺失和其他均为 0。gold 分数中位数/均值为 0.55525/0.49277，排名更高非 gold 为 0.34/0.38548，但 weak/irrelevant gold 仍被固定返回类别门直接排除。逐条移除任一已有负分分量都没有恢复最终 gold；仅跳过返回类别门的单规则审计反事实把 Recall@20/F1@20 从 0.4634/0.04625 提高到 0.6098/0.06018，恢复 6 条，但这是不可部署的上界，不是阈值建议。

AutoScholarQuery 冻结证据仅有 dev 3 条、val 1 条候选 gold：dev 有 1 条 weak gold 被过滤并同样出现保守词形/缩写失配，val 无误杀；所有单规则反事实均无最终指标变化。这个跨数据集信号支持单独评测一个默认关闭、严格保守的词法规范化替代方案，但 Auto 样本太小，不能据此修改默认阈值或生产逻辑。两次审计为 0 HTTP、0 LLM、0 Snapshot 写入，四份产物字节一致；固定口径和哈希见 `benchmark/relevance_filter_audit_result.json`，正式产物位于 `outputs/benchmark_runs/relevance_filter_audit_22a39e9_r{1,2}/`。

`lexical_normalization_v1` 在查看指标前由 `benchmark/lexical_normalization_v1_manifest.json` 冻结，只对 topic、must-have、method、domain 的既有词面证据启用 NFKC/casefold、标点/连字符、点分缩写、英文所有格和保守单复数等价；不改查询解析、权重、负分、阈值、类别门、候选、排序或 Top-20，生产默认保持关闭。`scripts/audit_lexical_normalization.py` 从既有 SciFact local-BM25、Auto dev/val Replay 重建相同候选，并先要求默认关闭分支与冻结 Judgement、排名和最终结果逐项一致，再计算实验分支；gold 只在重建后进入 evaluator。

SciFact 的 50 条中有 41 条查询、42 条 gold 关系可评估，候选 Recall 保持 0.7561（32 条候选 gold）。Recall@20/F1@20 从 0.4634/0.04625 升至 0.4878/0.04857，净增 1 条唯一 gold，逐查询改善/持平/退化为 1/49/0。此前 6 条词法型误杀恢复 1 条；另 4 条虽升分或升排名但仍未过最终类别/排名链，1 条不满足本保守等价规则。实验同时将 203 个未匹配 qrels 的候选新增到返回结果；1 个 gold-matching 重复候选被挤出，但同一唯一 gold 仍由另一稳定身份候选返回，因此唯一 gold 损失为 0。未匹配 qrels 只表示 benchmark non-gold，不能据此断言论文真实不相关。

AutoScholarQuery dev 的候选 Recall 保持 0.1500，Recall@20/F1@20 从 0.1250/0.01786 升至 0.1500/0.02619，恢复既有 1/1 词法误杀，逐查询为 1/9/0；val 的候选 Recall 0.2000、Recall@20 0.2000、F1@20 0.01905 均不变，逐查询为 0/5/0。两个集合分别新增 38/15 个 benchmark non-gold 返回候选，均无唯一 gold 损失。门槛“`SciFact` 提升且 Auto dev/val 不退化”通过，但样本和 precision 标注不足以支持默认启用；下一步只建议保持默认关闭并扩大独立、带负例的精度验证。

最终两轮纯 Replay 的网络、LLM 和 Snapshot 写入均为 0，65/65 case 的排序前候选身份与顺序一致。两轮 `aggregate.json`、`case_comparison.jsonl`、`candidate_diagnostics.jsonl` 逐字节一致，SHA-256 分别为 `5d7b58dd433a9641efdc9216f41b98ecf26d6f1da2633d4fdc281fb70f1e48b6`、`0b8ccb832830242d9481bd3a83e2054857ec0b33177b2be66e96319d53a55dc1`、`d671c11c230dc40c0822c1eac598cbd5d8cceafbe356f007453382ed14b2666c`；汇总见 `benchmark/lexical_normalization_v1_result.json`，正式产物位于 `outputs/benchmark_runs/lexical_normalization_v1_005794c_replay_r{5,6}/`。

`scripts/audit_paired_significance.py` 对上述逐查询结果执行预注册的成对显著性审计。固定协议见 `benchmark/lexical_normalization_significance_manifest.json`：每条可评估 query 等权，分别计算 Candidate Recall、Recall@20、F1@20 的实验减基线差值、20,000 次成对 query bootstrap 百分位 95% CI，以及双侧 paired sign-flip permutation；非零差值不超过 20 对时穷举全部符号排列，否则使用固定种子的 100,000 次 Monte Carlo。跨数据集合并不再给数据集加权；NFKC/casefold 后完全重复的 query 整组排除。功效规划固定以绝对提升 0.01、双侧 alpha 0.05、目标 power 0.8 和观测配对差值标准差作正态近似，只用于规划 1,000 条后续验证，不作为当前显著性证据。

65 条冻结 case 中，SciFact 有 9 条因 gold 身份不可评估而排除，剩余 56 条全部复用同一冻结 retrieval 终态且候选身份一致，因此全量可评估集与严格可比集相同；没有重复 query。Candidate Recall 在所有集合严格不变。SciFact 的 Recall@20/F1@20 平均差为 +0.02439/+0.002323，bootstrap CI 为 [0, 0.07317]/[0, 0.006969]，双侧 permutation p 均为 1.0，逐查询改善/持平/退化为 1/40/0。Auto dev 为 +0.025/+0.008333、CI [0, 0.075]/[0, 0.025]、p 均为 1.0，逐查询 1/9/0；Auto val 两项差值和 CI 均为 0，p=1.0，逐查询 0/5/0。56 条 query 等权合并后 Recall@20/F1@20 为 +0.02232/+0.003189，CI [0, 0.0625]/[0, 0.008078]，p 均为 0.5，逐查询 2/54/0。区间没有出现负值反映观测样本没有退化；置换检验仍不显著，是因为所有增益只集中在两个 query，不能把 bootstrap 区间解读为稳定跨 query 效应。

按预注册的 0.01 绝对最小提升，合并 Recall@20 的观测方差估计需要约 1,477 条 query 达到 80% power，1,000 条估计 power 约 0.635；SciFact/Auto dev 分别约需 1,915/491 条。合并 F1@20 的方差估计约需 23 条，但当前实际效应仅由两条 query 驱动，正态近似和小样本稀疏性限制必须同时保留。统计未提供“无效果”证据，也不改变先前的正向点估计；它只说明现有 41/10/5 小集合不足以证明收益超出随机符号波动。人工 Precision 审计完成前不得建议默认开启。两次统计产物逐字节一致，`statistics.json`、`paired_queries.jsonl`、manifest SHA-256 分别为 `7574bf06481d8547454623e396cdb0dac97dde225da80019389ff22a232beee5`、`0b459be237db291cea9b82a49359c6b146e012805206fdc60c9ec01946b165fa`、`a540f7d43bddb6e19f981056ecdac9f3e0e988b7dcc8361eb65a3ea98dce4af8`；摘要见 `benchmark/lexical_normalization_significance_result.json`，产物位于 `outputs/benchmark_runs/lexical_normalization_significance_3cd47c1_r{1,2}/`。这些仍是内部 Benchmark 指标，不是官方赛题成绩。

`scripts/audit_cluster_significance.py` 在旧 query 级结果之上增加连通分量级推断，
但不覆盖任何历史统计产物。协议在查看新结果前冻结于
`benchmark/lexical_normalization_cluster_significance_manifest.json`：AutoScholarQuery
严格复用既有独立性门禁的 `component_id`，不重算边或簇；同一分量先对 query
差值等权平均，再令每个分量等权。SciFact 不属于该图，因此每条 query 作为显式
外部单例分量。主检验固定使用 20,000 次分量 bootstrap 百分位 95% CI 和 20,000
次双侧 cluster sign-flip，随机种子均为 20260721；旧 query 等权结果只读保留并
作为对照。全量与预注册去污染视图并列，不按效应、p 值或 case 拆合分量。

Record160 的 160 条主分析 query 归入 134 个分量；Recall@20/F1@20 分量等权差值
为 +0.009328/+0.001333，CI `[0, 0.026119]`/`[0, 0.003376]`，双侧 p 为
0.4938/0.4969。去污染 88 条各自位于 88 个分量，差值 +0.011364/+0.001082，
CI `[0, 0.034091]`/`[0, 0.003247]`，p 均为 1.0。既有 65 条中 56 条可评估，
归入 55 个分量；差值 +0.022727/+0.003247，CI `[0, 0.063636]`/
`[0, 0.008225]`，p 为 0.5071/0.4959。其独立可评估部分只剩 41 条 SciFact
单例，差值 +0.024390/+0.002323，CI `[0, 0.073171]`/`[0, 0.006969]`，p
均为 1.0。所有视图 Candidate Recall 差值均为 0。

cluster-aware 结论与旧 query 级结论不变：点估计方向为正，但全部区间包含 0，
有效改善只来自极少分量，不能证明收益超出分量级符号波动。Record160 最大的
14-query 分量效应为 0，不是结果来源；但 Recall 的两个非零分量中单个分量占
绝对总贡献 80%，仍显示明显稀疏性。按 Record160 分量差值标准差 0.08889 和冻结
1000 条的 715 个分量，检测绝对 Recall 提升 0.01 的正态近似估计为约 85.3%
power，达到 80% 约需 621 个分量；该估计仅用于规划，且结构性 query ESS 约
98.56，不能当作正式效果证据。人工 Precision 完成前策略继续默认关闭，统计不显著
也不等于无效果。

版本化基准位于
`benchmark/lexical_normalization_cluster_significance_baseline/`，摘要为
`benchmark/lexical_normalization_cluster_significance_result.json`。回归门禁同时
检查冻结输入、实现 hash、逐 query 归簇、逐分量贡献及统计结果，且禁止网络、LLM
和 Snapshot 写入：

```bash
PYTHONPATH=src python scripts/audit_cluster_significance.py check \
  --manifest benchmark/lexical_normalization_cluster_significance_manifest.json \
  --output outputs/benchmark_runs/lexical_cluster_significance_gate

PYTHONPATH=src pytest -q -m cluster_significance_regression
```

`scripts/audit_lexical_normalization_expanded.py` 将同一冻结词法规则扩展到
AutoScholarQuery 1000 条基线运行中已经落盘的 162 条 Record 前缀。该审计只读
Record、retrieval Snapshot 与 gold 身份基线：每个实际 key 都重新校验 source、
adapted query 和 Snapshot 终态，随后在同一候选身份、顺序、QueryAnalysis 与
来源终态上重算默认过滤和 `lexical_normalization_v1`。Record 中没有成功来源的
2 条只保留终态，不进入指标；其余 160 条按成功来源数 1/2/3/4 分为
57/72/30/1。指标明确固定为 `deduplicated_gold_identity_v2`，不是官方 scorer，
也不能称为 1000 条正式基线或独立随机验证。

160 条共有 392 条查询内去重 gold 关系，候选命中 19 条，Candidate Recall 两组
均为 0.06885。Recall@20 从 0.03979 升至 0.04760（差值 0.00781，95% paired
bootstrap CI `[0, 0.021875]`，双侧 permutation `p=0.5`），F1@20 从
0.006564 升至 0.007680（差值 0.001116，CI `[0, 0.002827]`，`p=0.5`）；
改善/持平/退化为 2/158/0。排除既有 Auto dev/val 的 15 条重叠后，145 条新增
样本仍为 1/144/0，但 Recall/F1 的双侧 `p` 均为 1.0。收益方向与先前 65 条审计
一致，但仍只由极少数查询驱动，不能解释为显著或稳定效果。

实验恢复 2 条已召回但被默认过滤的 gold，同时放行 337 个 qrels 未匹配候选；
这些未匹配项不是可靠人工负例。两轮审计均为 0 网络、0 LLM、0 Snapshot 写入，
四份产物逐字节一致；`case_comparison.jsonl`、`candidate_diagnostics.jsonl`、
`aggregate.json` 的 SHA-256 分别为
`cc2372e329055fa82750e6bb0eb2bf0f33322c54a389952d032746d6d0a6a5d2`、
`5a61ab557f5a36eb1181f921d940fdd74d9ea42b6c708341a31cba5b70ea205a`、
`9ab00c3b4480f66bfaf1c12f3c33e3c1019d50910b5ea2945c7f612836e408b0`。
冻结协议和摘要见 `benchmark/lexical_normalization_record160_{manifest,result}.json`，
完整产物位于
`outputs/benchmark_runs/lexical_normalization_record160_813cf3a_r{5,6}/`。
人工 Precision 完成前策略继续默认关闭。

`scripts/build_lexical_record160_precision_package.py generate` 为上述 160 条扩大
验证建立穷举的 Top-20 变更盲标包。生成器只读取冻结 Record、Retrieval Snapshot
与两组已冻结的返回身份集合，以无 gold 的方式重建 query、title、abstract、year；
它不读取 candidate diagnostics 中的 `is_gold`/qrels 字段，也不发起网络、LLM 或
Snapshot 写入。纳入条件固定为跨越 Top-20 边界：实验换入 339 个关系、基线换出
132 个关系，共 471 个 query-paper 关系；双方共有但只发生榜内排名变化的论文不
进入本包。

公共包 `benchmark/lexical_normalization_record160_precision_annotation/` 仅公开
`sample_id/query/title/abstract/year` 五个扁平字段。全部 471 个关系中，439 个
生成新的随机化盲标项；另 32 个通过“规范 query + 统一身份等价”命中既有 65 条
审计的 200 项包，私有 mapping 引用原 sample ID 而不重复标注。递归字段泄漏检查
为 0，覆盖关系 439+32=471、未覆盖 0。两次生成的公共包、私有映射、双人模板、
仲裁模板和 manifest 逐字节一致，包树 SHA-256 为
`3d8df8680acdbe078eb175c31addeddb8feb106cf7a5273803424d75077fad94`。

两位标注者继续使用 `relevant / partially_relevant / not_relevant /
insufficient_information` 四分类并独立作答，分歧项必须仲裁。`score` 子命令在新包
和被引用旧包的人工标签均闭合后计算变更项两组 precision、Top-20 配对差值、换入
误放率、换出相关率、Cohen's kappa 及成功来源数/既有 dev-val 重叠分层。由于本包
刻意不重复标注双方共有的 Top-20 论文，绝对 Precision@20 在该 change-only 包中
保持 null；配对差值中的共有项严格抵消。当前所有模板为空，因此 Precision、差值、
误放率、相关率与 kappa 全部为 null，不形成任何人工精度结论。

`scripts/check_current_rules_regression.py check` 是默认 `current_rules` 的只读持续回归门禁。版本化协议和完整逐 case 语义基准分别位于 `benchmark/current_rules_regression_manifest.json` 与 `benchmark/current_rules_regression_baseline.json`；基准固定 SciFact 50 条、AutoScholarQuery dev 0/10、val 10/5 的四源 Replay，关闭 Query Evolution、RefChain、Semantic Seed、LLM 与其余实验开关。命令不构造网络连接器、不加载运行时密钥，也不调用 SearchService 的可写 Snapshot runtime；它只从 required retrieval key 读取冻结响应，再用当前统一身份去重、`current_rules` Judgement、Reranker、Top-20 选择与 evaluator 重建语义结果。

门禁逐次校验数据、原始配置/结果和 Snapshot tree 指纹，要求 SciFact/Auto dev/val 的 264/72/34 个 required retrieval key 与各自 group 完全一致、0 missing、0 reference key，并逐 key 比较来源、查询和成功/失败终态。语义基准逐 query 固定候选身份集合、最终返回顺序、Candidate Recall、Recall@20、F1@20、matched gold 和互斥 gold 终态；manifest 同时记录候选、核心指标、gold diagnostics 和来源终态的分区 SHA-256。`started_at`、代码提交/运行时 hash、resume signature 以及仓库外临时根路径显式排除或规范化，其他字段精确比较。漂移报告递归定位到最小 JSON path；候选/required-key 集合额外输出 added/removed，不允许用自动刷新基准掩盖回归。

当前两次门禁均检查 65/65 case，0 drift、0 HTTP、0 LLM、0 Snapshot 写入，`observed_profile.json` 与 tracked baseline 字节相同，SHA-256 为 `393ba0b2f314ce4b0fc792de86e6da67fa97318ffce7f3c42621c3f3e4a29f2f`；`regression_report.json` 为 `08018899b0571cc7d974e53add8019314304fd8d172f0fe35a6528e1f9908750`。冻结核心指标 Candidate Recall/Recall@20/F1@20 分别为 SciFact `0.12195/0.12195/0.01161`、Auto dev `0.15/0.125/0.01786`、Auto val `0.2/0.2/0.01905`。这些只是内部回归锚点，不是官方成绩，也不替代独立 1,000 条正式验证。

基准更新不属于普通门禁。独立的 `propose-baseline` 子命令必须同时提供固定 approval token 和可审计 reason，只输出 `proposed_baseline.json` 与 `baseline_update_audit.json`，不会修改 manifest、tracked baseline 或 Snapshot；任何采纳仍需单独人工审查与提交。快速门禁可执行 `PYTHONPATH=src pytest -q -m regression_gate`，外部 Record 永不进入 pytest 流程。

AutoScholarQuery 全量规划门禁使用独立的 query-only 输入 `benchmark/autoscholar_query_planning_input.jsonl`，字段契约只允许 `query_id/query`；它不调用 AutoScholarQuery evaluator adapter，也不读取、复制或统计 gold。`scripts/audit_autoscholar_query_planning.py check` 直接调用关闭 LLM 的 `current_rules` 规则规划器，固定 balanced、Top-20、四个默认来源、2026 有效年份及默认 `SearchBudget`，只执行 query understanding、约束抽取和 SearchPlan 构造，不进入 SearchService、connector、Judgement、排序、evaluator 或 Snapshot runtime。socket/LLM 调用护栏及 Snapshot 树前后签名把任何意外外部调用或写入转为门禁失败。

版本化协议 `benchmark/autoscholar_query_planning_manifest.json` 固定原数据版本、1000 条有序 query-only manifest、Planner/Prompt manifest 版本、默认关闭的实验开关、逐 query plan hash、完整 SearchPlan baseline 和结构化质量汇总。每条计划都经过 Pydantic JSON round-trip；门禁精确比较内容、顺序、预算、终态和汇总并输出最小 JSON path diff。wall-clock 单条延迟只写入明确排除于回归的 `runtime.json`，不进入计划或汇总哈希。普通门禁不能更新 baseline；`propose-baseline` 只生成待审查产物。快速运行命令为 `PYTHONPATH=src pytest -q -m planning_regression`。该审计不生成 Recall/F1，也不构成检索效果或官方成绩验证。

冻结结果为 1000/1000 Schema 与 JSON round-trip 成功、0 错误、0 warning、0 空查询、0 重复子查询、0 缺字段和 1000/1000 预算一致。共生成 2410 条子查询：590 个 case 为 2 条、410 个为 3 条；四源逻辑请求槽总数 9640，410 个三查询 case 的请求返回容量上界 240 会由既有全局 200 候选预算裁剪，但没有增加配置预算。文本解析覆盖 method 305、dataset 65、time range 9 个 case；数据没有独立 API 显式约束输入。两次实测单条规则规划 p95 为 0.659/0.702 ms、p99 为 0.770/1.206 ms，最大值 7.162/7.409 ms；孤立最大值未改变计划内容，长尾中 3 子查询 case 略多，未发现结构化失败模式。两次 `plans.jsonl`、`summary.json` 和回归报告逐字节一致，前两者 SHA-256 分别为 `8442fd2ddba1ef29615749f7a1a75a77cb1f9393f9fc2bff9c96730dee198b37` 与 `00f2d2246dce4b2ee123aa0ffaef90215a52f9ca67b147d460ee8761281bb766`。

AutoScholarQuery 全量 gold 身份输入门禁由
`scripts/audit_autoscholar_gold_identity.py` 提供，只在 evaluator 隔离层读取
1000 条数据的 2403 条 gold；它不导入 SearchService、connector、排序或 Prompt，
也不运行 Recall/F1。五类互斥终态固定为稳定 ID 可评估、严格标题-作者-年份
证据可评估、同类标识冲突、身份歧义和信息不足。稳定标识与无标识时的保守
标题证据完全复用 `identity.py`；不使用模糊标题、外部 crosswalk 或人工补全。

版本化协议、逐 gold/逐 query 基准和汇总分别位于
`benchmark/autoscholar_gold_identity_manifest.json`、
`benchmark/autoscholar_gold_identity_baseline/` 与
`benchmark/autoscholar_gold_identity_result.json`。门禁固定原数据、query-only
manifest、统一身份实现、evaluator 与审计实现 SHA-256，逐关系比较终态、稳定
标识、标题证据和身份 cluster，逐 query 比较原始/evaluator/安全去重计数；gold
增删、身份变化、计数或实现漂移均输出最小 JSON path 并失败。普通 check 不能
更新基准；独立 `propose-baseline` 只生成待审查产物。

冻结结果中 2403/2403 关系均凭 arXiv ID 进入稳定 ID 可评估终态，0 冲突、
0 歧义、0 信息不足，1000 个 query 均至少有一个可评估 gold。统一身份得到
2009 篇全局唯一论文，394 条是跨全量数据重复关系；其中 268 篇跨 query 复用，
涉及 657 条关系。历史 `legacy_gold_records_v1` evaluator 不在 query 内预去重：
2 个 query 合计有 5 条重复 denominator，原始 2403 条关系对应安全 query 内
去重 denominator 2398。
数据除标题与 arXiv ID 外均缺作者、年份、DOI、PMID、OpenAlex、S2 和 S2ORC
字段；这些缺失不影响本版精确 arXiv 身份可评估性。该审计只冻结 evaluator
输入质量，不是检索效果或官方成绩。

该目录现作为不可变的 v1 历史基准保留；启用 v2 后执行其旧 check 会按设计报告
evaluator 指纹及上述 2 个 query 的 denominator 漂移，不能通过更新旧产物来消除。
当前指标语义门禁使用下节的 v2 manifest。

```bash
PYTHONPATH=src python scripts/audit_autoscholar_gold_identity.py check \
  --manifest benchmark/autoscholar_gold_identity_manifest.json \
  --output-dir outputs/benchmark_runs/autoscholar_gold_identity_gate

PYTHONPATH=src pytest -q -m gold_identity_regression
```

### Gold 分母指标版本

内部 Benchmark 自 `deduplicated_gold_identity_v2` 起，在匹配与计算分母之前按
`identity.py` 对每个 query 的正向、可评估 gold 去重。稳定标识相交且无同类
标识冲突时合并；无稳定标识时仍只接受严格标题、作者与年份证据。一个 identity
cluster 只贡献一个 denominator，并合并成员的稳定标识用于匹配，graded relevance
保留 cluster 中最高等级。输入顺序不影响计分。旧 JSON 缺少 `metric_version` 时
按 `legacy_gold_records_v1` 解析；历史 current_rules 回归门禁也显式使用 legacy，
因此旧结果可读且不会被新口径冒充或覆盖。v2 是内部指标，不是官方 scorer。

迁移审计由 `scripts/audit_gold_metric_semantics.py` 提供，版本化协议与冻结结果位于
`benchmark/gold_metric_semantics_manifest.json`、
`benchmark/gold_metric_semantics_v2_baseline/` 和
`benchmark/gold_metric_semantics_result.json`。门禁同时保护历史 current_rules 与
gold identity v1 产物 SHA-256，验证候选身份、返回顺序、来源终态和 Snapshot key
在两种版本间完全一致；只允许 denominator 和由其派生的指标变化。全量 1000 条
由 2403 降至 2398，5 条被移除关系集中在 2 个 query，全部依据精确共享 arXiv
ID。SciFact 50、Auto dev 10、Auto val 5 未包含重复关系，故三个冻结集合的
Candidate Recall、Recall@20 与 F1@20 均无数值变化。

```bash
PYTHONPATH=src python scripts/audit_gold_metric_semantics.py check \
  --manifest benchmark/gold_metric_semantics_manifest.json \
  --output-dir outputs/benchmark_runs/gold_metric_semantics_gate

PYTHONPATH=src pytest -q -m metric_semantics_regression
```

`scripts/audit_autoscholar_snapshot_resume.py` 将上述 query-only 计划、冻结的 baseline `plan_round_2`、现有 retrieval Snapshot 与 Record 产物的顶层 `case_id/status` 合并为 gold-blind 缺失审计。Record JSONL 的其他字段由结构扫描器直接跳过，不加载数据集 adapter 或 evaluator。每个 required key 都重新计算 Snapshot key，并在已有文件上校验 source、规范查询、limit、adapter/query-adapter/connector 版本与 content hash；四类终态固定为已有 `success`、已有 `failed`、已结束 Record case 的 `missing`，以及尚未进入 Record case 的 `not_started`。来源、query-only manifest 顺序四分位、查询长度秩四分位、子查询数与 method/dataset/time 约束仅用于缺失机制审计，不生成 Recall/F1。

版本化 `benchmark/autoscholar_full1000_resume/resume_manifest.json` 只调度 frozen failed、missing 与 not-started key；成功 key 永不覆盖，冻结失败 key只统一重试一次。调度使用按各来源剩余总量归一化的确定性公平轮转，并以配置来源顺序破除并列；每个来源内部按 query-only manifest case 顺序轮转，同 case 仍有替代项时避免相邻发送。canonical Runner 仅在显式提供 `--resume-manifest` 时进入该路径，重新校验 required plan hash、key/request signature 与全部 retrieval 语义配置；`--resume-manifest-dry-run` 在加载项目环境之前返回进度，0 网络、0 Snapshot 写入。实际执行仍使用原 connector、request body、limit、重试与 Snapshot key，只改变跨请求调度顺序；无参数时原 Benchmark 路径不变。

网络恢复后的只读进度检查命令如下；移除 `--resume-manifest-dry-run` 才会按 manifest 串行补采，执行前必须再次确认外部来源可用：

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --dataset auto_scholar_query --dataset-split test --limit 1000 --offset 0 \
  --run-id autoscholar-full1000-resume --run-profile balanced \
  --sources openalex,arxiv,semantic_scholar,pubmed \
  --result-policy highly_and_partial --top-k 20 \
  --query-adapter-policy adaptive --query-planning-policy current_rules \
  --ranking-policy current_rules --judgement-policy current_rules \
  --query-evolution-policy off --retrieval-mode record-missing \
  --snapshot-dir outputs/benchmark_snapshots/autoscholar_current_rules_full1000_3cd47c1 \
  --resume-manifest benchmark/autoscholar_full1000_resume/resume_manifest.json \
  --resume-manifest-dry-run
```

`scripts/build_lexical_precision_annotation.py` 为该实验建立默认无标签的盲化人工 Precision 闭环。固定 manifest 使用三个数据集和三类 strata：规范化新增返回、baseline 独有返回、双方共有且返回列表名次绝对变化至少 5；总上限 200，按固定 dataset/stratum 顺序做均衡水位分配，每个 cell 内用固定 seed 的 SHA-256 顺序选择，再用独立命名空间随机化展示顺序。抽样和包生成只重建冻结候选、Judgement 与 Top-20，不访问 gold/qrels、connector、LLM 或 Snapshot 写路径。

正式盲包从 502 个 eligible query-paper 对中抽取 200 个唯一对：SciFact/Auto dev/val 为 84/71/45，新增/baseline 独有/共享显著变位为 71/48/81。公开 `blind_samples.jsonl` 每行严格只有不可编码隐藏字段的顺序 sample ID、query、标题、摘要和年份；策略、排名、case ID、来源、评分与 evaluator mapping 只存在于隔离的 `private/`。两位标注者分别使用四分类模板独立完成，分歧项才进入第三方仲裁。评分 CLI 在标签完整前只返回 `pending_human_labels` 和 null 指标；标签完成后计算仲裁前 Cohen's kappa、样本 Precision、带抽样覆盖说明的分层估计、新增候选误放率，并且仅在全部 Top-20 pair 均被人工覆盖时输出非空的完整 Precision@20。

两次生成的十个文件逐字节一致，包 tree SHA-256 为 `08a12db33a6d4705af1ccb2978437eea857290a45e963046d0ee3730c232c7a5`。冻结协议、待标注材料和私有映射位于 `benchmark/lexical_normalization_precision_annotation_manifest.json` 与 `benchmark/lexical_normalization_precision_annotation/`；统计与人工前置动作见 `benchmark/lexical_normalization_precision_annotation_result.json`。当前没有人工标签，因此不得报告 Cohen's kappa、Precision 或误放率，也不得据此改变默认策略。

冻结 Record160 全量 Top-20 变更包还提供独立的
`llm_relevance_judging_v1` 内部代理评审入口。协议把 439 个当前公开样本和 32 个既有包
引用绑定为 471 个 opaque item；评审消息只含 query、title、abstract、year，完整剥离
策略、方向、来源、排名、gold/qrels 和 case/query ID。两轮固定微批评审后只裁决标签
分歧，所有 item 闭合并生成 hash lock 后才允许离线解盲。失败和
`insufficient_information` 不会被强制当作负例。

该盲包不包含双方共有 Top-20 候选，因此绝对 arm Precision@20 始终为 null；评分只报告
变更项的 LLM-proxy 正类率及 candidate-minus-baseline 的 Top-20 正类贡献差值，并复用
冻结查询连通分量做 20,000 次固定种子 bootstrap/sign-flip。结果不是人工 Precision、
不是官方 scorer，也不自动改变 evidence registry。协议、resume/usage 语义与命令见
[`docs/llm-relevance-judging.md`](llm-relevance-judging.md)。

本次模型运行的第一轮仅锁定 447/471 项，3 个批次、24 项在两个有界 attempt 后仍为
严格 Schema failure；第二轮、裁决、解盲和效果统计因此均未执行。该结果是可审计的
不完整终态，不得据此给出 LLM-proxy Precision、置信区间或策略收益。脱敏调用证据和
完整缺失清单位于 `benchmark/llm_relevance_judging_v1_record160_incomplete/`。

后继 `llm_relevance_judging_v1_1` 保留 rubric、标签枚举、盲化字段、覆盖门槛、解盲条件和
cluster-aware 统计口径，仅加固结构化输出。v1 的脱敏证据只能确定 8 次严格 Schema 无效
与 1 次 batch member mismatch；所有 65 次调用均为供应商接受的原生 `json_object`，未
发生兼容回退。因未保存响应原文，不能把失败进一步断言为截断。v1.1 因此采用预注册的
单项请求、唯一顶层 `labels`/`decisions` 对象、精确键集合、固定 1024 token 运行配置摘要、
两次统一 attempt 和 1 秒逻辑退避；malformed 输出只记录 hash 与失败分类，禁止提取、修复
或补造标签。

v1.1 的 471 项必须从头生成新的 opaque identity，不读取 v1 的 447 个部分锁定结果。只有
第一轮、第二轮及全部分歧裁决依次完整闭合并生成 labels lock 后，评分器才可读取私有 arm
映射；任一阶段失败时 labels/statistics 继续为 null。冻结协议位于
`benchmark/llm_relevance_judging_v1_1_protocol.json`，默认 CLI 使用独立运行目录。

本次 v1.1 真实第一轮已按同一协议执行全部 471 项，291 项严格锁定、180 项在统一两次
attempt 后终止：106 项最终为 Schema failure，74 项最终为 provider failure。全部 689
次逻辑调用中，失败 attempt 细分为 233 次严格 Schema 无效、2 次 opaque identity
mismatch、155 次 HTTP 429 和 8 次 HTTP 503；供应商明确报告 391,725 prompt、55,507
completion、447,232 total tokens，费用不可用。因第一轮未达到 471/471，第二轮、裁决、
labels lock、解盲和统计均未启动，所有 LLM-proxy 效果值继续不可用。脱敏证据位于
`benchmark/llm_relevance_judging_v1_1_record160_incomplete/`。

`judge_backend_qualification_v1` 在不读取历史部分标签和响应正文的前提下，进一步对
v1/v1.1 脱敏失败证据做描述性归因，并对运行时已配置候选执行最多 24 次固定合成
canary。门禁只验证原生结构化模式、严格 Schema、opaque item binding、无 fallback/额外
HTTP attempt 及供应商 usage 可审计性；不生成相关性标签、效果统计、人工 Precision 或
官方成绩。资格阈值、canary、Prompt 和请求参数都在真实调用前由版本化 protocol 与
SHA-256 固定。只有 `probe` 加载项目正常 LLM 配置并调用配置的 LLM API，其余阶段离线；
所有阶段禁止学术 API 和 Snapshot 写入。协议、退出码与解释边界见
[`docs/judge-backend-qualification.md`](judge-backend-qualification.md)。

本次唯一配置候选 `openai_compatible` / `deepseek-ai/deepseek-v4-flash` 完成固定 24 条：
18 条 provider failure（11 timeout、7 HTTP 503），6 条通过 Schema/item binding，其中仅
3 条同时满足单 HTTP attempt，严格成功率为 12.5%。候选不合格，供应商只为 6 条响应
报告 2,157 tokens，费用不可用；没有标签、效果统计或后续全量运行建议。

`analyze_query_planning_policies.py` 将策略汇总、facet 贡献、逐查询原始/适配查询、候选、事后 gold、重复率、记录成本和无效原因写入 `outputs/benchmark_runs/initial_query_planning_analysis/`。四次 Replay 的实际 HTTP、重试和网络等待均为 0；gold 只在运行后参与诊断。

增加 `--diagnostics` 后，Runner 还会写入 `stage_metrics.json`、`error_analysis.json` 和 `gold_diagnostics.jsonl`。阶段快照只包含论文标识符、标题、年份、来源、子查询 provenance、rank、Judgement 分类与分数，不保存摘要或 Prompt；gold 只在 SearchService 返回后参与统一 identifier matching。

固定开发诊断使用 AutoScholarQuery 原始顺序前 10 条、`top_k=20`、关闭 LLM，并统一使用两轮、150 候选和 90 秒预算。当前完成了 arXiv-only 与三源 baseline；后者受到 Semantic Scholar 429、OpenAlex 400/超时影响，来源错误率为 0.402。Query Evolution 配置启动后仍持续失败，已按公共服务保护要求中止；RefChain 和 full 配置未运行，不生成替代结果。

两组完整结果都显示主要问题是初始检索召回：候选 Recall 分别为 0.150 和 0.170；三源 baseline 的端到端 F1@20 从 0.0179 增至 0.0259，但平均 API 请求从 2.7 增至 8.2、平均延迟从 4.01 秒增至 29.78 秒。10 条数据只用于开发诊断，不代表完整 Benchmark 或比赛成绩。

加入来源查询适配、run 内去重和限流降级后，使用同一 10 条、同一预算完成 A2/B2。A2 相对 A：候选 Recall 0.150→0.025、F1@20 0.0179→0.0083、平均 API 2.7→2.4、来源错误率均为 0、平均延迟 4.01→10.57 秒；延迟增加主要来自 arXiv 的 3 秒请求间隔。B2 相对 B：候选 Recall 0.170→0.045、F1@20 0.0259→0.0163、平均 API 8.2→5.1、来源错误率 0.402→0.059、平均延迟 29.78→19.93 秒，OpenAlex 400 从 10 次降为 0，Semantic Scholar 请求从 51 次降为 8 次。可靠性和成本改善，但该固定小样本上的召回与 F1 下降，不能声明质量提升。

查询适配回归修复后，A3 使用 `hybrid` 在相同前 10 条恢复候选 Recall 至 0.250，Recall@20 为 0.225、F1@20 为 0.0274，平均 API 3.6、平均延迟 32.73 秒、来源错误率为 0。三源 B3 因 Semantic Scholar 连续最终 429 在 9/10 安全停止；其部分指标不得与完整 B/B2 等同。独立原始顺序 10–14 的 5 条验证中，`safe_original` 与 `hybrid` 的候选 Recall 和 Recall@20 均为 0.200，F1@5/10/20 均为 0.0667/0.0364/0.0190；该小样本只说明 hybrid 未丢失 safe-original 候选，不代表总体提升。

A4 在相同开发集重新对比三种策略：safe、hybrid、adaptive 的候选 Recall 分别为 0.150/0.250/0.150，Recall@20 为 0.125/0.225/0.125，F1@20 为 0.0179/0.0274/0.0179，平均 API 为 2.6/3.5/2.7，平均延迟为 34.27/30.96/25.37 秒。adaptive 执行 compact 1/26 次（3.85%），平均新增 20 个唯一候选，事后 gold 增量为 0；来源错误率均为 0。延迟受公开服务波动影响，不应仅凭单次 safe/hybrid 顺序作因果解释。

独立 V2 在三组全部完成后才读取 gold：三种策略的候选 Recall、Recall@20 和 F1@5/10/20 均为 0.200、0.200 和 0.0667/0.0364/0.0190。adaptive 平均 API 2.2，与 safe 相同并低于 hybrid 的 3.0；平均延迟 50.50 秒，低于 hybrid 的 54.00 秒；compact 执行 0/12。该 5 条结果支持保留 adaptive 默认值，但样本太小，不能外推到完整数据集。

M4 使用 arXiv/OpenAlex 和 adaptive：候选 Recall 0.170、Recall@20 0.145、F1@20 0.0259、平均 API 5.7、平均延迟 54.61 秒；compact 执行 1/52，新增 20 个唯一候选且事后 gold 增量为 0。OpenAlex 出现两次最终超时和一次最终 429，来源错误率为 0.0526，HTTP 400 为 0；这次公开服务波动不应归因于 compact 请求。当前环境无 Semantic Scholar Key，未运行三源配置。

Runner 的 `--query-adapter-policy` 支持 `safe_original`、`hybrid` 与 `adaptive`，并把实际值写入 `config.json`。`adaptive` 先执行安全原查询，仅在候选数量、核心维度、约束覆盖或元数据不足时补充核心查询；预算、冷却、等价查询和低信息保留会阻止第二请求。compact 执行率、新增唯一候选与事后 gold 增量进入阶段诊断，gold 不参与在线决策。产品默认使用 `adaptive`。

## 运行命令

运行确定性 sample fixture 并生成 Markdown 汇总：

```bash
PYTHONPATH=src python scripts/eval_search_service.py \
  --fixtures-dir datasets/eval_fixtures/sample \
  --output-root outputs/eval_runs \
  --run-id sample

PYTHONPATH=src python scripts/summarize_eval_results.py \
  outputs/eval_runs/sample/result.json
```

评测已有 batch 结果：

```bash
PYTHONPATH=src python scripts/evaluate_search_batch.py \
  --batch-results outputs/manual_smoke/results.jsonl \
  --gold datasets/eval_fixtures/manual_smoke/qrels.filled.jsonl \
  --output outputs/manual_smoke/eval.json \
  --k 5 --k 10 --k 20 \
  --result-policy highly_and_partial
```

`--include-partial` 作为旧参数继续等价于 `--result-policy highly_and_partial`。与 `--result-policy highly_only` 同时使用会报错。

## 结构化输出证据追溯门禁

`scripts/check_structured_output_provenance.py` 对冻结 Replay 中已有的公共 API
结构化结果执行只读门禁，不加载 gold，也不调用 connector、LLM 或 Snapshot
写路径。版本化规则位于
`benchmark/structured_output_provenance_gate_manifest.json`，固定覆盖 SciFact 50
条与 AutoScholarQuery dev/val 15 条，并以公共 API 实际返回的
`highly_relevant_papers + partially_relevant_papers` 作为唯一可引用论文集合。

门禁先通过 `SearchRunResultResponse` 校验 Schema，再把返回论文的 rank 和统一
身份对齐到冻结 `final_ranked`，并从 retrieval Snapshot 重建原始论文以核验标题、
作者、年份、摘要、稳定标识、来源和 URL。身份判断只复用 `identity.py` 的稳定
标识或严格标题-作者-年份规则，不使用模糊标题。证据表须精确引用返回候选及其
原始 evidence 字段；finding、摘要引用、method group、timeline 和 citation graph
继续逐级引用这些已验证对象。任何越界引用、身份冲突、重复论文、顺序或统计漂移
都会形成 `failed_validation`，无法取得结构化输出或只读 Snapshot 时形成明确
`blocked_*`，不会删除失败 case 或把无法证实的内容静默算作通过。

```bash
PYTHONPATH=src python scripts/check_structured_output_provenance.py \
  --manifest benchmark/structured_output_provenance_gate_manifest.json \
  --output outputs/benchmark_runs/structured_output_provenance_gate
```

产物为 `case_gate.jsonl`、`provenance.jsonl` 和 `aggregate.json`；前两者分别
保留每个 case 的闭合终态及每项结构化声明的 source path。该门禁只证明冻结
输出与冻结候选之间的可追溯性，不证明论文事实本身正确，也不是官方赛题成绩。

## AutoScholarQuery query→gold 信息泄漏门禁

`scripts/audit_query_gold_leakage.py` 只在 benchmark/evaluator 隔离层读取 query 与
gold，不导入 SearchService、connector、排序或 Prompt。检测规则在查看统计前冻结于
`benchmark/autoscholar_query_gold_leakage_protocol.json`：query/title 统一做 HTML
反转义、Unicode NFKC、casefold、Unicode 标点分词和空白折叠；标题规则仅适用于
规范后至少 24 字符且至少 3 个 token 的标题。互斥优先级固定为稳定标识或 URL
精确出现、引号内完整标题、规范标题完整出现、标题 informative token 覆盖、未检测
到泄漏。高覆盖要求至少 5 个去停用词 token 且覆盖率不低于 0.8，不使用模糊匹配、
Embedding、外部搜索或人工词典。

版本化 manifest、逐关系/逐 query 基准和汇总位于
`benchmark/autoscholar_query_gold_leakage_manifest.json`、
`benchmark/autoscholar_query_gold_leakage_baseline/` 与
`benchmark/autoscholar_query_gold_leakage_result.json`。普通 check 会重新计算 1000
条 query 与 2403 条 query-gold 关系，校验原数据、gold identity 基线、协议、冻结
Replay 及 SciFact 输入 SHA-256，并逐 JSON path 比较终态、证据和汇总；数据、规则或
基准漂移都会失败。socket 护栏与 Snapshot 树前后签名保证运行是 0 网络、0 LLM、
0 Snapshot 写入。该门禁只审计内部评测有效性，不是检索指标或官方 scorer。

冻结统计为 15/2403（0.6242%）关系和 14/1000（1.4%）query 命中检测规则：
0 条标识/URL、0 条引号标题、5 条关系（涉及 4 个 query）规范标题完整出现、10 条
关系/10 个 query 高 token 覆盖。预注册风险阈值据此给出 moderate：未达到“直接
泄漏 query 至少 1% 或任意泄漏至少 5%”的 high 条件，但也不是低于 1% 的 low。
跨 query 重复涉及 657 条关系、268 个重复 identity；其中 5 条泄漏关系，5 个重复
identity 在不同 query 间呈混合泄漏状态，因此重复论文不应被当作独立污染来源。

冻结命中分层只复用已有命中诊断，不重算 Recall/F1。Auto dev 10 和 val 5 中均无
检测泄漏，已有最终命中分别为 2 和 1 条非泄漏关系。Record160 主分析中 2/160 query
命中泄漏规则，已有 candidate-hit 为泄漏 2、非泄漏 15，final-hit 为泄漏 1、
非泄漏 11；另有 2 条无成功来源保持单列。SciFact 50 中 1 条 query 命中规则，
candidate-hit 为泄漏 1、非泄漏 30，final-hit 为泄漏 1、非泄漏 18。小泄漏分层
出现命中集中迹象但样本过小，不能作因果或显著性结论。后续内部报告应同时给出
全量结果和非泄漏诊断分层，不自动删除样本，也不得用分层结果替代全量结果或冒充
官方成绩。

```bash
PYTHONPATH=src python scripts/audit_query_gold_leakage.py check \
  --manifest benchmark/autoscholar_query_gold_leakage_manifest.json \
  --output outputs/benchmark_runs/autoscholar_query_gold_leakage_gate

PYTHONPATH=src pytest -q -m query_gold_leakage_regression
```

## AutoScholarQuery 查询独立性门禁

`scripts/audit_query_independence.py` 在 evaluator 隔离层审计 1000 条 query 的重复、
共享 gold 和冻结证据分层，不调用 SearchService、connector、LLM 或 Snapshot 写路径。
检测协议在查看统计前冻结于
`benchmark/autoscholar_query_independence_protocol.json`：query 统一做 HTML 反转义、
Unicode NFKC、casefold、Unicode 标点分词和空白折叠；完全重复按规范文本相等；
近重复按去固定停用词后的 Unicode token 集 Jaccard 计算，双方至少 6 个有效 token，
阈值固定为 `>=0.8`。不使用 Embedding、语义模型、外部搜索或人工合并。

图的无向边仅来自规范 query 完全重复、预注册词法近重复或统一 identity cluster 的
共享 gold；连通分量使用传递闭合，每个 query 恰好属于一个稳定 hash 标识的分量。
分层原始 membership 可重叠；独占 partition 固定按 Auto dev、Auto val、Record160-only、
其余样本的优先级建立。一个分量只要包含跨冻结 membership 复用的同一 query，或
连接多个独占 partition，即标为跨层污染。该标记只生成与全量并列的诊断视图，
不删除数据、不重划正式 split。

冻结结果没有规范完全重复或词法近重复 query；715 个连通分量中的 103 个非单例
分量全部由共享 gold 形成，涉及 388 条 query，最大分量 89 条。268 个共享 gold
identity cluster 形成 506 条 query-pair 边，其中 83 个 identity cluster、135 条边
跨独占分层。46 个分量、237 条 query 被标为跨层污染，剩余 763 条独立。Auto dev
10 条和 val 5 条全部与 Record160 直接复用同一 query，因此二者不能被当作相互独立
验证证据；Record160 主分析 160 条中，协议过滤后保留 88 条独立诊断样本。

现有 65 条全量内部诊断的 Candidate Recall 为 0.5982，baseline/lexical 的
Recall@20 为 0.3795/0.4018、F1@20 为 0.03875/0.04194；去除跨层分量后只剩不属于
AutoScholarQuery 图的 SciFact 50 条，不能解释为 Auto dev/val 的独立成绩。
Record160 全量 Candidate Recall 为 0.06885，baseline/lexical Recall@20 为
0.03979/0.04760、F1@20 为 0.006564/0.007680；88 条去污染诊断对应 0.06477、
0.03068/0.04205 和 0.004936/0.006018。共享 gold 非单例簇在 Record160 的 candidate
hit 率为 9.52%，低于单例的 11.34%；现有命中没有集中在词法重复 query，因为本版
没有检测到此类 query。以上均为内部冻结诊断，不是官方成绩。

版本化 manifest、逐 query/边/分量/指标基准和摘要位于
`benchmark/autoscholar_query_independence_manifest.json`、
`benchmark/autoscholar_query_independence_baseline/` 与
`benchmark/autoscholar_query_independence_result.json`。门禁校验数据、协议、identity
基线、冻结 Replay、实现与所有产物 hash，并逐 JSON path 报告数据、阈值或簇归属
漂移。后续跨分层显著性分析不得把复用 query 当作独立观测；应并列保留全量结果和
预注册去污染视图，并以连通分量作为重采样单位。

```bash
PYTHONPATH=src python scripts/audit_query_independence.py check \
  --manifest benchmark/autoscholar_query_independence_manifest.json \
  --output outputs/benchmark_runs/autoscholar_query_independence_gate

PYTHONPATH=src pytest -q -m query_independence_regression
```

## 实验证据注册表与默认策略门禁

`benchmark/evidence_registry_manifest.json` 冻结 `experiment_evidence_registry_v1`
的 schema、排序、扫描范围和策略枚举。范围覆盖当前代码中具名且可选择的查询规划、
Query Evolution、引用/推荐扩展、排序、Judgement、词法证据匹配、query adapter、
结果过滤、本地来源以及 benchmark-only 检索实验；运行 profile、任意来源组合和不改变
检索策略的结构化输出归纳不作为独立策略项。枚举直接从对应 Literal 类型、来源常量和
显式 feature 名称构造，新增实现却没有证据记录会使门禁失败。

每条记录都包含实现/评测 commit、默认状态、数据范围、唯一配置差异、指标版本、已跟踪
证据及其 SHA-256、历史 Replay/产物 hash（存在时）、调用完整性、核心内部指标、效率、
结论和阻断。只有存在当前仓库可验证机器产物的记录才能写入核心指标；历史文档保留但
原始产物未跟踪时必须使用 `evidence_unavailable`、清空指标/效率并保留
`tracked_primary_artifact_unavailable` 阻断。负面、阻断、不可判定和未验证结论均不得
隐藏或提升为通过证据。

冻结矩阵位于 `benchmark/evidence_registry_baseline/`：当前恰好 24 项，其中 4 项有
已跟踪机器证据、20 项为 `evidence_unavailable`；决策为 1 项 validated default、
2 项 promising default-off、11 项 negative、4 项 inconclusive、1 项 blocked、
5 项 unvalidated。唯一允许默认开启的是 `current_rules`。任何实验开关默认开启、
`current_rules` 缺少通过证据、证据文件/hash 漂移、指标版本漂移、重复/冲突记录或策略
遗漏都会输出最小 JSON path diff 并失败。

注册表只汇总内部 Benchmark 与审计证据，不是官方赛题成绩。官方材料仍缺少精确
scorer、F1/K、身份去重与平均口径，因此 `official_scorer_unavailable` 作为全局阻断
保留。CLI 全程只读 Git 跟踪文件，并以 socket 护栏和 Snapshot 树前后签名保证 0 网络、
0 LLM、0 Snapshot 写入、0 Benchmark；显式基准文件不在自身证据扫描范围内，避免
自引用 hash 漂移。

```bash
PYTHONPATH=src python scripts/check_evidence_registry.py check \
  --manifest benchmark/evidence_registry_manifest.json \
  --output outputs/benchmark_runs/evidence_registry_gate

PYTHONPATH=src pytest -q -m evidence_registry_regression
```

人工评审开始前可使用 `human_annotator_qualification_intake_v1` 对 annotator-A、
annotator-B 和 adjudicator 做合成 rubric 校准及结构化冲突检查。该资格门禁不加载真实
471 项盲标内容，不生成真实标签、Precision 或官方成绩；合格输出只是角色分配提案，
仍需后续真实双人完整标注、合法裁决及既有人工门禁验证。

`human_annotation_assignment_activation_v1` 只验证合格匿名角色与冻结 A/B 盲包、
adjudicator rubric 包的一次性签发、确认和 append-only 状态链。12 项合成攻击矩阵不
填写真实标签；三份真实 qualification/acknowledgement 未闭合时标签导入固定禁用，
readiness 返回 exit 3。该门禁不计算 Precision，不改变 adjudication 统计门槛，也不
解除正式人工 Precision 阻断。

## 运行产物谱系与完整性门禁

新 Benchmark/Replay 运行可使用 `run_manifest_v1` 绑定数据输入 SHA-256、query-only
身份与顺序、Prompt 版本、来源/预算、evaluator 版本、确定性参数、完成进度、checkpoint
父子链、Git 工作区状态及封闭输出文件清单。`scripts/check_run_provenance.py` 提供
`generate`、`validate` 与 `audit-legacy` 三个离线入口；校验能定位文件缺失/篡改/未登记、
query 重排、配置元数据漂移、完成数不足和谱系断裂/循环。输出仅是内部可复现性证据，
不是官方 scorer 或正式比赛成绩。

现有 AutoScholarQuery 160 分析输入及未完成 1000 条运行没有这些字段，文件只读核验
结果固定为 `legacy_metadata_incomplete`（退出码 3），不得补造 v1 manifest 或宣称通过。
完整字段、退出码、新运行生成方法与当前 legacy 口径见
[`docs/run-provenance.md`](run-provenance.md)。离线回归入口为：

```bash
PYTHONPATH=src pytest -q -m run_provenance_regression
```

## 实验对照隔离与配对完整性

新成对离线实验使用 `comparison_plan_v1` 预先声明唯一处理变量及完整 query population，
并把计划文件哈希绑定进两侧 `run_manifest_v1` 和原子提交代。离线门禁逐项验证 Prompt、
来源、预算、seed、并发、超时、重试、约束、evaluator、规范化与执行 profile 等共同
契约，只允许计划中精确叶子 JSON Pointer 对应的差异。query 成功、失败、取消与预声明
排除均保留配对；单侧缺失、重复、后验排除、非对称 resume 或仅共同成功分析都会失败。

该检查不读取 gold/qrels，不计算 Candidate Recall、Recall/F1 或显著性。配对完整性通过
只是后续统计的前置条件，不能代替人工 Precision 或官方 scorer。旧 Record160/Full1000
缺少预绑定计划时固定返回 `not_eligible`。契约、CLI 和退出码见
[`docs/experiment-pairing-integrity.md`](experiment-pairing-integrity.md)。

```bash
PYTHONPATH=src python scripts/check_experiment_pairing.py check-fixture
PYTHONPATH=src python scripts/check_experiment_pairing.py audit-registry
PYTHONPATH=src python scripts/check_experiment_pairing.py audit-frozen
```

## 离线分片执行与确定性归并

新式长任务可用 `shard_plan_v1` 在执行前冻结完整 opaque query 集合、全局顺序、
数据/Replay 摘要、共同配置及固定 `ordered_round_robin_v1` 分配。每个 shard attempt
必须把 plan SHA、shard 编号、预期 query 摘要和显式 supersession 关系同时绑定到
`run_manifest_v1` 与 generation-zero；没有绑定的历史运行不能后验认定为合法分片。

`scripts/check_sharded_execution.py` 检查分区闭合、attempt 唯一末端、共同配置、提交代、
完整终态和全局顺序，并可生成只读 `shard_aggregate_v1`。aggregate 不会过滤失败、取消或
排除项，不会选择表现较好的重试；任一 shard 未完成时退出码为 3 且不生成 completed
产物。门禁只比较结果、排序、统一身份、终态、语义事件与字段血缘的跨运行等价性，
不读取 gold/qrels、不计算质量指标，也不是官方 scorer。

```bash
PYTHONPATH=src python scripts/check_sharded_execution.py check-fixture
PYTHONPATH=src python scripts/check_sharded_execution.py check-fixture --fault duplicate_query
PYTHONPATH=src python scripts/check_sharded_execution.py audit-frozen
PYTHONPATH=src pytest -q -m sharded_execution_integrity_regression
```

完整契约、attempt/retry 选择规则和退出码见
[`docs/sharded-execution-integrity.md`](sharded-execution-integrity.md)。

## 运行资源账本与预算守恒

新格式 Benchmark 可显式开启 `--resource-ledger`，让默认关闭的
`resource_ledger_v1` 观察器复用生产 connector diagnostics、SearchBudgetRuntime、LLM
usage、取消/终态事件以及 checkpoint generation，生成逐 operation 权威账目。真实请求、
分页、重试、cache、预算拒绝、取消与部分完成分别记录；没有供应商证据的 token/cost 是
`not_available`，不能以 0 或估算价格替代。

`resource_accounting_integrity_v1` 验证 run=query=operation 汇总、预算预留/消费/释放/
剩余额守恒、调用唯一性、取消后零新增消费、resume 已提交 attempt 唯一计数及 shard 只汇总
最终 attempt。账本进入原子 completion generation，并可登记到 `run_manifest_v1` 和复现
胶囊；日志、兼容文件和旧汇总不具权威性。历史 Record160/Full1000 缺少逐操作账本，
只读资格为 `not_eligible`，不会被反推或补造。

`provider_ingest_provenance_v1` 在资源账本之上增加 parser 前原始响应取证：每个真实 HTTP
attempt 对应一个脱敏 envelope，精确响应字节写入内容寻址的只读 tar；离线 replay 调用四源
生产 record parser，并要求 `parsed = accepted + rejected`、accepted 顺序/来源身份摘要一致、
分页无断链/循环/重复消费。该门禁只验证来源 ingestion 可追溯性，不计算 Precision、Recall、
显著性或官方成绩。Record160/162 缺少 parser 前字节，必须保持 `not_eligible`。详情见
[`docs/provider-ingest-provenance.md`](provider-ingest-provenance.md)。

`full1000_launch_control_v1` 只验证未来正式执行的启动授权和操作完整性，不计算任何质量
指标。它要求全 1000 从头运行、`current_rules` 唯一默认、tie-break v2 关闭，并将计划、
取证协议、资源账本、原子提交和 20-shard attempt 规则绑定到两阶段自哈希凭证。只有所有
shard 的唯一最终 attempt 完成后才可 aggregate；旧 Record162、fake dry-run、未授权 resume
或直接 runner 输出均不能解除 Full1000 未完成阻断。详情见
[`docs/full1000-launch-control.md`](full1000-launch-control.md)。

`formal_run_disaster_recovery_v1` 只验证完整提交代能被增量备份、异地恢复并从最后权威
query cursor 继续，且恢复后的资源账本、操作事件、aggregate 与不中断 fake 运行一致。
它不读取 gold/qrels，不发起网络/LLM/Snapshot 写入，也不把稳定性证据解释为效果证据。
Record160/162 与真实 Full1000 当前均没有符合该新契约的正式备份，因此真实 readiness
保持 `external_run_not_started`。详情见
[`docs/formal-run-disaster-recovery.md`](formal-run-disaster-recovery.md)。

```bash
PYTHONPATH=src python scripts/check_resource_accounting.py check-fixture
PYTHONPATH=src python scripts/check_resource_accounting.py check-fixture --resume-shard
PYTHONPATH=src python scripts/check_resource_accounting.py audit-frozen
PYTHONPATH=src pytest -q -m resource_accounting_integrity_regression
```

该门禁只证明资源记录与既有预算一致，不优化预算、不计算质量指标，也不替代运行产物完整性、
实验成本分析或官方 scorer。完整协议和退出码见
[`docs/resource-accounting-integrity.md`](resource-accounting-integrity.md)。

## 非可信学术元数据隔离

`untrusted_metadata_isolation_v1` 使用本地四源形态恶意 fixture 和 fake LLM，检查论文标题、
摘要、作者、venue、URL 及来源错误始终留在数据角色中。门禁覆盖伪 system/tool 消息、
XML/Markdown/JSON 边界、控制与双向字符、超长输入、危险 URL、敏感资源请求、Schema
走私及跨 query 污染；业务阶段真实网络、LLM、工具、子进程、Snapshot 写入和质量指标均
为 0。权威论文原值与身份不改写，只有 LLM 传输、活动链接、诊断和导出使用有血缘的派生
表示。

```bash
PYTHONPATH=src python scripts/check_untrusted_metadata_isolation.py check-fixture
PYTHONPATH=src python scripts/check_untrusted_metadata_isolation.py \
  check-fixture --fault role_escape
PYTHONPATH=src python scripts/check_untrusted_metadata_isolation.py \
  check-fixture --fault cross_query_pollution
PYTHONPATH=src python scripts/check_untrusted_metadata_isolation.py audit-frozen
PYTHONPATH=src pytest -q -m untrusted_metadata_isolation_regression
```

退出码 `0/2/3/4` 分别表示通过、隔离违规、`not_eligible` 和用法错误。旧
Record160/Full1000 缺少原始来源字段、消息边界及隔离观察记录，只读资格为
`not_eligible`；该状态不能用于反推历史抗注入能力。完整字段规则和安全边界见
[`docs/untrusted-metadata-isolation.md`](untrusted-metadata-isolation.md)。

## Record160 四源融合覆盖消融

`source_fusion_ablation_v1` 是只读、无 gold 的覆盖与排序稳定性审计。协议在分析前固定
Record160 的 160 条主分析 query、既有 2 条无成功来源排除、四源顺序、Top-20、统一
身份规则、有限外推 RBO（`p=0.9`）、20,000 次 connected-component bootstrap/sign-flip
和按指标对四个 LOO 比较执行的 Holm-Bonferroni 校正。门禁只读取冻结 query plan、
source-level Snapshot 响应、生产阶段快照和既有 opaque component assignment；不会加载
数据集 gold/qrels，也不会根据 source failure 或消融结果重新筛样。

每条主分析 query 必须通过生产 current_rules 的去重、候选预算、约束判定、排序和结果
过滤，并逐项精确匹配冻结的 `initial_deduplicated`、`initial_judged`、
`initial_reranked`、`final_returned`。之后才计算 full、四个 leave-one-source-out 和四个
single-source 的 Top-20 填充、统一身份、来源独占/重叠、Jaccard、RBO 和共同身份排名
位移。若源级 Snapshot 缺失或完整四源无法精确重建，退出码 3 (`not_eligible`)；重建或
统计契约违规为 2，用法错误为 4。

```bash
PYTHONPATH=src python scripts/check_source_fusion_ablation.py run
PYTHONPATH=src python scripts/check_source_fusion_ablation.py verify
```

该审计的“覆盖”“独占”只描述冻结候选身份和排名稳健性，不代表论文相关性、人工
Precision、Recall/F1 或官方成绩，也不会改变任何来源或实验策略的默认开关。

## Record160 来源可靠性漏斗

`source_reliability_diagnostics_v1` 在四源融合审计的同一冻结 Record162 上复用 160 条
主分析 query、2 条既有排除、1,093 个检索 Snapshot key，以及生产 adapter 后论文模型、
统一身份去重、候选预算、current_rules 约束、排序和 Top-20 选择路径。协议预先固定请求
终态、失败分类、漏斗阶段、unknown 处理、查询语法分层和 connected-component bootstrap；
来源失败、空响应或低产出不会触发样本删除。

漏斗分别记录 logical request、权威 Snapshot、parsed Paper、canonical identity、源内唯一
身份、跨源去重、全局预算、约束放行和 Top-20 贡献。旧 Snapshot 没有持久化 provider
原始响应或 pre-parser record count，因此 raw→parsed 损失必须为 `unknown`；不得从已解析
Paper 数反推。重复 Snapshot 引用只计一次响应，unknown Schema、状态/记录矛盾、失败伪装
为空成功及数量不守恒均由门禁拒绝。

```bash
PYTHONPATH=src python scripts/check_source_reliability.py run
PYTHONPATH=src python scripts/check_source_reliability.py verify
```

退出码 `0/2/3/4` 分别表示完成、重建或守恒违规、`not_eligible`、用法错误。输出只用于
诊断来源可用性、结构化产出和流水线静默损失；产出量、约束放行和 Top-20 出现均不证明
相关性、Precision、Recall/F1 或官方成绩，也不会自动改变来源配置或 evidence 结论。

## Record160 显式约束决策审计

`constraint_decision_audit_v1` 在读取冻结候选前固定八个 `QueryConstraint` 字段、生产谓词
版本、字段来源、缺失值语义、组合顺序、单约束留出和查询结构分层。实现直接调用
`JudgementAgent` 的生产 term、paper-type、venue 与 time 谓词，并沿用统一身份去重、
`current_rules` Judgement、reranker 和 `highly_and_partial` selector；不存在第二套过滤器。
每个候选的 expected/matched 参数仅保存计数和哈希，论文身份、查询身份及 component 均为
opaque 摘要，原 query、标题、摘要、标识和错误正文不进入审计产物。

决策状态固定为 `passed/failed/not_applicable/unknown`。null、空字符串和字段存在分别记录；
缺少 title+abstract、venue 或 year 时为 `unknown`，同时保留生产布尔结果，避免把无法验证
静默写成明确不匹配。`QueryConstraint.domains` 只参与 planning，候选 Judgement 的 domain
信号来自 `QueryAnalysis.domain`，因此该字段明确标为 candidate predicate 不适用。审计要求
全部候选记录所有失败集合，并从记录独立重建 constraint survivor 与 Top-20 身份；约束值及
`explicit_fields` 顺序置换必须保持 score、category、排名和结果不变。

```bash
PYTHONPATH=src python scripts/check_constraint_decisions.py run \
  --protocol benchmark/constraint_decision_audit_v1_protocol.json \
  --output /tmp/constraint-decision-audit
PYTHONPATH=src python scripts/check_constraint_decisions.py verify \
  --output /tmp/constraint-decision-audit
```

退出码 `0/2/3/4` 分别表示完成、重建或字段语义违规、`not_eligible`、用法错误。每次单字段
留出都重新调用同一生产判断/排序链，只描述规则选择性、正负证据耦合与 Top-20 填充变化；
它不是可部署替代方案，不证明相关性、Precision、Recall 或官方成绩，也不会自动放宽约束。

## Record160 Judgement 与排序决策审计

`ranking_decision_audit_v1` 在读取冻结结果统计前固定 `current_rules` 的 16 个 Judgement
分数组件、`0.72/0.45/0.25` 类别阈值及 `>=` 边界、rerank 四项分量和完整排序键：类别层级、
综合分、Judgement 分、引用、年份、标题 `casefold`、原输入序号。审计不另算分数，而是读取
生产 `JudgementFeatureVector`、生产 `RerankScoreBreakdown` 和生产 `_sort_key` 的观察性 trace；
Top-20 语义固定为“完整 rerank 后截取前 20，再保留 high/partial”。

冻结 Record162 中 160 条主分析查询、2 条既有无成功来源排除、1,093 个 Snapshot key 和
4,599 个预算内唯一候选均闭合。每条候选记录 opaque query/paper identity、字段合并血缘摘要、
组件与总分、相邻阈值 margin、rerank 前后位置、脱敏可重建排序键、Top-20 cutline 与终态；
不保存 query、标题、摘要、标识符或 gold。组件求和、类别、排序和最终返回均须逐候选重建
冻结阶段，NaN/Infinity、未登记组件、缺失排序键、重复身份及截断异常均使门禁失败。

```bash
PYTHONPATH=src python scripts/check_ranking_decisions.py run \
  --protocol benchmark/ranking_decision_audit_v1_protocol.json \
  --output /tmp/ranking-decision-audit
PYTHONPATH=src python scripts/check_ranking_decisions.py verify \
  --output /tmp/ranking-decision-audit
```

退出码 `0/2/3/4` 分别表示完成、重建或排序语义违规、`not_eligible`、用法错误。输入置换
边界测试允许且显式报告生产最后一级 `original_index` 对完全同键候选的影响；这属于可审计
tie-break 风险，不会被静默排序掩盖。报告仅解释既有类别门、rerank 和 Top-20 决策，不证明
相关性、Precision、Recall/F1 或官方成绩，也不得用于后验调整阈值、权重或排序策略。

## Record160 确定性 tie-break v2 资格审计

`deterministic_tiebreak_qualification_v1` 在读取 tie 统计前冻结 exact-tie 定义、稳定键优先级、
六种候选置换和资格门槛。exact tie 仅指生产排序键去掉最后一级 `original_index` 后六个值逐值
完全相等；不使用 epsilon、四舍五入桶或近似标题扩大集合。可选
`deterministic_tiebreak_v2` 优先使用统一身份的稳定标识，其次使用完整的精确
title+author+year 身份；身份不足时必须使用字段血缘登记的来源记录引用、来源名和规范化字段
摘要，不能使用输入位置、完成顺序、进程状态、query/case 标识或任何效果信息。

门禁从 1,093 个只读 Snapshot key 经生产 canonicalization、身份去重、Judgement、reranker 和
selector 精确重建 160 条主分析查询及 4,599 个候选，再对不可变去重候选池执行原序、反序、
来源块反转/轮转、固定随机及 shard `2→0→1` 完成顺序。每个置换必须得到字节一致的 v2
逐查询语义投影；score、category、身份集合、字段血缘、阶段事件和非 tie 相对顺序必须不变。
任一 Top-20 成员、最终返回成员、非 tie 顺序或权威身份变化均为 `not_qualified`。只有 exact
tie 内部顺序变化且所有语义不变量成立时才是 `qualified_for_review`，仍不得自动启用。

```bash
PYTHONPATH=src python scripts/check_tiebreak_qualification.py run \
  --protocol benchmark/deterministic_tiebreak_qualification_v1_protocol.json \
  --output /tmp/tiebreak-qualification
PYTHONPATH=src python scripts/check_tiebreak_qualification.py verify \
  --output /tmp/tiebreak-qualification
```

退出码 `0/2/3/4` 分别表示 `qualified_for_review`、语义或稳定性违规、`not_qualified`/
`not_eligible`、用法错误。该资格只证明稳定决胜规则的可复现性，不证明排序相关性、
Precision、Recall/F1 或官方成绩，也不修改 current_rules 默认行为或 evidence 结论。

## Record160 Top-20 交付保真

`top20_delivery_fidelity_v1` 在读取统计前冻结 `final_returned` 权威阶段、最多 20 条语义、
统一身份、必要/可空字段、UTF-8 编码、分页和出口版本。它从 1,093 个只读 Snapshot key
经生产 canonicalization、去重、`current_rules` Judgement、reranker 和正式 selector 精确
重建 160 条主分析 query 的 1,396 条最终结果；2 条既有无成功来源排除保持不变。

门禁逐项验证公共 API、JSON、JSONL 和前端展示的数量、身份、顺序、字段及分页
round-trip。公共结果新增 `result_identity` 作为稳定前端 key，并以 `authority_digest` 绑定
展示安全变换前的完整权威结果；未知年份保持 `null`。产品默认 tie-break 仍为
`original_index_v1`，`deterministic_tiebreak_v2` 不进入这次权威重建。仓库没有生产 CSV/
表格结果出口，因此固定报告 `unsupported_export`；Record160 旧运行缺少新式 run manifest
和提交代，复现胶囊固定报告 `not_eligible`，两者均不得伪造为通过。

```bash
PYTHONPATH=src python scripts/check_top20_delivery_fidelity.py run \
  --contract benchmark/top20_delivery_contract_v1.json \
  --output /tmp/top20-delivery-fidelity
PYTHONPATH=src python scripts/check_top20_delivery_fidelity.py verify \
  --output /tmp/top20-delivery-fidelity
```

退出码 `0/2/3/4` 分别表示全出口通过、交付或 round-trip 违规、`not_eligible`/
`unsupported_export`、用法错误。该门禁不生成比赛提交文件，不猜测官方 schema，也不证明
相关性、Precision、Recall/F1 或官方成绩。完整只读清单见
[`docs/top20-delivery-fidelity.md`](top20-delivery-fidelity.md)。

## 限制

`public_contract_compatibility_v1` 只验证评测相关 API、CLI 和机器 JSON 在版本演进中的读取与
交付兼容性。它不运行 evaluator、不读取 gold/qrels，也不定义或推断官方 scorer Schema；
兼容性通过不能解释为 Precision、Recall/F1 或官方成绩。协议和命令见
[`docs/public-contract-compatibility.md`](public-contract-compatibility.md)。

sample fixture 使用本地假检索器，只验证评测流程、分组开关和输出可复现性，不代表真实 benchmark 性能。

`formal_multivolume_storage_v1` 只验证 Full1000 shard 的文件系统拓扑、逐卷容量与
aggregate 引用完整性。其 1000-query fake 演练仅证明存储控制语义，不产生
Precision、Recall、显著性或官方成绩，也不能外推正式运行结果。

`formal_shard_streaming_retention_v1` 只验证 shard 完成后的归档、受控本地释放、
恢复与混合 aggregate 语义。预注册窗口 1/2/4 的合成 1000-query 演练要求结果与
全量常驻运行等价、外部调用零重复且资源账本守恒。容量结果仅说明当前主盘可容纳
有限活动窗口；备份容量和独立故障域未验证时仍返回 `not_ready`。该报告不计算或
暗示任何检索质量指标。

`portable_execution_site_attestation_v1` 只验证候选执行节点是否满足既有多卷容量、
故障域和原子文件系统资格，并验证证明能否安全导入启动授权链。无仓库双环境验证与合成
节点矩阵不启动检索、不读取 gold/qrels、不计算质量指标，也不证明主机身份或 Full1000
结果。

`formal_network_request_manifest_v1` 只验证 Full1000 计划中的 query/subquery/source
请求身份、分页规则、缓存键和理论请求上限是否确定性闭合，并只读量化历史 Snapshot
覆盖缺口。它不发送请求、不读取 gold/qrels、不推断来源产出或未完成 query 表现，也不
产生 Precision、Recall、显著性或官方成绩。

`formal_provider_pacing_v1` 只验证冻结的 9,640 个来源 intent 在外部容量声明下是否满足
token bucket、分钟窗口、并发、退避、暂停恢复和 19,280 attempt 预算守恒。合成 profile
中的逻辑等待步与并发峰值不是来源性能基准；真实容量未知时必须阻断启动，不能从历史错误、
现有 connector 延迟或结果产出推测限额。协议和命令见
[`docs/formal-provider-pacing.md`](formal-provider-pacing.md)。

`provider_capacity_declaration_intake_v1` 只验证容量声明的离线接收、结构约束、challenge
防重放和导入后的请求集合守恒。四源合成声明矩阵覆盖缺失、过期、单位/窗口矛盾、scope
漂移、篡改、动态降额和撤销；合成值不代表供应商真实配额，也不用于真实性能估计。
当前真实四源声明仍为 `not_available`，因此 readiness 返回 exit 3。该门禁不改变检索、
排序、预算或指标，也不生成质量结论。详见
[`docs/provider-capacity-declaration-intake.md`](provider-capacity-declaration-intake.md)。

`formal_backup_target_attestation_v1` 只验证 Full1000 流式保留所需的独立备份目标：
容量、inode、quota、原子写入、恢复能力和故障域独立性。合成目标矩阵不产生查询、
检索或质量指标；当前缺少真实备份目标时返回
`not_ready_no_qualified_backup_target`，并继续阻断流式释放、灾备复审和正式 launch。

`official_scorer_package_intake_v1` 只验证未来官方 scorer 包、输入/输出 Schema、指标
namespace 与来源证据能否安全离线接收，并复用现有 scorer sandbox 做合成输入的无成绩
conformance。合成矩阵不读取 Record160/Full1000 结果，不生成或展示质量值；哈希也不证明
赛事方身份。当前 package、Schema、namespace 和真实性材料缺失，因此 readiness 返回
exit 3，官方 scorer/schema 阻断保持不变。

当前真实运行还包括一次 offset 20 的 30 条固定保留集复核，但只使用 arXiv，且候选阶段只召回 1/65 gold。所有子集都不能代表完整 Benchmark、比赛成绩或多源长期性能；完整 1000 条基线、重复运行和稳定的多领域统计尚未完成。

`human_annotation_submission_intake_v1` 只验证现有 471 项 A/B 锁定导出的完整性、
assignment/principal/包/commit 绑定以及分歧队列的完整盲化。任一侧缺失时不进行
解盲、比较、共同项筛选或统计；两侧闭合后生成的队列保留原始 A/B 判断，并包含全部且
仅包含分歧。本轮合成矩阵不生成真实标签、Precision、agreement、κ、裁决结果或官方
成绩。当前真实提交缺失，readiness 返回 exit 3。
`human_adjudication_activation_v1` 只接收上述门禁产生的 all-and-only-disagreements
队列。裁决者看到 query/title/abstract/year、A/B 原始标签、冻结 rubric 和专属
dispute alias；一致项、策略、排名、来源、gold/qrels 与全局身份始终不可见。统计
解锁复用 `human_precision_adjudication_v1` 的 change-only、cluster-aware 定义，
标记为 `human_internal_non_official`，且不支持从 change-only 包计算绝对
Precision@20。本轮合成矩阵不生成真实标签、κ、Precision 或正式成绩。

### 赛题三 200 条资格门禁

P0 精确 ID 语料和 Faiss 索引变化后，固定 AutoScholarQuery test 前 200 条及其顺序，比较 `contest_qual200_bm25_v1`、`contest_qual200_dense_v1` 和 `contest_qual200_reranker_v1`。候选必须在 F1@20 或 Recall@20 上严格提升，且配对 query-level bootstrap 95% 区间下界大于 0，baseline/candidate 资源账本均通过，才有资格运行完整 1000 条。

正式四组为规则基线、语义召回、语义召回+Qwen3 Reranker、语义召回+Qwen3 Reranker+`llm_semantic`。LLM 组每条查询最多一次调用、最多两条补充查询、temperature=0、严格 JSON Schema、始终保留原始查询，并记录模型、Prompt 版本、Token、调用次数、延迟、失败和 `current_rules` 回退。Provider 不可用时只报告未完成，不伪造指标。
