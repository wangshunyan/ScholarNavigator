# 阶段一：候选召回与 Judgement 离线瓶颈分析

更新时间：2026-08-23。

本页只记录当前 checkout 可以读取的证据。旧版本曾引用服务器上的
`contest_full_rules_v1`、Dense、Reranker 和 Soft-Judgement RunId，但这些目录、
配置、代码指纹、输入/索引哈希和资源账本不在当前本地证据链中；旧表格和数字已删除，
不能作为参赛成绩或当前实现结论。服务器结果必须先按
[`server-evidence-sync.md`](server-evidence-sync.md) 导出脱敏 bundle，再重新审计。

## 当前可验证结论

- 本地 BM25 标题语料约 569,432 条，arXiv ID 唯一；摘要、作者、年份、venue、DOI
  完整度当前为 0。
- 本地 semantic legacy 语料约 31,136 条，title/abstract 完整，但排序元数据缺失，
  不能作为正式 P0/Faiss 语料。
- 当前代码提供 rules、local BM25、local hybrid/Faiss、RRF、Reranker 和 LLM 开关，
  但 Dense/Reranker/RRF 质量收益尚没有当前 checkout 可核验的成对资格运行。
- `current_rules`、Query Evolution、RefChain、RRF、质量过滤和 LLM 均应保持默认保守；
  只有同一查询顺序、输入、候选预算、模型和资源约束下的成对实验出现正向区间，才允许
  改变默认路径。

## 正式实验门槛

1. 先取得合法、带稳定 arXiv ID、摘要、作者、年份、venue 和 DOI 的元数据输入，重建
   语料与索引并保存输入/索引 SHA-256、字段完整度和资源账本。
2. 在前 200 条固定 query 上配对 rules/BM25、Dense、Reranker 和 RRF；gold/qrels 只
   在检索结束后的 evaluator 使用。
3. 报告 candidate Recall、F1@5/10/20、Recall@20、MRR、延迟、调用数、失败率、
   fallback 和 paired-bootstrap 区间。没有正向收益的策略保持关闭。
4. 只有 200 条资格门禁、资源账本和完成标记全部通过，才允许运行完整 1000 条；内部
   F1/Recall 必须明确标记为工程指标，不等同官方 scorer。

## 当前阻塞

- P1-01：本地没有满足正式元数据完整度的输入。
- P1-02：缺少当前代码指纹绑定的 Dense/Reranker/RRF 成对资格产物。
- GPU、LLM Provider、开放全文许可、官方 scorer 和服务器实验 bundle 是外部依赖，
  不通过读取 `.env`、SSH 凭据或旧聊天记录解除。
