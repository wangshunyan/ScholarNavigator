# 本地语义数据审计（2026-08-23）

本次审计只读取本地文件，不连接服务器，也不读取 `.env`、SSH 密钥或其他凭据。结论用于决定哪些资产可以进入正式评测和 GitHub 发布。

| 文件 | SHA-256 | 结构/字段结论 | 处理决定 |
|---|---|---|---|
| `datasets/semantic/arxiv_data.csv.zip` | `8169c3ad2de7bf87a9811a63da981f20eb2876d79740cd1c7167b0f56cff640a` | ZIP 可读，内部 `arxiv_data.csv` 仅有 `titles,summaries,terms`；没有稳定 arXiv ID、作者、年份、venue、DOI。 | 仅作来源线索，不进入正式语料，不上传 GitHub。 |
| `datasets/semantic/arxiv_paper_abstracts.zip` | `d906a415c73d53fbd371dc58c0497766c2cb76046221741ad14cc3f08c384157` | ZIP 不完整，缺少中央目录；无法通过标准 ZIP 读取或完整校验。 | 视为损坏传输副本，不解压、不使用、不上传。 |
| `datasets/semantic/pasa_papers_with_abstracts.jsonl` | 已由 `scripts/audit_corpus_metadata.py` 审计 | 31,136 条、arXiv ID 唯一、title/abstract 完整；authors/year/venue/doi 完整度为 0。 | 保留为 legacy/title+abstract 诊断输入；不得作为正式资格语料。 |

## 正式数据准入条件

只有在取得来源和许可证可核验的 JSONL 元数据后，才可使用：

```text
scripts/audit_corpus_metadata.py <metadata.jsonl> \
  --require-fields title,abstract,authors,year,venue,doi
scripts/merge_paper_metadata.py \
  --base datasets/local_bm25/pasa_papers.jsonl \
  --metadata <metadata.jsonl> \
  --output outputs/pasa_papers_enriched.jsonl \
  --report outputs/pasa_papers_enriched.report.json
```

合并前必须记录来源、许可证、下载日期和 SHA-256；合并器只按稳定 arXiv ID 填充缺失字段，不推断或覆盖字段。合并后先重新审计，再启动 200 条资格门禁；在门禁通过前不得启动 1000 条正式评测。

## 发布边界

上述压缩包、语义模型、索引和实验输出均属于本地/服务器运行资产，不随 source-only 发布包进入 GitHub。队友从 GitHub clone 后可按 README 使用公开标题型 BM25 smoke；完整语料和模型需单独提供可审计下载来源。
