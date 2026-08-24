# 赛题三 5 分钟演示脚本

本脚本用于录屏和现场演示，不替代 AutoScholarQuery 正式评测，也不把单次演示结果写成泛化指标。演示前先确认浏览器、终端和日志中没有 API Key、`.env` 内容、服务器地址或本地绝对路径。

## 0:00–0:30 开场与问题定义

说明系统面向“科研场景下复杂学术查询”，输入不仅是主题，还可以包含时间、方法、数据集、会议、排除条件和证据要求。强调默认路径不依赖 LLM，Provider 不可用时仍保留首轮检索和降级结果。

## 0:30–1:20 输入复杂查询

使用页面“现场演示查询”下拉框或演示查询集中的一条，例如：

> 找 2021 年以后关于扩散模型用于医学图像分割的论文，要求结果里说明数据集和评价指标。

下拉框只会把预置复杂查询填入输入框，不会自动提交，也不会改变检索源；录屏前仍应确认查询内容后手动点击发送。可交互录屏使用 `demo-queries.md`；批量复现使用其中前 5 条的无 gold JSONL 版本 `demo-queries.jsonl`。

展示查询理解区域中的时间、主题、任务和证据约束；不要展示或暗示隐藏的 gold/qrels。

## 1:20–2:20 展示检索过程

提交查询后依次展示“理解查询、检索候选、相关性判断、重排序、综合”。指出每个阶段的候选数、延迟、失败来源和降级原因来自运行时诊断，而不是手工填写。若外部来源不可用，展示错误被隔离且结果仍返回。

## 2:20–3:30 展示排序和证据

打开前 3 条论文，说明：

- 论文排名、相关性分数、类别和排序理由；
- 普通证据的来源与置信度；
- 有许可全文时展开“全文证据与定位”，查看来源 URL、许可证、文档/段落 SHA-256、段落编号和字符范围；
- 没有许可全文时明确显示摘要结果仍保留，不把摘要冒充全文证据。

如需现场演示受限全文入口，只使用已经人工核验许可的公开来源，并显式提供 allow-list；没有许可确认时命令会失败关闭：

```bash
PYTHONPATH=src python scripts/fetch_open_full_text.py \
  --url https://<已核验公开主机>/<paper> \
  --license-id CC-BY-4.0 --license-verified \
  --allowed-host <已核验公开主机> \
  --output outputs/demo-full-text-evidence.json
```

该命令只获取指定 URL，不做来源发现或任意重定向；输出中的内容哈希和段落定位可直接在论文卡片/导出中展示。没有可核验来源时展示 `license_unverified` 降级结果，不要使用未知版权 PDF。

## 3:30–4:10 展示引用关系与导出

打开引用图或方法聚类，随后分别点击 JSON 和 Markdown 导出。Markdown 导出应包含查询、成本、排名理由、普通证据以及全文证据定位，便于评委复核。

## 4:10–4:40 展示成本和安全边界

展示 API 调用次数、检索轮数、延迟、缓存命中和错误计数。说明质量信号和 LLM feedback 默认关闭或受门禁控制；未知风险不扣分，内部 F1/Recall 只作为工程指标，不是官方成绩。

## 4:40–5:00 复现入口

展示仓库 README 的安装与启动命令，以及离线 smoke 命令：

```bash
cp .env.example .env
```

Windows PowerShell 使用 `Copy-Item .env.example .env`。复制模板后，首次本地 BM25 建索引可能需要十几秒；模板保持 LLM disabled，不需要 API Key。

```bash
PYTHONPATH=src python scripts/check_clean_clone_smoke.py \
  --output outputs/clean_clone_smoke_local.json
```

验收 5 条演示查询可连续返回结果时，可运行：

```bash
PYTHONPATH=src python scripts/check_demo_reproducibility.py
```

该命令要求每条演示至少有一条可见结果，并检查零网络、零 LLM 与 `gold_or_qrels_loaded=false`；失败时应展示结构化原因，不要把单条失败包装成成功。

最后说明：公开 GitHub 包不含 `.env`、模型缓存、语义大语料和临时实验输出；正式 Dense/Reranker/RRF 成绩需使用绑定当前代码、输入哈希和资源账本的新运行重新核验。
