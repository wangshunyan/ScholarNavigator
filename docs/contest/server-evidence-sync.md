# 服务器实验结果同步与 GitHub 发布

代码、轻量评测输入、测试和文档通过 Git 同步；大语料、索引、缓存、原始 `results.jsonl` 与完整实验目录不进入 GitHub。这样其他人 clone 后能获得可运行项目，而不会下载数 GB 数据、私有路径或配置。

## 推荐流程

1. 本地先运行测试，通过后提交并推送代码；服务器只拉取该指定 commit，不在服务器直接改代码。
2. 服务器运行完成一个明确的 RunId 后，在服务器项目根目录执行：

```bash
PYTHONPATH=src python scripts/package_server_evidence.py \
  --run-dir outputs/benchmark_runs/<run-id> \
  --output /tmp/<run-id>-evidence.zip
```

3. 将生成的 zip 复制到本地 `outputs/imported_server_evidence/`，保留为本地审计材料。工具会要求完整运行标记和核心产物，并会拒绝凭据字段；它会将绝对路径和服务器地址从可导出 JSON/JSONL/Markdown 中替换为占位符，同时记录源文件与导出文件 SHA-256。
4. GitHub 只提交可复现实验的代码、命令、脱敏 manifest 摘要和结论；不要提交 zip、`outputs/benchmark_runs/`、`datasets/semantic/`、`.env`、密钥、原始模型或索引。
5. 其他人 clone 后按 `README.md` 安装依赖，并使用公开/自行获得的数据重新构建索引与复现实验。内部 F1/Recall 只能写为工程指标，不能描述成赛事官方成绩。

## 同步前检查

- 服务器与本地代码 commit 一致；RunId 的 `config.json` 中记录该 commit、数据哈希和索引哈希。
- Run 具备 `config.json`、`metrics.json`、`results.jsonl`、`resource_ledger.json` 和 `.run_complete` 或 `.run_committed`。
- 本地导入后先审阅 bundle 的 `manifest.json`，确认不含路径、用户、主机、凭据或不应公开的原文内容。
- 要把结果用于论文/答辩时，保留原始私有证据包和 SHA-256；公开仓库只保留脱敏摘要。

该流程不连接服务器、不读取 SSH 凭据或 `.env`，也不会自动推送 GitHub。
