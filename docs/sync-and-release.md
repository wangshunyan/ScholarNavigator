# 本地、服务器与 GitHub 同步规范

## 结论

- GitHub 只保存 Git 已跟踪的源代码、配置模板、测试、文档和公开语料。
- outputs/、模型缓存、Faiss/embedding 大文件、服务器实验目录、.env 和 SSH 私钥不上传 GitHub。
- 服务器实验结果通常位于 outputs/benchmark_runs/<run_id>/；只有脱敏 bundle 才复制回本地 outputs/ 做审计。
- 必须比较提交、配置、语料哈希、运行 ID 和产物清单，不能仅凭能 clone 判断数据一致。

## 服务器端（由实验操作者执行）

在服务器项目目录中确认状态，不要复制整个项目目录：

    cd <server-project-root>
    git rev-parse HEAD
    git status --short
    find outputs/benchmark_runs -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail

对单个运行目录执行脱敏打包：

    PYTHONPATH=src python scripts/package_server_evidence.py --run-dir outputs/benchmark_runs/<run_id> --output <temporary-evidence-path>/<run_id>-evidence.zip

脚本离线运行，不读取 .env 或 SSH 文件。将 zip 复制到本地后再审计；不要上传 zip 到 GitHub。

## 本地端

在提交或接收服务器证据前，可先运行只读同步审计：

    PYTHONPATH=src python scripts/check_sync_state.py

只有输出 `status: ready` 且 `github_in_sync: true` 时，才表示当前源代码已
与 GitHub 的 `origin/main` 对齐；这不代表服务器实验结果已经同步。若出现
`tracked_output_paths`，它们是历史上已公开的轻量报告，建议下一次整理为
`docs/` 下的公开摘要或从 Git 索引移除；原始实验输出不应继续提交。

将 bundle 放入 outputs/server_evidence/ 后，解压并核对其中的 manifest.json、文件哈希、提交和完成标记；如需重新生成或验证原始运行目录，使用同一个 package_server_evidence.py。bundle 本身不应提交 Git。

记录本地代码状态和 GitHub 是否已同步：

    git status --short
    git rev-parse HEAD
    git fetch origin
    git rev-parse origin/main
    git log --oneline --decorate -5

确认 diff、测试和发布包检查通过后，再提交并推送明确的源代码/文档/测试文件。

## 结果保存位置与一致性判定

## 合法元数据导入

获得合法的、以 `arxiv_id` 为稳定键的外部 JSONL 后，可离线生成新语料；脚本默认只填充缺失字段，不覆盖已有值：

    PYTHONPATH=src python scripts/merge_paper_metadata.py --base datasets/local_bm25/pasa_papers.jsonl --metadata <合法元数据.jsonl> --output outputs/pasa_papers_enriched.jsonl --report outputs/pasa_papers_enriched.report.json

先用 `scripts/audit_corpus_metadata.py` 审计报告，再把新语料用于 smoke/200 条成对评测。合并报告会记录 base、metadata 和 merged 输出的 SHA-256，三者必须随运行配置保存。不要把来源不明、标题猜测、gold/qrels 或服务器私有数据当作元数据输入。

每次 benchmark 应在 outputs/benchmark_runs/<run_id>/ 生成 config.json、metrics.json、results.jsonl、failures.jsonl 和资源账本。outputs/ 默认被忽略，不会随 GitHub 发布。

本地与服务器只有在以下字段全部相符时，才可称为“同一实验输入：

- code.commit 相同且 code.dirty 均为 false；
- query 集身份/顺序、profile、候选预算和运行时哈希相同；
- corpus、模型和索引 SHA-256 相同；
- 结果行数完整，failures.jsonl 为空或已解释；
- bundle 清单和哈希审计通过。

否则应称为“不同运行”或“服务器结果的本地审计副本”，不能直接覆盖本地结果，也不能把指标写成官方比赛成绩。

## 给队友的下载说明

队友从 GitHub clone 后可安装依赖并运行离线 BM25 smoke；语义模型、Faiss 索引、服务器实验结果和 Provider 凭证需要按项目文档单独准备。发布包检查应确认不含 .env、outputs/、服务器 IP、模型缓存或私钥。
