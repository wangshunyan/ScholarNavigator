# 来源容量声明离线接收

`provider_capacity_declaration_intake_v1` 为四个来源的容量所有者生成无需仓库、无需网络、
无需项目依赖的声明包。它只接收结构化容量事实，不读取 `.env`、endpoint、请求头、URL
参数、query 文本或凭据，也不认证声明者的现实身份。内容 SHA-256 只证明声明内容与导入
时一致。

## 声明契约

每个 kit 只绑定一个来源和一个 scope alias：

| source | scope alias |
| --- | --- |
| `openalex` | `openalex_polite_pool_optional` |
| `arxiv` | `public_anonymous` |
| `semantic_scholar` | `semantic_scholar_api_key_optional` |
| `pubmed` | `ncbi_api_key_optional` |

容量所有者只能填写声明版本、每秒/分钟请求数、burst、最大并发、冷却秒数、生效/失效
epoch、预定义 evidence type、生命周期和可选 supersession 哈希。单位、Retry-After
语义、Full1000 plan、9,640 个 request intent、pacing 协议、代码提交和一次性 challenge
由 kit 冻结。未知字段必须保留 `not_available`，不能以 0 或默认值代替。

## 离线 kit

```bash
PYTHONPATH=src python scripts/check_provider_capacity_intake.py build-kit \
  --source openalex \
  --challenge <64-hex-one-time-challenge> \
  --issued-epoch <integer-epoch> \
  --output <temporary-kit.zip>
```

归档固定成员顺序、时间、权限和 JSON 编码。`verify.py` 只依赖 Python 标准库，可在无仓库
环境中运行：

```bash
python -I -S verify.py verify \
  --contract declaration_contract.json \
  --declaration declaration.json \
  --current-epoch <integer-epoch>
```

kit 不进入正式 evidence registry；操作者应从可信渠道单独取得 verifier 副本。归档成员
只作为数据校验，不授予执行归档内任意代码的权限。

## 导入与激活边界

导入要求四源声明全部 active、未过期、scope/commit/plan/protocol 精确匹配，且各 challenge
尚未消费。append-only challenge ledger 拒绝重放、跨来源借用、跨提交复用、声明篡改和撤销
后继续使用。合格声明会交给现有 `formal_provider_pacing_v1` 重算发送时序；门禁重新核对
9,640 个 intent、19,280 attempt 上限及 20 shard，容量不得删除、合并、重写请求或改变
预算。

```bash
PYTHONPATH=src python scripts/check_provider_capacity_intake.py \
  audit-readiness
```

当前没有真实四源声明，因此命令稳定返回 exit 3
`not_ready_missing_real_declarations`。合成矩阵和 dry-run 只证明接收链、重放防护及请求
集合守恒，不会进入真实 readiness，不会启动 Full1000，也不会解除正式验证阻断。
