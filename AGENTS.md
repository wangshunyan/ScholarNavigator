# 开发代理规则

## 入口

- 后端主入口：`src/scholar_agent/app/main.py` 中的 `app`。
- 检索服务入口：`src/scholar_agent/services/search_service.py` 中的 `SearchService.run_search`。
- 前端主入口：`frontend/src/app/page.tsx`，主界面组件位于 `frontend/src/components/scholar-navigator-app.tsx`。

## 安全与修改边界

1. 不读取、输出、提交或泄露 `.env`；配置说明只引用 `.env.example`。
2. 不修改 `third_party/` 中的第三方源码，除非用户明确要求。
3. 删除文件前必须检查代码、测试、脚本和文档中的引用，并迁移仍然唯一有效的信息。
4. 不得在文档中声明未经测试或正式评测验证的能力。
5. 不得在生产代码中为单条 benchmark query 编写特例。

## 同步要求

1. 修改代码必须同步新增或更新测试。
2. 修改 API 时必须同步 Pydantic Schema、API mapper、前端类型和相关测试。
3. 修改架构后必须更新 `docs/architecture.md`。
4. 修改评测口径、指标或匹配规则后必须更新 `docs/evaluation.md`。
5. 修改运行时 Prompt 时必须同步 `src/scholar_agent/prompts/manifest.json` 版本号和 Prompt 测试。
6. 不得删除、跳过或弱化测试来使测试通过。

## 完成检查

任务完成后必须依次运行：

```bash
PYTHONPATH=src pytest -q
```

```bash
cd frontend
npm run lint
npm run build
```
