# Repository Guidelines（中文版）

## 项目简介
- 本仓库是一个全栈研究型 Agent 示例：前端使用 React + Vite，后端使用 FastAPI + LangGraph。
- 当前后端以 Gemini 为模型与搜索能力来源，执行“生成查询词 -> 网页检索 -> 反思补充 -> 最终答案”的循环。
- 主要入口：`frontend/src/App.tsx`（页面交互）与 `backend/src/agent/graph.py`（图流程）。

## 项目结构与模块组织
- `frontend/`：前端界面与组件。`src/components/ui` 放通用 UI，`src/lib` 放工具函数。
- `backend/`：Agent 核心逻辑。`src/agent/` 包含 `graph.py`、`prompts.py`、`state.py`、`app.py`。
- `backend/langgraph.json`：声明图入口与 HTTP 应用入口。
- 根目录：`Makefile`、`Dockerfile`、`docker-compose.yml` 负责本地和部署编排。

## 开发、构建与检查命令
- `make dev`：同时启动前后端开发服务。
- `make dev-frontend` / `make dev-backend`：分别启动前端或后端。
- `cd frontend && npm run dev|build|lint`：本地开发、构建、前端静态检查。
- `cd backend && langgraph dev`：启动后端 LangGraph 服务。
- `cd backend && make lint|format|test`：执行 Ruff/mypy、格式化、pytest。

## 重写路线（先 OpenAI SDK，再 LangGraph）
- 阶段 1：先改成 OpenAI SDK 形态。
- 目标：替换 Gemini 相关调用为 OpenAI Python SDK（`OPENAI_API_KEY`），保留现有 API/前端交互不变。
- 建议：新增 `backend/src/agent/llm_client.py` 统一封装模型调用；把 `graph.py` 中模型与搜索调用集中迁移到该封装层。
- 验收：`backend/examples/cli_research.py` 和前端对话都能正常返回答案与引用。
- 阶段 2：再整理为更规范的 LangGraph 结构。
- 目标：将节点输入输出严格类型化，固定 `generate_query -> web_research -> reflection -> finalize_answer` 的状态流。
- 建议：将提示词、模型配置、重试策略拆分到独立模块，并补充最小可运行测试。

## 代码风格与提交规范
- Python 使用 4 空格缩进与 `snake_case`；TypeScript 组件文件使用 `PascalCase`；前端路径别名使用 `@/`。
- 提交信息建议使用简短祈使句或 Conventional Commits（如 `feat:`、`fix:`）。
- PR 需包含改动说明、验证命令结果，UI 改动附截图。

## 安全与配置
- 从 `backend/.env.example` 复制 `.env`，配置密钥后再运行。
- 不要提交任何 API Key 或生产凭据；Docker 运行通过环境变量注入密钥。
