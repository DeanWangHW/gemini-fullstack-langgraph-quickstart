# Repository Guidelines（中文版）

## 项目定位
- 本仓库是一个全栈研究型 Agent 示例：
  - 前端：React + Vite
  - 后端：FastAPI + LangGraph
- 当前后端已经迁移到 **OpenAI SDK**，核心流程固定为：
  `generate_query -> web_research -> reflection -> finalize_answer`
- 主要入口：
  - 前端交互入口：`frontend/src/App.tsx`
  - 图流程入口：`backend/src/agent/graph.py`

## 5 分钟快速上手

### 1. 环境准备
- Python：建议 `3.11+`
- Node.js：建议 `18+`
- 包管理：`npm`、`pip`（或你常用的虚拟环境工具）

### 2. 安装依赖
- 后端依赖：
  - `cd backend`
  - `pip install -e .`
- 前端依赖：
  - `cd frontend`
  - `npm install`

### 3. 配置密钥（必做）
- 在 `backend` 目录下创建 `.env`：
  - `cp .env.example .env`
- 至少配置：
  - `OPENAI_API_KEY=你的密钥`
- 可选配置（代理/网关场景）：
  - `OPENAI_BASE_URL=...`

### 4. 启动项目
- 一键启动前后端（推荐）：
  - 根目录执行：`make dev`
- 分别启动：
  - 前端：`make dev-frontend`
  - 后端：`make dev-backend`

### 5. 最小可运行验证
- CLI 单次回答：
  - `cd backend && python examples/cli_research.py "LangGraph 是什么？" --initial-queries 1 --max-loops 1 --reasoning-model gpt-4.1-mini`
- CLI 流式调试（看每个节点输出）：
  - `cd backend && python examples/cli_research_stream.py "LangGraph 是什么？" --initial-queries 1 --max-loops 1 --reasoning-model gpt-4.1-mini --pretty`
- 单元测试：
  - `cd backend && pytest -q tests/unit_tests`

## 目录速览（先看这些文件）
- `backend/src/agent/graph.py`：LangGraph 节点编排与状态流转。
- `backend/src/agent/state.py`：节点输入输出 TypedDict 与全局状态结构。
- `backend/src/agent/llm_client.py`：OpenAI 调用、DDG 检索、摘要与重试实现。
- `backend/src/agent/prompt_builder.py`：提示词渲染（按节点构造）。
- `backend/src/agent/configuration.py`：运行时配置合并（默认值 / config / 环境变量）。
- `backend/src/agent/search_retry.py`：搜索重试策略。
- `backend/examples/`：命令行脚本（普通/流式）。

## 开发、构建与检查命令
- `make dev`：同时启动前后端开发服务。
- `make dev-frontend` / `make dev-backend`：分别启动前端或后端。
- `cd frontend && npm run dev|build|lint`：前端开发、构建、静态检查。
- `cd backend && langgraph dev`：启动 LangGraph 服务。
- `cd backend && pytest -q tests/unit_tests`：执行后端单测。

## 常见问题（排查优先级）
1. 无法返回模型结果：先检查 `OPENAI_API_KEY` 是否已配置且可用。
2. 搜索为空或失败：确认网络可访问，且 `ddgs` 依赖已安装。
3. 导入失败：确认在 `backend` 目录执行命令，且已安装后端依赖。
4. 前端页面空白：先执行 `cd frontend && npm install && npm run dev`。

## 代码风格与提交规范
- Python 使用 4 空格缩进与 `snake_case`。
- TypeScript 组件文件使用 `PascalCase`，前端路径别名使用 `@/`。
- 提交信息建议使用简短祈使句或 Conventional Commits（如 `feat:`、`fix:`）。
- PR 需包含改动说明、验证命令结果；UI 变更建议附截图。

## 安全与配置
- 不要提交任何 API Key 或生产凭据。
- Docker / CI 环境请通过环境变量注入密钥。
