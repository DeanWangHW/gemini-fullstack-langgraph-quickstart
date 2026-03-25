"""Pytest 全局测试夹具与路径初始化配置。

Notes
-----
- 将 `backend/src` 动态加入 `sys.path`，确保测试环境导入路径与运行时一致；
- 为依赖 `LLMClient._get_client` 的测试提供稳定的默认 `OPENAI_API_KEY`，
  避免单测因本地环境变量缺失产生偶发失败。
"""

import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = BACKEND_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Keep imports deterministic in tests that instantiate the client.
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
