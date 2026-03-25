import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = BACKEND_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Keep imports deterministic in tests that instantiate the client.
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
