import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_gemini_key_from_env_file_is_picked_up_regardless_of_cwd(tmp_path, monkeypatch):
    """Regression test for the real bug: a user placed GEMINI_API_KEY in a
    .env file and still got the extractive fallback, because config.py only
    ever read os.environ directly. Verifies the fix by writing a temporary
    .env at the project root and confirming build_wiring() resolves
    GeminiLLMProvider as configured, from a *different* working directory
    (simulating running the CLI from inside interface/, as in the bug
    report) — then cleans the .env file back up either way."""
    env_path = PROJECT_ROOT / ".env"
    backed_up = env_path.exists()
    original = env_path.read_bytes() if backed_up else None
    try:
        env_path.write_text("GEMINI_API_KEY=test-key-not-real\n", encoding="utf-8")
        # Simulate invocation from a different cwd, and a fresh process's
        # environment (no GEMINI_API_KEY pre-set), by shelling out.
        env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
        result = subprocess.run(
            [sys.executable, "-c",
             "from infrastructure.config import build_wiring; "
             "w = build_wiring('/tmp/wiretest-envcheck'); print(w.llm.name)"],
            cwd=str(tmp_path),  # deliberately NOT the project root
            env={**env, "PYTHONPATH": str(PROJECT_ROOT)},
            capture_output=True, text=True, timeout=30,
        )
        assert "gemini" in result.stdout.lower(), result.stdout + result.stderr
    finally:
        if backed_up:
            env_path.write_bytes(original)
        else:
            env_path.unlink(missing_ok=True)
