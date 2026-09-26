"""
Vercel serverless entrypoint.

On every cold start, this module:
  1. Adds the repository root to sys.path so all internal imports resolve.
  2. Copies any pre-seeded data from the read-only repo `data/` folder into
     the writable /tmp/data scratchpad, so SQLite and vector-store files
     can be opened and written without hitting Vercel's read-only filesystem.
  3. Sets the DATA_DIR environment variable so the rest of the application
     resolves all storage paths through /tmp/data automatically.
  4. Imports and re-exports the FastAPI `app` object for Vercel to serve.

No changes to domain, application, or infrastructure layers are needed — the
read/write indirection is fully contained here and in the DATA_DIR env var.
"""
import os
import sys
import shutil
from pathlib import Path

# 1. Add repository root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 2. Define source (repo read-only) and target (writable serverless scratchpad)
REPO_DATA_DIR = ROOT_DIR / "data"
WRITABLE_DATA_DIR = Path("/tmp/data")

# 3. On cold start, copy existing /data into /tmp/data so SQLite can read & write
if not WRITABLE_DATA_DIR.exists():
    if REPO_DATA_DIR.exists():
        # Copies all pre-seeded databases, chunks, and indexes from repo to /tmp/data
        shutil.copytree(str(REPO_DATA_DIR), str(WRITABLE_DATA_DIR))
    else:
        WRITABLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 4. Set environment variable so the app reads from /tmp/data
os.environ["DATA_DIR"] = str(WRITABLE_DATA_DIR)

# 5. Import and expose the FastAPI app
from interface.http_api import app  # noqa: E402  (must come after env setup)
