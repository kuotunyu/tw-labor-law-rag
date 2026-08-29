"""Run the FastAPI app locally without Docker (puts src/ on sys.path first;
inside docker-compose, PYTHONPATH is set directly in the Dockerfile instead
and the app runs via plain `uvicorn rag.api.main:app`).
"""

import _bootstrap  # noqa: F401
import uvicorn

if __name__ == "__main__":
    uvicorn.run("rag.api.main:app", host="127.0.0.1", port=8000, reload=False)
