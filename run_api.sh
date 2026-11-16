#!/bin/bash
cd "$(dirname "$0")"
# Production runs WITHOUT --reload: the reloader would watch the working
# directory, including the SQLite database that changes on every write, and
# leak file descriptors.
# For local code editing use: DEV_RELOAD=1 ./run_api.sh  — reload is then
# scoped to src/ only, so it never watches the DB.
RELOAD_ARGS=()
if [ "${DEV_RELOAD:-0}" = "1" ]; then
    RELOAD_ARGS=(--reload --reload-dir src)
fi
PYTHONPATH=src .venv/bin/uvicorn job_agent.api:app \
    --host 0.0.0.0 --port 8502 "${RELOAD_ARGS[@]}"
