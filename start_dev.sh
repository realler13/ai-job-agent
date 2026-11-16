#!/bin/bash
# Unified startup script for AI Job Agent dev environment.
# Starts FastAPI backend + Next.js frontend.
# Usage: ./start_dev.sh [--prod]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

# --- Parse flags ---
MODE="dev"
NEXT_PORT=3000
if [[ "${1:-}" == "--prod" ]]; then
    MODE="prod"
    NEXT_PORT=8503
fi

FASTAPI_PORT=8502
FASTAPI_LOG="$LOGS/fastapi_dev.log"
NEXTJS_LOG="$LOGS/nextjs_dev.log"

# PIDs to track for cleanup
FASTAPI_PID=""
NEXTJS_PID=""

cleanup() {
    echo ""
    echo "Shutting down..."
    [[ -n "$FASTAPI_PID" ]] && kill "$FASTAPI_PID" 2>/dev/null && echo "  Stopped FastAPI (pid $FASTAPI_PID)"
    [[ -n "$NEXTJS_PID" ]]  && kill "$NEXTJS_PID"  2>/dev/null && echo "  Stopped Next.js (pid $NEXTJS_PID)"
    wait 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

port_in_use() {
    lsof -iTCP:"$1" -sTCP:LISTEN -t >/dev/null 2>&1
}

# --- FastAPI ---
if port_in_use "$FASTAPI_PORT"; then
    echo "[FastAPI]  Port $FASTAPI_PORT already in use — skipping."
else
    echo "[FastAPI]  Starting on port $FASTAPI_PORT (log: $FASTAPI_LOG)"
    # --reload-dir src: watch ONLY source. Watching the repo root would watch
    # the SQLite database (changes on every write) and leak file descriptors.
    # See run_api.sh.
    PYTHONPATH="$ROOT/src" "$ROOT/.venv/bin/uvicorn" job_agent.api:app \
        --host 0.0.0.0 --port "$FASTAPI_PORT" --reload --reload-dir src \
        >> "$FASTAPI_LOG" 2>&1 &
    FASTAPI_PID=$!
fi

# --- Next.js ---
if port_in_use "$NEXT_PORT"; then
    echo "[Next.js]  Port $NEXT_PORT already in use — skipping."
else
    if [[ "$MODE" == "prod" ]]; then
        echo "[Next.js]  Building and starting in prod mode on port $NEXT_PORT (log: $NEXTJS_LOG)"
        (cd "$ROOT/web" && npm run build >> "$NEXTJS_LOG" 2>&1 && npm run start >> "$NEXTJS_LOG" 2>&1) &
        NEXTJS_PID=$!
    else
        echo "[Next.js]  Starting dev server on port $NEXT_PORT (log: $NEXTJS_LOG)"
        (cd "$ROOT/web" && npm run dev >> "$NEXTJS_LOG" 2>&1) &
        NEXTJS_PID=$!
    fi
fi

# --- Summary ---
echo ""
echo "=== AI Job Agent dev stack ($MODE mode) ==="
echo "  FastAPI  : http://localhost:$FASTAPI_PORT  ${FASTAPI_PID:+(pid $FASTAPI_PID)}"
echo "  Next.js  : http://localhost:$NEXT_PORT  ${NEXTJS_PID:+(pid $NEXTJS_PID)}"
echo ""
echo "Logs:"
echo "  $FASTAPI_LOG"
echo "  $NEXTJS_LOG"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# Wait for background processes — if either exits, keep waiting for the other
wait
