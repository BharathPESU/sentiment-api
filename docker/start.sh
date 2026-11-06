#!/bin/bash
set -e

echo "[STARTUP] Starting FastAPI backend on 127.0.0.1:8001..."
uvicorn src.app.main:app --host 127.0.0.1 --port 8001 &
FASTAPI_PID=$!

echo "[STARTUP] Starting Streamlit frontend on 127.0.0.1:8501..."
API_URL="http://127.0.0.1:8001" streamlit run src/app/streamlit_app.py \
  --server.port 8501 \
  --server.address 127.0.0.1 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.serverAddress 0.0.0.0 &
STREAMLIT_PID=$!

echo "[STARTUP] Starting Nginx gateway on 0.0.0.0:8000..."
nginx -c /app/docker/nginx.conf -g "daemon off;" &
NGINX_PID=$!

cleanup() {
  echo "[SHUTDOWN] Stopping services..."
  kill -TERM "$FASTAPI_PID" "$STREAMLIT_PID" "$NGINX_PID" 2>/dev/null || true
  wait "$FASTAPI_PID" "$STREAMLIT_PID" "$NGINX_PID" 2>/dev/null || true
}

trap cleanup SIGINT SIGTERM

# Wait for any process to exit
wait -n
EXIT_CODE=$?
cleanup
exit $EXIT_CODE

