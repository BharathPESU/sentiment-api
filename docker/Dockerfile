# docker/Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models/sentiment \
    LOG_LEVEL=INFO \
    API_URL=http://127.0.0.1:8001

WORKDIR /app

# Install Nginx and curl for reverse proxying and health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx curl && \
    rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image far smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY streamlit_app.py ./
COPY docker/ ./docker/
COPY models/sentiment/ ./models/sentiment/
RUN chmod -R a+r /app/models/sentiment && \
    chmod +x /app/docker/start.sh

# Run as a non-root user for safety
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app /var/log/nginx /var/lib/nginx
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD curl -sf http://localhost:8000/health || exit 1

CMD ["/app/docker/start.sh"]
