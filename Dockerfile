# ---------------------------------------------------------------
# Transit Tracker API — Dockerfile
# Target: python:3.11-slim (supports ARM64 / aarch64 for Oracle Cloud)
# ---------------------------------------------------------------
FROM python:3.11-slim

# Prevent .pyc files and enable unbuffered stdout (better Docker logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build deps needed for psycopg2-binary and asyncpg C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer cache optimisation:
# dependencies are only re-installed when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY train_eta_model.py .

# Create the models directory so the ETA service can write .joblib files
RUN mkdir -p models

# Expose the Uvicorn port
EXPOSE 8000

# Health-check so Docker knows when the API is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
