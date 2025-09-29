# Production serving image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy requirements first for caching
COPY pyproject.toml ./

# Install Python dependencies (runtime only)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy application code
COPY env ./env
COPY models ./models
COPY agents ./agents
COPY rlhf ./rlhf
COPY eval ./eval
COPY serve ./serve
COPY ops/configs ./ops/configs

# Create directories for models
RUN mkdir -p checkpoints models/risk_accept/artifacts && \
    chown -R appuser:appuser checkpoints models

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Run application
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8080"]

