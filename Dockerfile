FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml /app/
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -e .

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Run as non-root user
RUN useradd -m -u 1000 appuser

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/pyproject.toml .

# Install dependencies from wheels
RUN pip install --no-cache /wheels/*

# Copy application code
COPY src /app/src
COPY models /app/models

# Install the application itself
RUN pip install -e .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "churn_predictor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
