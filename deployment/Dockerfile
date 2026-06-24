FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Run the API as a non-root user in production containers.
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup appuser

# requirements.txt composes runtime and torch dependency groups.
COPY requirements.txt .
COPY requirements-runtime.txt .
COPY requirements-torch.txt .

RUN python -m pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt

COPY app ./app
COPY main.py .
COPY scripts ./scripts
COPY artifacts ./artifacts

# /app/data stores the SQLite prediction log volume at runtime.
RUN mkdir -p /app/data /app/artifacts && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
