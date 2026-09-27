# The Regression & Telemetry Gate - Week 5 Project
#
# Simple, single-stage image appropriate for a learning project. See
# Week 1's Dockerfile for notes on what a hardened production image
# would add (non-root user, multi-stage build, etc).

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY telemetry/ ./telemetry/

EXPOSE 8000

# Note: OPENROUTER_API_KEY and REDIS_URL must be passed at runtime, e.g.:
#   docker run -p 8000:8000 --env-file .env regression-telemetry-gate
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
