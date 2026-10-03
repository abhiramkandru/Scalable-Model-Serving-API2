FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional extras, enabled with --build-arg:
#   docker build --build-arg EXTRAS="nvidia-ml-py" ...        real GPU telemetry (run on a GPU node)
ARG EXTRAS=""
RUN if [ -n "$EXTRAS" ]; then pip install --no-cache-dir $EXTRAS; fi
COPY requirements-otel.txt .
ARG WITH_OTEL=false
RUN if [ "$WITH_OTEL" = "true" ]; then pip install --no-cache-dir -r requirements-otel.txt; fi

COPY app ./app
RUN useradd --system --uid 10001 app
USER 10001

# Vertex AI injects AIP_HTTP_PORT / AIP_PREDICT_ROUTE / AIP_HEALTH_ROUTE; the app honours the routes, the port is read here.
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-8000} --no-access-log"]
