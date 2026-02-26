FROM python:3.11-slim

ARG APP_UID=10001
ARG APP_GID=10001

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libopenblas-dev sqlite3 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY README.md ./
COPY .env.example ./.env.example

RUN groupadd --gid "${APP_GID}" appuser \
    && useradd --uid "${APP_UID}" --gid appuser --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /data /tmp/recommenderr \
    && chown -R appuser:appuser /app /data /tmp/recommenderr

ENV PYTHONUNBUFFERED=1
VOLUME ["/data"]

USER appuser

CMD ["python", "-m", "app.cli", "serve"]
