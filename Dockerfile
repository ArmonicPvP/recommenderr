FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libopenblas-dev sqlite3 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY README.md ./
COPY .env.example ./.env.example

ENV PYTHONUNBUFFERED=1
VOLUME ["/data"]

CMD ["python", "-m", "app.cli", "serve"]