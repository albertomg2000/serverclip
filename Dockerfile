FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY server1.py .
COPY text_embeddings_h14.pt .
COPY text_embeddings_modelos_h14.pt .

# Render pone PORT; por defecto 8080 si no está
ENV PORT=8080

# Un solo worker para no duplicar memoria
CMD ["python", "-m", "uvicorn", "server1:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

