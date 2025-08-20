FROM python:3.11-slim

# Configuración básica
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# (Opcional) descargar un modelo más ligero para que Render free no muera
RUN python - <<'PY'
import open_clip
open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
print("Pesos descargados")
PY

# Copiar código y pesos
COPY server1.py .
COPY text_embeddings_h14.pt .
COPY text_embeddings_modelos_h14.pt .

# Render usa PORT
ENV PORT=8080

CMD ["uvicorn", "server1:app", "--host", "0.0.0.0", "--port", "8080"]
