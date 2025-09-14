# Dockerfile (CPU-only)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libjpeg62-turbo libpng16-16 libglib2.0-0 libgl1 libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -U pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio && \
    pip install --no-cache-dir -r /app/requirements.txt

# app code + artifacts
COPY src /app/src
COPY embeddings /app/embeddings
COPY data/tmp /app/data/tmp
COPY data/images /app/data/images

ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "cabi_outfits.api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
