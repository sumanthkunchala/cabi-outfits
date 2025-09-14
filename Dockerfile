# Dockerfile (CPU-only)
FROM python:3.11-slim

# System deps for numpy/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libjpeg62-turbo libpng16-16 libglib2.0-0 libgl1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -U pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio && \
    pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY src /app/src

# Catalog + artifacts (adjust if you externalize images later)
COPY embeddings /app/embeddings
COPY data/tmp /app/data/tmp
COPY data/images /app/data/images

ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "cabi_outfits.api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
