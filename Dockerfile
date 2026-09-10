FROM python:3.11-slim

# System deps: OpenCV headless runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements change).
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source.
COPY preprocessing/ ./preprocessing/
COPY api/ ./api/

# Render injects $PORT; default 8000 for local runs.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port $PORT"]
