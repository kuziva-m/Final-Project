# Runs the Flask ledger-OCR app (web/app.py) on Hugging Face Spaces.
FROM python:3.11-slim

# tesseract-ocr is the OCR engine used by pytesseract.
# libgl1 / libglib2.0-0 are required for opencv-python to import on a
# headless slim image (otherwise: "ImportError: libGL.so.1: cannot open
# shared object file").
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY . .

EXPOSE 7860

CMD ["python", "-m", "web.app"]
