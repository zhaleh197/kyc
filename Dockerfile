# CPU-only serving image. Models are trained elsewhere (Kaggle GPU) and
# mounted or copied into /app/models at deploy time - they are never baked
# into the image, so a retrained model does not mean a rebuilt container.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# opencv-python-headless still needs libgl/libglib for a few codecs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/base.txt

COPY kyc/ kyc/
COPY scripts/ scripts/

# Run as a non-root user: this service receives identity documents.
RUN useradd --create-home --uid 10001 kyc && chown -R kyc:kyc /app
USER kyc

ENV KYC_MODELS_DIR=/app/models
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "kyc.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
