FROM python:3.10-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir fastai==2.7.15 torch==2.2.0 torchvision==0.17.0 \
    mlflow==2.11.0 boto3==1.34.0 fastapi==0.110.0 uvicorn==0.27.0 \
    python-multipart==0.0.9 pillow==10.2.0 python-dotenv==1.0.0

# Code de l'application
COPY api/ ./api/
COPY src/ ./src/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
