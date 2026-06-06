"""
Téléchargement et preprocessing des images depuis GitHub.
Sauvegarde dans MinIO (S3 local).
"""
import os
import io
import logging
import requests
import boto3
from botocore.client import Config
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data"
LABELS = ["dandelion", "grass"]
N_IMAGES = 200  # 00000000 à 00000199

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"http://{os.getenv('MINIO_ENDPOINT', 'localhost:9000')}",
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        config=Config(signature_version="s3v4"),
    )

def ensure_bucket(s3_client, bucket_name: str):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except Exception:
        s3_client.create_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' créé.")

def preprocess_image(img: Image.Image, size=(224, 224)) -> Image.Image:
    """Redimensionne et convertit en RGB."""
    return img.convert("RGB").resize(size)

def download_and_upload(label: str, idx: int, bucket: str, s3_client):
    filename = f"{idx:08d}.jpg"
    url = f"{BASE_URL}/{label}/{filename}"
    s3_key = f"images/{label}/{filename}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        img = Image.open(io.BytesIO(response.content))
        img = preprocess_image(img)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        s3_client.put_object(Bucket=bucket, Key=s3_key, Body=buffer)
        logger.info(f"✅ Uploadé : {s3_key}")
        return {"url_source": url, "url_s3": f"s3://{bucket}/{s3_key}", "label": label}

    except Exception as e:
        logger.error(f"❌ Erreur {url}: {e}")
        return None

def run_ingestion(bucket: str = "plant-classifier"):
    s3_client = get_s3_client()
    ensure_bucket(s3_client, bucket)

    records = []
    for label in LABELS:
        for idx in range(N_IMAGES):
            record = download_and_upload(label, idx, bucket, s3_client)
            if record:
                records.append(record)

    logger.info(f"Ingestion terminée : {len(records)} images uploadées.")
    return records

if __name__ == "__main__":
    run_ingestion()
