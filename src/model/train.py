"""
Entraînement du modèle FastAI (ResNet18) avec tracking MLflow.
"""
import os
import io
import tempfile
import logging
import boto3
import mlflow
import mlflow.pytorch
from botocore.client import Config
from fastai.vision.all import (
    ImageDataLoaders, aug_transforms, vision_learner,
    resnet18, accuracy, error_rate, ClassificationInterpretation
)
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET = os.getenv("MINIO_BUCKET", "plant-classifier")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "plant-classifier")


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"http://{os.getenv('MINIO_ENDPOINT', 'localhost:9000')}",
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        config=Config(signature_version="s3v4"),
    )


def download_images_from_s3(local_dir: Path):
    """Télécharge les images depuis MinIO vers un dossier local temporaire."""
    s3 = get_s3_client()
    logger.info("Téléchargement des images depuis MinIO...")

    for label in ["dandelion", "grass"]:
        (local_dir / label).mkdir(parents=True, exist_ok=True)
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=f"images/{label}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = Path(key).name
                dest = local_dir / label / filename
                s3.download_file(BUCKET, key, str(dest))

    logger.info(f"Images téléchargées dans {local_dir}")


def train(epochs: int = 5, lr: float = 1e-3):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir)
        download_images_from_s3(data_path)

        # DataLoaders FastAI
        dls = ImageDataLoaders.from_folder(
            data_path,
            valid_pct=0.2,
            seed=42,
            item_tfms=aug_transforms(size=224),
        )

        # Modèle : ResNet18 pré-entraîné
        learn = vision_learner(dls, resnet18, metrics=[accuracy, error_rate])

        with mlflow.start_run() as run:
            mlflow.log_params({
                "epochs": epochs,
                "learning_rate": lr,
                "architecture": "resnet18",
                "img_size": 224,
                "train_size": len(dls.train_ds),
                "valid_size": len(dls.valid_ds),
            })

            # Entraînement
            learn.fine_tune(epochs, base_lr=lr)

            # Métriques finales
            val_loss, acc, err = learn.validate()
            mlflow.log_metrics({
                "val_loss": float(val_loss),
                "accuracy": float(acc),
                "error_rate": float(err),
            })

            logger.info(f"Accuracy: {acc:.4f} | Error rate: {err:.4f}")

            # Sauvegarde du modèle dans MLflow
            mlflow.pytorch.log_model(
                learn.model,
                artifact_path="model",
                registered_model_name="plant-classifier",
            )

            # Export vers MinIO
            model_path = Path(tmpdir) / "model.pkl"
            learn.export(model_path)
            s3 = get_s3_client()
            s3.upload_file(str(model_path), BUCKET, "models/plant_classifier.pkl")
            logger.info("Modèle sauvegardé dans MinIO : s3://plant-classifier/models/plant_classifier.pkl")

            return run.info.run_id


if __name__ == "__main__":
    run_id = train(epochs=5)
    logger.info(f"Run MLflow : {run_id}")
