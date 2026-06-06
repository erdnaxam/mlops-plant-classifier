"""
API FastAPI — Serving du modèle de classification d'images.
"""
import os
import io
import logging
import boto3
import tempfile
from botocore.client import Config
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Plant Classifier API",
    description="API de classification d'images : Dandelion vs Grass",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle chargé au démarrage
_learner = None

def get_learner():
    global _learner
    if _learner is not None:
        return _learner

    logger.info("Chargement du modèle depuis MinIO...")
    try:
        from fastai.vision.all import load_learner
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
            config=Config(signature_version="s3v4"),
        )
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            s3.download_fileobj("plant-classifier", "models/plant_classifier.pkl", f)
            tmp_path = f.name

        _learner = load_learner(tmp_path)
        logger.info("Modèle chargé avec succès.")
    except Exception as e:
        logger.error(f"Impossible de charger le modèle : {e}")
        raise RuntimeError(f"Modèle non disponible : {e}")

    return _learner


@app.get("/health", tags=["Système"])
def health():
    """Vérification que l'API est opérationnelle."""
    return {"status": "ok", "service": "plant-classifier-api"}


@app.post("/predict", tags=["Prédiction"])
async def predict(file: UploadFile = File(...)):
    """
    Envoie une image et reçoit une prédiction (dandelion ou grass).

    - **file** : image JPG ou PNG
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        learn = get_learner()
        from fastai.vision.all import PILImage
        pred_class, pred_idx, probs = learn.predict(PILImage.create(img))

        return {
            "prediction": str(pred_class),
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                "dandelion": float(probs[0]),
                "grass": float(probs[1]),
            },
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur de prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")


@app.get("/model/info", tags=["Modèle"])
def model_info():
    """Retourne des informations sur le modèle chargé."""
    return {
        "model_name": "plant-classifier",
        "architecture": "ResNet18",
        "framework": "FastAI",
        "classes": ["dandelion", "grass"],
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
    }
