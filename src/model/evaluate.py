"""
src/model/evaluate.py
Évalue le modèle en production et génère un rapport de métriques.
"""

import os
import logging
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_production_model():
    """Charge le modèle en stage Production depuis MLflow."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model_name = os.getenv("MODEL_NAME", "plant-classifier")
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pytorch.load_model(model_uri)
    return model


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Calcule les métriques de classification."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted"),
    }
    report = classification_report(y_true, y_pred, target_names=["dandelion", "grass"])
    logger.info(f"\n{report}")
    return metrics


def evaluate_on_validation_set(learn, dls) -> dict:
    """Évalue le modèle sur le jeu de validation."""
    val_preds, val_targets = learn.get_preds(dl=dls.valid)
    y_pred = val_preds.argmax(dim=1).numpy()
    y_true = val_targets.numpy()
    return compute_metrics(y_true.tolist(), y_pred.tolist())


def log_metrics_to_mlflow(metrics: dict, run_id: str = None):
    """Log les métriques dans MLflow."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    with mlflow.start_run(run_id=run_id):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    logger.info(f"Métriques loggées: {metrics}")
