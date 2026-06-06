"""
Script de réentraînement automatisé.
Déclenché par : Airflow DAG, drift détecté, ou nouvelles données.
"""
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def should_retrain(trigger: str = "scheduled") -> bool:
    return trigger in ["scheduled", "drift", "new_data", "performance_drop"]


def run_retraining(trigger: str = "scheduled", epochs: int = 5):
    """Pipeline complet de réentraînement."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    logger.info(f"=== Réentraînement [{trigger}] — {datetime.now()} ===")

    if not should_retrain(trigger):
        logger.info("Réentraînement non nécessaire.")
        return None

    if trigger == "new_data":
        from src.data.download import run_ingestion
        from src.features.feature_store import save_features
        records = run_ingestion()
        save_features(records)

    from src.model.train import train
    run_id = train(epochs=epochs)

    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("plant-classifier", stages=["None"])
    if versions:
        client.transition_model_version_stage(
            name="plant-classifier",
            version=versions[0].version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"Modèle v{versions[0].version} promu en Production.")

    return run_id


if __name__ == "__main__":
    import sys
    trigger = sys.argv[1] if len(sys.argv) > 1 else "scheduled"
    run_retraining(trigger=trigger)
