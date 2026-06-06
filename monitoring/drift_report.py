"""
Monitoring avec Evidently — Détection de data drift et dégradation des performances.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import DatasetDriftMetric

REPORT_DIR = Path("./monitoring/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_THRESHOLD = 0.3  # Au-delà, on déclenche un réentraînement


def generate_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Compare les distributions de features entre données de référence et courantes.
    Retourne un dict avec le statut du drift et le chemin du rapport HTML.
    """
    report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
    report.run(reference_data=reference_df, current_data=current_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"drift_report_{timestamp}.html"
    report.save_html(str(report_path))

    result = report.as_dict()
    drift_score = result["metrics"][1]["result"]["drift_share"]
    drift_detected = drift_score > DRIFT_THRESHOLD

    summary = {
        "timestamp": timestamp,
        "drift_score": drift_score,
        "drift_detected": drift_detected,
        "report_path": str(report_path),
        "threshold": DRIFT_THRESHOLD,
    }

    # Sauvegarde du résumé JSON
    summary_path = REPORT_DIR / f"summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def generate_performance_report(y_true, y_pred, y_prob) -> dict:
    """Rapport de performance du modèle sur les données de production."""
    df = pd.DataFrame({
        "target": y_true,
        "prediction": y_pred,
        "dandelion_prob": [p[0] for p in y_prob],
        "grass_prob": [p[1] for p in y_prob],
    })

    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=None, current_data=df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"performance_report_{timestamp}.html"
    report.save_html(str(report_path))

    return {"timestamp": timestamp, "report_path": str(report_path)}


def check_and_trigger_retraining(drift_summary: dict) -> bool:
    """Retourne True si un réentraînement doit être déclenché."""
    if drift_summary["drift_detected"]:
        print(f"⚠️  Drift détecté ({drift_summary['drift_score']:.3f} > {DRIFT_THRESHOLD}). Réentraînement nécessaire.")
        return True
    print(f"✅ Pas de drift significatif ({drift_summary['drift_score']:.3f}).")
    return False


if __name__ == "__main__":
    # Exemple de test avec données simulées
    np.random.seed(42)
    ref = pd.DataFrame({"feature_1": np.random.normal(0, 1, 200), "feature_2": np.random.normal(0, 1, 200)})
    cur = pd.DataFrame({"feature_1": np.random.normal(0.5, 1.5, 100), "feature_2": np.random.normal(0.2, 1, 100)})
    summary = generate_drift_report(ref, cur)
    print(json.dumps(summary, indent=2))
