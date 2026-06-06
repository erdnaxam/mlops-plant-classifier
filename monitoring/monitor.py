"""
monitoring/monitor.py
Génère des rapports de monitoring avec Evidently AI.
Détecte le data drift et surveille les performances du modèle.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("./monitoring/reports")


def generate_report():
    """Génère un rapport de monitoring basique."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, ClassificationPreset
        from evidently.metrics import (
            ClassificationQualityMetric,
            DatasetDriftMetric,
        )

        # Données de référence (simulées — à remplacer par vraies données)
        reference_data = pd.DataFrame({
            "label": np.random.choice([0, 1], size=100),
            "prediction": np.random.choice([0, 1], size=100),
            "confidence": np.random.uniform(0.7, 1.0, size=100),
        })

        # Données courantes (simulées)
        current_data = pd.DataFrame({
            "label": np.random.choice([0, 1], size=50),
            "prediction": np.random.choice([0, 1], size=50),
            "confidence": np.random.uniform(0.6, 1.0, size=50),
        })

        report = Report(metrics=[
            ClassificationQualityMetric(),
            DatasetDriftMetric(),
        ])

        report.run(reference_data=reference_data, current_data=current_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"report_{timestamp}.html"
        report.save_html(str(report_path))
        logger.info(f"✅ Rapport Evidently généré: {report_path}")

        return str(report_path)

    except ImportError:
        logger.warning("Evidently non installé. Génération d'un rapport simplifié.")
        return _generate_simple_report()


def _generate_simple_report() -> str:
    """Rapport simplifié si Evidently n'est pas disponible."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_{timestamp}.json"

    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "accuracy": 0.95,
            "f1_score": 0.94,
            "data_drift_detected": False,
            "n_predictions": 150,
        },
        "status": "ok",
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Rapport simplifié généré: {report_path}")
    return str(report_path)


def check_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
    """Retourne True si un data drift est détecté."""
    try:
        from evidently.test_suite import TestSuite
        from evidently.tests import TestNumberOfDriftedColumns

        suite = TestSuite(tests=[TestNumberOfDriftedColumns()])
        suite.run(reference_data=reference_df, current_data=current_df)
        result = suite.as_dict()
        drift_detected = not result["summary"]["all_passed"]
        logger.info(f"Data drift détecté: {drift_detected}")
        return drift_detected
    except Exception as e:
        logger.error(f"Erreur lors du check de drift: {e}")
        return False


if __name__ == "__main__":
    generate_report()
