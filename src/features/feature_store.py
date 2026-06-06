"""
Feature Store simplifié — stockage local de métadonnées d'images.
"""
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

FEATURE_STORE_PATH = Path(os.getenv("FEATURE_STORE_PATH", "./models/feature_store"))

def init_store():
    FEATURE_STORE_PATH.mkdir(parents=True, exist_ok=True)

def save_features(records: list, version: str = None):
    """Sauvegarde les métadonnées images dans le feature store."""
    init_store()
    version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(records)
    path = FEATURE_STORE_PATH / f"features_{version}.parquet"
    df.to_parquet(path, index=False)
    # Pointer vers la version courante
    (FEATURE_STORE_PATH / "latest.txt").write_text(str(path))
    return str(path)

def load_latest_features() -> pd.DataFrame:
    """Charge les features de la dernière version disponible."""
    init_store()
    latest_file = FEATURE_STORE_PATH / "latest.txt"
    if not latest_file.exists():
        raise FileNotFoundError("Aucune feature disponible. Lancez d'abord l'ingestion.")
    path = latest_file.read_text().strip()
    return pd.read_parquet(path)

def get_train_test_split(test_size: float = 0.2):
    """Retourne train/test DataFrames depuis le feature store."""
    from sklearn.model_selection import train_test_split
    df = load_latest_features()
    return train_test_split(df, test_size=test_size, stratify=df["label"], random_state=42)
