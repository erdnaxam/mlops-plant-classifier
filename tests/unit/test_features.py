"""Tests unitaires du feature store."""
import pytest
import tempfile
import os
import pandas as pd
from pathlib import Path
from unittest.mock import patch

def test_save_and_load_features(tmp_path):
    with patch("src.features.feature_store.FEATURE_STORE_PATH", tmp_path):
        from src.features.feature_store import save_features, load_latest_features
        records = [
            {"url_source": "http://x.com/1.jpg", "url_s3": "s3://b/1.jpg", "label": "dandelion"},
            {"url_source": "http://x.com/2.jpg", "url_s3": "s3://b/2.jpg", "label": "grass"},
        ]
        path = save_features(records)
        df = load_latest_features()
        assert len(df) == 2
        assert set(df.columns) == {"url_source", "url_s3", "label"}
        assert "dandelion" in df["label"].values

def test_load_features_no_store(tmp_path):
    with patch("src.features.feature_store.FEATURE_STORE_PATH", tmp_path / "empty"):
        from src.features.feature_store import load_latest_features
        with pytest.raises(FileNotFoundError):
            load_latest_features()
