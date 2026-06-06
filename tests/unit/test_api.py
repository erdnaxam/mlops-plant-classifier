"""Tests unitaires de l'API FastAPI."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
from PIL import Image

# Mock du modèle pour éviter de télécharger depuis MinIO
@pytest.fixture
def mock_learner():
    mock = MagicMock()
    mock.predict.return_value = ("dandelion", 0, [0.95, 0.05])
    return mock

@pytest.fixture
def client(mock_learner):
    with patch("api.main.get_learner", return_value=mock_learner):
        from api.main import app
        return TestClient(app)

def make_test_image():
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_returns_result(client):
    img_buf = make_test_image()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] in ["dandelion", "grass"]

def test_predict_invalid_file(client):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400

def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "architecture" in data
    assert "classes" in data
