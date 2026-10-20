from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_prediction_endpoint():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()

    assert body["prediction"] in (0, 1)
    assert body["label"] in ("positive", "negative")
    assert 0 <= body["probability_heart_disease"] <= 1
    assert body["model_name"] == "Logistic Regression"
    assert "Educational demonstration only" in body["disclaimer"]


def test_prediction_rejects_missing_feature():
    payload = VALID_PAYLOAD.copy()
    payload.pop("age")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
