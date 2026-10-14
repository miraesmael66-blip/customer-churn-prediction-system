import numpy as np
import pytest
from fastapi.testclient import TestClient

import main


class FakeModel:
    feature_names_in_ = np.array(main.MODEL_FEATURES)
    classes_ = np.array([0, 1])

    def predict(self, frame):
        assert list(frame.columns) == main.MODEL_FEATURES
        assert "Status" not in frame.columns
        assert "Churn" not in frame.columns
        assert "Avg_Usage_Duration" in frame.columns
        return np.array([1])

    def predict_proba(self, frame):
        # class 0 probability is 0.25; class 1 probability is 0.75
        return np.array([[0.25, 0.75]])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda path: FakeModel())
    with TestClient(main.app) as test_client:
        yield test_client


def valid_payload():
    return {
        "Call  Failure": 8,
        "Complains": 0,
        "Subscription  Length": 38,
        "Charge  Amount": 0,
        "Seconds of Use": 4370,
        "Frequency of use": 71,
        "Frequency of SMS": 5,
        "Distinct Called Numbers": 17,
        "Age Group": 3,
        "Tariff Plan": 1,
        "Age": 30,
        "Customer Value": 197.64,
    }


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["prediction_endpoint"] == "POST /predict"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_loaded": True}


def test_valid_prediction_and_churn_probability(client):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == 1
    assert body["churn_label"] == "Churn"
    assert body["churn_probability"] == pytest.approx(0.75)


def test_invalid_data_type(client):
    payload = valid_payload()
    payload["Call  Failure"] = "many"
    assert client.post("/predict", json=payload).status_code == 422


def test_invalid_category(client):
    payload = valid_payload()
    payload["Complains"] = 3
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_required_field(client):
    payload = valid_payload()
    payload.pop("Age")
    assert client.post("/predict", json=payload).status_code == 422


def test_out_of_observed_range(client):
    payload = valid_payload()
    payload["Age"] = 100
    assert client.post("/predict", json=payload).status_code == 422


def test_unexpected_field_rejected(client):
    payload = valid_payload()
    payload["Status"] = 1
    assert client.post("/predict", json=payload).status_code == 422


def test_probability_maps_to_actual_churn_class(client, monkeypatch):
    class ReversedClassModel(FakeModel):
        classes_ = np.array([1, 0])

        def predict_proba(self, frame):
            # class 1 is at index 0, not index 1
            return np.array([[0.8, 0.2]])

    monkeypatch.setattr(main, "load_model", lambda path: ReversedClassModel())
    with TestClient(main.app) as test_client:
        response = test_client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    assert response.json()["churn_probability"] == pytest.approx(0.8)


def test_missing_artifact_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model artifact is missing"):
        main.load_model(tmp_path / "does-not-exist.joblib")


def test_inference_error_returns_safe_500(monkeypatch):
    class BrokenModel(FakeModel):
        def predict(self, frame):
            raise RuntimeError("sensitive internal details")

    monkeypatch.setattr(main, "load_model", lambda path: BrokenModel())
    with TestClient(main.app) as test_client:
        response = test_client.post("/predict", json=valid_payload())
    assert response.status_code == 500
    assert "sensitive internal details" not in response.text


def test_health_not_ready_when_model_missing(monkeypatch):
    monkeypatch.setattr(main, "load_model", lambda path: (_ for _ in ()).throw(FileNotFoundError()))
    with TestClient(main.app) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 503
