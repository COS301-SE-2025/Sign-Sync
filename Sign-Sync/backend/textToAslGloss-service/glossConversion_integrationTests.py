from fastapi.testclient import TestClient
from textConversion_api import app

client = TestClient(app)

def test_database_phrase():
    response = client.post("/translate", json={"sentence": "how are you"})
    assert response.status_code == 200
    assert response.json() == {"source": "database", "gloss": "HOW YOU"}

def test_template_applied():
    response = client.post("/translate", json={"sentence": "What time do we leave tomorrow?"})
    assert response.status_code == 200
    assert response.json()["source"] == "template"
    assert "TOMORROW" in response.json()["gloss"]

def test_fallback_model():
    response = client.post("/translate", json={"sentence": "I am happy"})
    assert response.status_code == 200
    assert response.json()["source"] == "model"
    assert response.json()["gloss"] == "I HAPPY"
