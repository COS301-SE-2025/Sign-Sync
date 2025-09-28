import pytest
import requests

# Just note that this testing must be run with "python -m pytest -v -s"
# This is opposed to the unit testing which can just be run like normla python files

ENDPOINT = "http://localhost:8006"

def test_predict():
    payload = {"sentence": "I Go"}
    response = requests.post(ENDPOINT + "/predict", json=payload)
    assert response.status_code == 200
    
    payload = {}
    response = requests.post(ENDPOINT + "/predict", json=payload)
    assert response.status_code == 422

def test_translate():
    payload = {"text": ""}
    response = requests.post(ENDPOINT + "/translate", json=payload)
    data = response.json()
    assert data["translation"] == ""
    assert response.status_code == 200

    payload = {"text": "       "}
    response = requests.post(ENDPOINT + "/translate", json=payload)
    data = response.json()
    assert data["translation"] == ""
    assert response.status_code == 200

    payload = {"text": "STORE I GO"}
    response = requests.post(ENDPOINT + "/translate", json=payload)
    data = response.json()
    assert data["translation"] == "I go to the store."
    assert response.status_code == 200

