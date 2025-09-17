import pytest
import requests

ENDPOINT = "http://localhost:8006"

def test_predict():
    payload = {"sentence": "I Go"}
    response = requests.post(ENDPOINT + "/predict", json=payload)
    assert response.status_code == 200
    
    payload = {}
    response = requests.post(ENDPOINT + "/predict", json=payload)
    assert response.status_code == 422


