import sys
import os
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import API
from API import app, SEQ_LEN, FEATURES

API.model = MagicMock()
API.model.predict.return_value = np.array([[0.1]*10 + [0.9] + [0.0]*25])
API.label_map = {str(i): f"mocked_gloss_{i}" for i in range(36)}

client = TestClient(app)

def generate_valid_sequence():
    return [[float(i % 10) for i in range(FEATURES)] for _ in range(SEQ_LEN)]

def test_predict_valid_input():
    payload = { "sequence": generate_valid_sequence() }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    assert response.json()["gloss"] == "mocked_gloss_10"

def test_sequence_too_short():
    short = generate_valid_sequence()[:-1]
    response = client.post("/predict/", json={ "sequence": short })
    assert response.status_code == 422

def test_sequence_too_long():
    long = generate_valid_sequence() + [[0.0]*FEATURES]
    response = client.post("/predict/", json={ "sequence": long })
    assert response.status_code == 422

def test_frame_with_wrong_feature_length():
    bad = generate_valid_sequence()
    bad[0] = [0.0] * (FEATURES - 1)
    response = client.post("/predict/", json={ "sequence": bad })
    assert response.status_code == 422

def test_invalid_input_format():
    response = client.post("/predict/", json={ "sequence": "not a list" })
    assert response.status_code == 422
