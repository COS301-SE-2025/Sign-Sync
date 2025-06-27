import sys
import os
import numpy as np
import pytest
import test
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from API import app, SEQ_LEN, FEATURES

client = TestClient(app)

def generate_valid_sequence():
    """Generates a valid keypoint sequence of shape (SEQ_LEN, FEATURES)."""
    return [[float(i % 10) for i in range(FEATURES)] for _ in range(SEQ_LEN)]


def test_integration_accuracy():
    gloss = test.predict_single_file()
    assert gloss == "go"


def test_sequence_too_short():
    short_seq = generate_valid_sequence()[:-1]
    payload = {"sequence": short_seq}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422


def test_sequence_too_long():
    long_seq = generate_valid_sequence() + [[0.0] * FEATURES]
    payload = {"sequence": long_seq}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422


def test_frame_with_wrong_feature_length():
    bad_seq = generate_valid_sequence()
    bad_seq[0] = [0.0] * (FEATURES - 1) 
    payload = {"sequence": bad_seq}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422


def test_invalid_input_format():
    payload = {"sequence": "this is not a list of lists"}
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422
