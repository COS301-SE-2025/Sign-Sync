import pytest
from fastapi.testclient import TestClient

import os, sys

# This workaround will be replaced by treating the folders and files as packages
tests_dir = os.path.dirname(__file__)  
backend_root = os.path.abspath(os.path.join(tests_dir, os.pardir))
service_dir = os.path.join(backend_root, "speechToText-service")
os.chdir(service_dir)
sys.path.insert(0, service_dir)

from API import app

client = TestClient(app)

import io
import wave
import numpy as np

def make_test_wav(duration_s=0.1, rate=16000):
    t = np.linspace(0, duration_s, int(rate*duration_s), endpoint=False)
    data = (0.3 * np.sin(2*np.pi*440*t) * 32767).astype(np.int16) ## 440Hz sin wave and converts to 16 bit integers (required for API)
    buf = io.BytesIO()

    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)  # mono audio (also required by API)
        w.setsampwidth(2) # 16 bits as needed and stated above
        w.setframerate(rate) # need 16000Hz sample rate
        w.writeframes(data.tobytes()) # writes to the file liek object
        
    buf.seek(0) # ewinds buffer to start from beginning
    return buf

def test_upload_audio():
    audio_buf = make_test_wav()
    response = client.post(
        "/api/upload-audio",
        files={"file": ("test.wav", audio_buf, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert isinstance(data["text"], str)



import numpy as np
import torch

# This workaround will be replaced by treating the folders and files as packages
tests_dir = os.path.dirname(__file__)  
backend_root = os.path.abspath(os.path.join(tests_dir, os.pardir))
service_dir1 = os.path.join(backend_root, "alphabetTranslate-service")
os.chdir(service_dir1)
sys.path.insert(0, service_dir1)

import translator_api

# When landmarks are at the "wrist" (1), they are normalized to 0. So if all landmarks are 1 then the normalized coordinates will all be 0
# Also tests format of output
def test_normalize_keypoints_zero_scale():
    landmarks = [translator_api.Landmark(x=1, y=1, z=1) for _ in range(21)]
    norm = translator_api.normalize_keypoints(landmarks)
    assert isinstance(norm, np.ndarray)
    assert norm.shape == (63,)
    assert np.allclose(norm, np.zeros(63))

# Same concept except one landmark is non-zero
def test_normalize_keypoints_nonzero():
    coords = [(0, 0, 0)] + [(0, 0, 0)] * 8 + [(3, 4, 0)] + [(0, 0, 0)] * 11
    landmarks = [translator_api.Landmark(x=x, y=y, z=z) for x, y, z in coords]
    norm = translator_api.normalize_keypoints(landmarks)

    expected = np.zeros(63)
    expected[27] = 0.6  
    expected[28] = 0.8 
    assert np.allclose(norm, expected, atol=1e-6)

# monkey patch used to mock PyTorch
def test_predict_endpoint(monkeypatch):
    class DummyModel:
        def __call__(self, tensor):
            batch_size = tensor.shape[0]
            num_labels = len(translator_api.index_to_label)
            out = torch.zeros((batch_size, num_labels))
            out[:, 0] = 1
            return out

    monkeypatch.setattr(translator_api, 'model', DummyModel())
    monkeypatch.setitem(translator_api.index_to_label, 0, 'dummy_label')

    client = TestClient(translator_api.app)
    payload = {'keypoints': [{'x': 0, 'y': 0, 'z': 0}] * 21}
    response = client.post('/predict', json=payload)

    assert response.status_code == 200
    assert response.json() == {'prediction': 'dummy_label'}
