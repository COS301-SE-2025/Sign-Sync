import requests
import pytest

ENDPOINT = "http://localhost:8007/api/"

def test_asl_to_english_translate():
    payload = {"text": "store i go"}

    response = requests.post(ENDPOINT + "word/translate", json=payload)

    assert response.status_code == 200

    response.raise_for_status()
    data = response.json()
    translation = data.get("translation", "")

    assert translation == "I go to the store."

    payload = {"text": "i go house"}

    response = requests.post(ENDPOINT + "word/translate", json=payload)

    assert response.status_code == 200
    response.raise_for_status()
    data = response.json()
    translation = data.get("translation", "")
    assert translation == "I go to the house."