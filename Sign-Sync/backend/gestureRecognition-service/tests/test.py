import os
import numpy as np
import requests

API_URL = "http://127.0.0.1:8000/predict/"
FILENAME = "tests/69345.npy"
SEQ_LEN = 50
FEATURES = 126

def predict_single_file():
    if not os.path.isfile(FILENAME):
        raise FileNotFoundError(f"File '{FILENAME}' not found in the current directory.")

    sequence = np.load(FILENAME)

    if sequence.shape[1] != FEATURES:
        raise ValueError(f"Expected {FEATURES} features, but got {sequence.shape[1]}")

   
    if sequence.shape[0] > SEQ_LEN:
        sequence = sequence[:SEQ_LEN]
    elif sequence.shape[0] < SEQ_LEN:
        pad_len = SEQ_LEN - sequence.shape[0]
        sequence = np.vstack([sequence, np.zeros((pad_len, FEATURES), dtype=np.float32)])

    payload = {"sequence": sequence.tolist()}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["gloss"] 
    except requests.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")
    except KeyError:
        raise ValueError("Gloss not found in API response.")

