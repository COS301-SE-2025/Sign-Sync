import json
import os
import numpy as np
import requests
import pandas as pd
import random

json_path = "data/WLASL_v0.3.json"
npy_dir = "cached"
api_url = "http://127.0.0.1:8000/predict/"

SEQ_LEN = 50
FEATURES = 126

target_glosses = [
    "computer", "dog", "go", "hearing", "bed", "before", "bird", "cow", "drink", "family",
    "graduate", "help", "accident", "apple", "blue", "bowling", "chair", "fine", "dark",
    "can", "corn", "all", "cousin", "book", "hat", "black", "change", "color", "cool",
    "dance", "deaf", "finish", "fish", "hot", "kiss", "candy", "clothes"
]

with open(json_path, "r") as f:
    data = json.load(f)

results = []
num_correct = 0
num_total = 0

for gloss in target_glosses:
    candidates = []

    for entry in data:
        if entry.get("gloss") != gloss:
            continue

        for instance in entry.get("instances", []):
            video_id = instance.get("video_id")
            npy_path = os.path.join(npy_dir, f"{video_id}.npy")
            if os.path.isfile(npy_path):
                candidates.append((video_id, npy_path))

    if not candidates:
        results.append({
            "Gloss": gloss,
            "Video ID": "Not found",
            "Predicted Gloss": "-",
            "Class ID": "-",
            "Confidence": "-",
            "Correct": "-"
        })
        continue

    video_id, npy_path = random.choice(candidates)
    sequence = np.load(npy_path)

    if sequence.shape[0] > SEQ_LEN:
        sequence = sequence[:SEQ_LEN]
    elif sequence.shape[0] < SEQ_LEN:
        pad_len = SEQ_LEN - sequence.shape[0]
        sequence = np.vstack([sequence, np.zeros((pad_len, FEATURES), dtype=np.float32)])

    payload = { "sequence": sequence.tolist() }

    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            predicted_gloss = result["gloss"]
            correct = predicted_gloss == gloss
            if correct:
                num_correct += 1
            num_total += 1

            results.append({
                "Gloss": gloss,
                "Video ID": video_id,
                "Predicted Gloss": predicted_gloss,
                "Class ID": result["predicted_class"],
                "Confidence": round(result["confidence"], 3),
                "Correct": "yes" if correct else "no"
            })
        else:
            results.append({
                "Gloss": gloss,
                "Video ID": video_id,
                "Predicted Gloss": "API Error",
                "Class ID": "-",
                "Confidence": "-",
                "Correct": "-"
            })
    except Exception as e:
        results.append({
            "Gloss": gloss,
            "Video ID": video_id,
            "Predicted Gloss": f"wrong {str(e)}",
            "Class ID": "-",
            "Confidence": "-",
            "Correct": "-"
        })

df = pd.DataFrame(results)
print("\n Prediction Results:")
print(df.to_string(index=False))

if num_total > 0:
    accuracy = num_correct / num_total * 100
    print(f"\nCorrect Accuracy: {num_correct} / {num_total} correct ({accuracy:.2f}%)")
else:
    print("\n No valid predictions made to compute accuracy.")
