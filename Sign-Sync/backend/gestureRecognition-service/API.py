from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
import numpy as np
import tensorflow as tf
import json
import os

app = FastAPI()

model = None
label_map = {}
SEQ_LEN = 50
FEATURES = 126

class KeypointSequence(BaseModel):
    sequence: List[List[float]]

    @validator("sequence")
    def validate_sequence_shape(cls, v):
        if len(v) != SEQ_LEN:
            raise ValueError(f"Sequence must have {SEQ_LEN} timesteps")
        for i, frame in enumerate(v):
            if len(frame) != FEATURES:
                raise ValueError(f"Frame {i} must have {FEATURES} features")
        return v

@app.on_event("startup")
def load_model_and_labels():
    global model, label_map
    model_path = "best_model_val_accccccd.keras"
    label_map_path = "label_map.json"

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    if not os.path.exists(label_map_path):
        raise RuntimeError(f"Label map file not found: {label_map_path}")

    model = tf.keras.models.load_model(model_path)

    with open(label_map_path, "r") as f:
        label_map = json.load(f)

@app.post("/predict/")
def predict_gloss(data: KeypointSequence):
    try:
        sequence = np.array(data.sequence, dtype=np.float32)
        sequence = np.expand_dims(sequence, axis=0)

        pred_probs = model.predict(sequence)[0]
        class_idx = int(np.argmax(pred_probs))
        gloss = label_map[str(class_idx)]
        confidence = float(pred_probs[class_idx])

        return {
            "predicted_class": class_idx,
            "gloss": gloss,
            "confidence": confidence
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
