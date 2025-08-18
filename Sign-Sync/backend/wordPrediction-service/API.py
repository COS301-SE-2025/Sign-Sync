from __future__ import annotations
import os
from Trie import Trie
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Tuple, List
import random

class PredictIn(BaseModel):
    sentence: str
    add_k: float = 0.0
    min_count: int = 1
    backoff: bool = True

class PredictOut(BaseModel):
    token: Optional[str]
    prob: float

JSON_PATH = "trie.json"
app = FastAPI()

@app.on_event("startup")
def load_trie():
    if not os.path.exists(JSON_PATH):
        raise HTTPException(status_code=404, detail="Trie file not found")
    app.state.trie = Trie.load_json(JSON_PATH)

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn) -> PredictOut:
    trie = app.state.trie

    preds: List[Tuple[str, float]] = trie.predict_next(
        body.sentence,
        top_k=10_000_000,  
        min_count=body.min_count,
        backoff=body.backoff,
        add_k=body.add_k,
    )

    if not preds:
        return PredictOut(token=None, prob=0.0)

    max_p = preds[0][1]
    TIE_EPS = 1e-12
    tied = [tok for tok, p in preds if abs(p - max_p) <= TIE_EPS]
    token = random.choice(tied)
    return PredictOut(token=token, prob=max_p)
