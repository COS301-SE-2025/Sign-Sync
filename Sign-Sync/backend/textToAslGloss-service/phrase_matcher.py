import json
from rapidfuzz import process

# with open("textToAslGloss-service/phrase_db.json", "r") as f:
#     PHRASE_DB = json.load(f)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHRASE_DB_PATH = os.path.join(BASE_DIR, "phrase_db.json")

with open(PHRASE_DB_PATH, "r", encoding="utf-8") as f:
    PHRASE_DB = json.load(f)

def match_phrase(text, threshold=95):
    phrases = list(PHRASE_DB.keys())
    text = text.lower().strip()
    match, score, _ = process.extractOne(text, phrases)
    if score >= threshold:
        return PHRASE_DB[match]
    return None
