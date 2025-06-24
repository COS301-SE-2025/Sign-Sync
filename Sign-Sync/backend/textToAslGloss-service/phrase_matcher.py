import json
from rapidfuzz import process

with open("textToAslGloss-service/phrase_db.json", "r") as f:
    PHRASE_DB = json.load(f)

def match_phrase(text, threshold=95):
    phrases = list(PHRASE_DB.keys())
    text = text.lower().strip()
    match, score, _ = process.extractOne(text, phrases)
    if score >= threshold:
        return PHRASE_DB[match]
    return None
