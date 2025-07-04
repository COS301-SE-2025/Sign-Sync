from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from phrase_matcher import match_phrase
from gloss_converter import convert_to_gloss
from asl_templates import apply_asl_template

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

class TranslationRequest(BaseModel):
    sentence: str

class TranslationResponse(BaseModel):
    source: str
    gloss: str


@app.post("/translate", response_model=TranslationResponse)
def translate(req: TranslationRequest):
    phrase_result = match_phrase(req.sentence)
    if phrase_result:
        return {"source": "database", "gloss": phrase_result}

    # Try applying ASL grammar template
    template_result = apply_asl_template(req.sentence)
    if template_result:
        return {"source": "template", "gloss": template_result}

    # Fall back to rule-based gloss
    fallback = convert_to_gloss(req.sentence)
    return {"source": "model", "gloss": " ".join(fallback.split())}