from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer
import json
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # This needs to change before production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = Model("models/vosk-model-small-en-us-0.15")
SAMPLE_RATE = 16000

@app.websocket("/api/speech-to-text")
async def speech_to_text(websocket: WebSocket):
    await websocket.accept()
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)

    try:
        while True:
            data = await websocket.receive_bytes()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                await websocket.send_text(result.get("text", ""))
            else:
                partial_result = json.loads(recognizer.PartialResult())
                await websocket.send_text(partial_result.get("partial", ""))
    except Exception as e:
        print(f"Error: {e}")
