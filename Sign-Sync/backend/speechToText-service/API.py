# from fastapi import FastAPI, WebSocket
# from fastapi.middleware.cors import CORSMiddleware
# from vosk import Model, KaldiRecognizer
# import json
# import asyncio
# import os

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], # This needs to change before production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# SAMPLE_RATE = 16000

# def get_model():
#     return Model("models/vosk-model-small-en-us-0.15")


# @app.websocket("/api/speech-to-text")
# async def speech_to_text(websocket: WebSocket):
#     await websocket.accept()
#     model = get_model()
#     recognizer = KaldiRecognizer(model, SAMPLE_RATE)
#     recognizer.SetWords(True)

#     try:
#         while True:
#             data = await websocket.receive_bytes()
#             if recognizer.AcceptWaveform(data):
#                 result = json.loads(recognizer.Result())
#                 await websocket.send_text(result.get("text", ""))
#             else:
#                 partial_result = json.loads(recognizer.PartialResult())
#                 await websocket.send_text(partial_result.get("partial", ""))
#     except Exception as e:
#         print(f"Error: {e}")


from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer
import subprocess
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = Model("models/vosk-model-small-en-us-0.15")

@app.post("/api/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    input_path = "temp.wav"
    raw_path = "temp.raw"

    # Save uploaded audio to disk
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Convert to 16-bit mono 16kHz PCM raw using FFmpeg
    # subprocess.run([
    #     "ffmpeg", "-y", "-i", input_path,
    #     "-ac", "1", "-ar", "16000", "-f", "s16le", raw_path
    # ], check=True)

    ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg", "ffmpeg.exe")

    subprocess.run([
        ffmpeg_path, "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-f", "s16le", raw_path
    ], check=True)

    # Transcribe with Vosk
    rec = KaldiRecognizer(model, 16000)
    with open(raw_path, "rb") as f:
        while True:
            data = f.read(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)

    result = json.loads(rec.FinalResult())
    return {"text": result.get("text", "")}