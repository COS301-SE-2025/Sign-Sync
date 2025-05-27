import asyncio
import websockets

async def send_audio():
    uri = "ws://localhost:8000/api/speech-to-text"
    async with websockets.connect(uri) as websocket:
        with open("test_audio.raw", "rb") as audio_file:
            while chunk := audio_file.read(4000):
                await websocket.send(chunk)
                response = await websocket.recv()
                print(f"Received response: {response}")

asyncio.run(send_audio())