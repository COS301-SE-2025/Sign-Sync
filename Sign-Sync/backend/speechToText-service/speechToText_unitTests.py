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

# import pytest
# from fastapi.testclient import TestClient
# from API import app  # Correct import since API.py is in the same directory

# from unittest.mock import patch, MagicMock
# import os
# import subprocess

# client = TestClient(app)

# @patch("API.subprocess.run")  # Corrected path for subprocess.run
# @patch("API.KaldiRecognizer")  # Corrected path for KaldiRecognizer
# def test_upload_audio(mock_recognizer, mock_subprocess):
#     # Arrange: Mock the recognizer behavior to simulate transcription
#     mock_recognizer_instance = MagicMock()
#     mock_recognizer.return_value = mock_recognizer_instance
#     mock_recognizer_instance.AcceptWaveform.return_value = True
#     mock_recognizer_instance.FinalResult.return_value = '{"text": "Hello world"}'

#     # Mock subprocess to avoid executing FFmpeg
#     mock_subprocess.return_value = None

#     # Simulate uploading an audio file (we're using a small file as an example)
#     test_audio = open("temp.wav", "rb")

#     # Act: Send the POST request with the mocked file
#     response = client.post(
#         "/api/upload-audio",
#         files={"file": ("temp.wav", test_audio, "audio/wav")}
#     )

#     # Assert: Verify that the transcription was successful
#     assert response.status_code == 200
#     assert response.json() == {"text": "Hello world"}

#     # Check that subprocess was called with the expected arguments
#     mock_subprocess.assert_called_once_with(
#         os.path.join(os.path.dirname(__file__), "ffmpeg", "ffmpeg.exe"),
#         "-y", "-i", "temp.wav",
#         "-ac", "1", "-ar", "16000", "-f", "s16le", "temp.raw",
#         check=True  # Include check=True here
#     )

#     # Check that KaldiRecognizer was called with correct parameters
#     mock_recognizer_instance.AcceptWaveform.assert_called()

#     test_audio.close()


# @patch("API.subprocess.run")  # Corrected path for subprocess.run
# @patch("API.KaldiRecognizer")  # Corrected path for KaldiRecognizer
# def test_upload_audio_subprocess_error(mock_recognizer, mock_subprocess):
#     # Arrange: Mock the recognizer behavior
#     mock_recognizer_instance = MagicMock()
#     mock_recognizer.return_value = mock_recognizer_instance
#     mock_recognizer_instance.AcceptWaveform.return_value = True
#     mock_recognizer_instance.FinalResult.return_value = '{"text": "Hello world"}'

#     # Simulate subprocess error by raising an exception
#     mock_subprocess.side_effect = subprocess.CalledProcessError(1, "ffmpeg", "FFmpeg error")

#     test_audio = open("temp.wav", "rb")

#     # Act: Send the POST request
#     response = client.post(
#         "/api/upload-audio",
#         files={"file": ("temp.wav", test_audio, "audio/wav")}
#     )

#     # Assert: Verify that the response is an error (500 status code)
#     assert response.status_code == 500
#     assert "Error processing audio" in response.text

#     test_audio.close()
