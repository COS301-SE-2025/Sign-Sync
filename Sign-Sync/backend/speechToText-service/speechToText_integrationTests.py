import pytest
from fastapi.testclient import TestClient
from API import app  # Import the FastAPI app

client = TestClient(app)

# Integration Test 1: Test successful audio upload and transcription
def test_upload_audio_integration():
    # Act: Simulate uploading an actual audio file (or a mock file)
    with open("temp.wav", "rb") as f:
        response = client.post(
            "/api/upload-audio",
            files={"file": ("temp.wav", f, "audio/wav")}
        )
    
    # Assert: Ensure the transcription response is correct
    assert response.status_code == 200
    assert "text" in response.json()

# # Integration Test 2: Test error handling when a bad file is uploaded (non-audio)
# def test_upload_audio_bad_file():
#     # Act: Upload a non-audio file (e.g., a text file)
#     with open("test_text.txt", "rb") as f:
#         response = client.post(
#             "/api/upload-audio",
#             files={"file": ("test_text.txt", f, "text/plain")}
#         )

#     # Assert: Ensure that an error response is returned
#     assert response.status_code == 422
#     assert "Invalid file type" in response.json()["detail"]  # Check the error message

