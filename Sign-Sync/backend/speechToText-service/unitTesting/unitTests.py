# unitTests.py
import sys
import os
import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect
from unittest.mock import patch, MagicMock

# Ensure parent directory is on sys.path to import API.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_websocket_full_result():
    with patch("API.get_model") as mock_get_model, patch("API.KaldiRecognizer") as mock_recognizer_class:
        from API import app
        client = TestClient(app)

        mock_model = MagicMock()
        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.return_value = True
        mock_recognizer.Result.return_value = '{"text": "test"}'

        mock_get_model.return_value = mock_model
        mock_recognizer_class.return_value = mock_recognizer

        with client.websocket_connect("/api/speech-to-text") as websocket:
            websocket.send_bytes(b'\x00\x01')
            response = websocket.receive_text()
            assert response == "test"
            mock_recognizer.AcceptWaveform.assert_called_once()
            mock_recognizer.Result.assert_called_once()

def test_websocket_partial_result():
    with patch("API.get_model") as mock_get_model, patch("API.KaldiRecognizer") as mock_recognizer_class:
        from API import app
        client = TestClient(app)

        mock_model = MagicMock()
        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.return_value = False
        mock_recognizer.PartialResult.return_value = '{"partial": "hello"}'

        mock_get_model.return_value = mock_model
        mock_recognizer_class.return_value = mock_recognizer

        with client.websocket_connect("/api/speech-to-text") as websocket:
            websocket.send_bytes(b'\x01\x02')
            response = websocket.receive_text()
            assert response == "hello"
            mock_recognizer.PartialResult.assert_called_once()

def test_websocket_error_handling():
    with patch("API.get_model") as mock_get_model, patch("API.KaldiRecognizer") as mock_recognizer_class:
        from API import app
        client = TestClient(app)

        mock_model = MagicMock()
        mock_recognizer = MagicMock()
        mock_recognizer.AcceptWaveform.side_effect = Exception("Mocked error")

        mock_get_model.return_value = mock_model
        mock_recognizer_class.return_value = mock_recognizer

        with client.websocket_connect("/api/speech-to-text") as websocket:
            websocket.send_bytes(b'\x01\x02')
           
