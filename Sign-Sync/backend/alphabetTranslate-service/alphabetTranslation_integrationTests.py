from fastapi.testclient import TestClient
from translator_api import app

# Create a TestClient instance
client = TestClient(app)

# Integration Test for the /predict API route
def test_predict():
    # Create a mock request payload with 21 landmarks
    mock_payload = {
        "keypoints": [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 1.0, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 1.0, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": 1.0},
            {"x": 1.0, "y": 1.0, "z": 1.0},
            {"x": -1.0, "y": -1.0, "z": -1.0},
            {"x": 0.5, "y": 0.5, "z": 0.5},
            {"x": -0.5, "y": -0.5, "z": -0.5},
            {"x": 0.25, "y": 0.25, "z": 0.25},
            {"x": -0.25, "y": -0.25, "z": -0.25},
            {"x": 0.0, "y": 2.0, "z": 0.0},
            {"x": 1.0, "y": 2.0, "z": 1.0},
            {"x": 0.0, "y": 0.0, "z": 2.0},
            {"x": 1.0, "y": 0.0, "z": 2.0},
            {"x": 2.0, "y": 2.0, "z": 2.0},
            {"x": -1.0, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": -1.0, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": -1.0},
            {"x": 0.0, "y": 1.0, "z": -1.0},
            {"x": 1.0, "y": -1.0, "z": 0.0},
            {"x": -1.0, "y": 1.0, "z": 0.0}
        ]
    }
    
    # Send POST request to /predict endpoint
    response = client.post("/predict", json=mock_payload)
    
    # Check if the response status is 200
    assert response.status_code == 200
    
    # Check if the response contains a "prediction" field
    assert "prediction" in response.json()
    
    # Check if the prediction is a string (since it's an alphabetic letter)
    prediction = response.json()["prediction"]
    assert isinstance(prediction, str)
    assert len(prediction) == 1  # It should be a single character (e.g., 'A', 'B', etc.)

