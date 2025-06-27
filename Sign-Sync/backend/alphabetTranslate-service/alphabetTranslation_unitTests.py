import pytest
import numpy as np
import torch
from translator_api import normalize_keypoints, KeypointClassifier, Landmark, index_to_label

# Unit Test for the normalize_keypoints function
def test_normalize_keypoints():
    # Create mock landmarks (21 landmarks with random values)
    mock_landmarks = [Landmark(x=float(i), y=float(i), z=float(i)) for i in range(21)]
    
    # Call the normalize function
    normalized_keypoints = normalize_keypoints(mock_landmarks)
    
    # Check if the output shape is correct (it should be a flattened array)
    assert normalized_keypoints.shape == (63,)  # 21 landmarks * 3 = 63
    assert isinstance(normalized_keypoints, np.ndarray)
    
    # Further, check if the normalization logic is working correctly.
    # This will depend on the actual logic of normalize_keypoints.
    assert np.allclose(normalized_keypoints[0], 0)  # Ensure the wrist is at the origin



# Unit Test for the KeypointClassifier model
def test_keypoint_classifier():
    model = KeypointClassifier()
    model.eval()  # Set model to evaluation mode

    # Mock normalized input (21 landmarks * 3 values)
    mock_input = torch.randn(1, 63)  # (batch_size, 63)
    
    # Run model prediction
    with torch.no_grad():
        output = model(mock_input)
    
    # Check the output shape (should be of size len(index_to_label))
    assert output.shape == (1, len(index_to_label))
    
    # Check if output is a tensor
    assert isinstance(output, torch.Tensor)
    
    # Get the predicted index and corresponding label
    pred_idx = output.argmax(1).item()
    
    # Check that the predicted index maps to a valid label in index_to_label
    assert pred_idx in index_to_label, f"Predicted index {pred_idx} is not in the index_to_label map"
    
    # Ensure the label is a single character (assuming labels are single alphabetic characters)
    label = index_to_label[pred_idx]
    assert isinstance(label, str)
    assert len(label) == 1, f"Label {label} is not a single character"

