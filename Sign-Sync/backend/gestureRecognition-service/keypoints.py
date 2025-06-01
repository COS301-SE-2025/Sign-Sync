import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

def normalize_keypoints(keypoints):
    if keypoints.size == 0:
        return keypoints
    min_vals = np.min(keypoints, axis=0)
    max_vals = np.max(keypoints, axis=0)
    range_vals = max_vals - min_vals + 1e-5
    return ((keypoints - min_vals) / range_vals)

def extract_hand_keypoints(video_path, bbox=None, max_frames=50):
    capture = cv2.VideoCapture(video_path)
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    sequence = []

    while capture.isOpened() and len(sequence) < max_frames:
        ret, frame = capture.read()
        if not ret:
            break

        if bbox:
            x1, y1, x2, y2 = bbox
            frame = frame[y1:y2, x1:x2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Create empty keypoints for left and right hands
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                if label == 'Left':
                    left_hand = arr
                else:
                    right_hand = arr

        combined = np.vstack((left_hand, right_hand))  # shape (42, 3)
        keypoints = normalize_keypoints(combined).flatten()
        sequence.append(keypoints)

    capture.release()
    hands.close()
    return np.array(sequence)
