import os, cv2, time
import numpy as np
import mediapipe as mp
from pathlib import Path

# -------- CONFIG --------
OUTPUT_BASE    = "dataset/raw"
WORD_LABEL     = "want"
PREP_SECONDS   = 2
RECORD_SECONDS = 3.0
CAMERA_INDEX   = 0
FPS_TARGET     = 25
FRAME_SIZE     = (1280, 720)
START_INDEX    = 1
SIGNER_TAG     = "_mike"
# ------------------------

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

word_dir = Path(OUTPUT_BASE) / WORD_LABEL
word_dir.mkdir(parents=True, exist_ok=True)

def next_existing_index(d: Path) -> int:
    ids = []
    for p in d.glob("sample_*.npz"):
        try:
            stem = p.stem 
            num = stem.split("_")[1]
            ids.append(int(num))
        except:
            pass
    return (max(ids)+1) if ids else 1

def next_available_at_or_after(d: Path, start: int) -> int:
    i = max(1, int(start))
    while True:
        pattern_main = d / f"sample_{i:03d}.npz"
        pattern_tag  = d / f"sample_{i:03d}{SIGNER_TAG}.npz"
        if not pattern_main.exists() and not pattern_tag.exists():
            return i
        i += 1

def pose_to_array(pose_lm, include_visibility=True):
    if not pose_lm:  # 33 points
        return np.zeros((33, 4 if include_visibility else 3), dtype=np.float32)
    arr = []
    for lm in pose_lm.landmark:
        arr.append([lm.x, lm.y, lm.z, lm.visibility] if include_visibility else [lm.x, lm.y, lm.z])
    return np.array(arr, dtype=np.float32)

def hand_to_array(hand_lm):
    if not hand_lm:  # 21 points
        return np.zeros((21, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32)

def draw_countdown(img, secs_left):
    txt = str(int(np.ceil(secs_left)))
    cv2.putText(img, txt, (img.shape[1]//2-20, img.shape[0]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255), 6, cv2.LINE_AA)

def draw_progress(img, elapsed, total):
    w = img.shape[1] - 40
    x0, y0 = 20, 50
    cv2.rectangle(img, (x0, y0), (x0+w, y0+20), (200,200,200), 2)
    fill = int(w * min(elapsed/total, 1.0))
    cv2.rectangle(img, (x0, y0), (x0+fill, y0+20), (0,255,0), -1)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

print("Press 's' to start (countdown → auto-record), 'q' to quit.")

IDLE, PREP, RECORD = 0, 1, 2
state = IDLE
prep_start = None
rec_start = None

if START_INDEX is None:
    sample_id = next_existing_index(word_dir)
else:
    sample_id = next_available_at_or_after(word_dir, START_INDEX)

pose_buf   = []
left_buf   = []
right_buf  = []
