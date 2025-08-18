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

with mp_holistic.Holistic(model_complexity=1) as holistic:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        vis   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        mp_draw.draw_landmarks(vis, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(vis, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(vis, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        fname_preview = f"sample_{sample_id:03d}{SIGNER_TAG}.npz"
        if state == IDLE:
            cv2.putText(vis, f"Word: {WORD_LABEL}   Next: {fname_preview}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(vis, "Press 's' to start  (3..2..1 then auto record)",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if key == ord('s'):
                pose_buf, left_buf, right_buf = [], [], []
                prep_start = time.time()
                state = PREP

        elif state == PREP:
            secs_left =  PREP_SECONDS - (time.time() - prep_start)
            if secs_left <= 0:
                rec_start = time.time()
                state = RECORD
                print(f"[{WORD_LABEL}] Recording {fname_preview} ...")
            else:
                draw_countdown(vis, secs_left)
                cv2.putText(vis, "Get into position", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        elif state == RECORD:
            # collect raw landmarks each frame
            pose_arr  = pose_to_array(results.pose_landmarks, include_visibility=True)  # (33,4)
            left_arr  = hand_to_array(results.left_hand_landmarks)                      # (21,3)
            right_arr = hand_to_array(results.right_hand_landmarks)                     # (21,3)

            pose_buf.append(pose_arr)
            left_buf.append(left_arr)
            right_buf.append(right_arr)

            elapsed = time.time() - rec_start
            draw_progress(vis, elapsed, RECORD_SECONDS)
            cv2.putText(vis, "Recording...", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            if elapsed >= RECORD_SECONDS:
                out_path = word_dir / fname_preview
                np.savez_compressed(
                    out_path,
                    pose_xyzw=np.stack(pose_buf, axis=0),   # [T,33,4]
                    left_xyz=np.stack(left_buf, axis=0),    # [T,21,3]
                    right_xyz=np.stack(right_buf, axis=0)   # [T,21,3]
                )
                print(f"Saved {out_path}")

                sample_id = next_available_at_or_after(word_dir, sample_id + 1)
                state = IDLE

        cv2.imshow("Record RAW Word Samples (Countdown + Auto Stop)", vis)

cap.release()
cv2.destroyAllWindows()
