import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import keypoints  # you already use this

def _process_video(item, cache_dir, maxlen):
    try:
        video_path = item["video_path"]
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        cache_path = os.path.join(cache_dir, f"{video_id}.npy")

        if os.path.exists(cache_path):
            return cache_path, item["gloss"]

        seq = keypoints.extract_hand_keypoints(video_path, item["bbox"], max_frames=maxlen)
        if seq.shape[0] == 0:
            return None

        np.save(cache_path, seq)
        return cache_path, item["gloss"]
    except Exception as e:
        print(f"[ERROR] Failed to process {video_path}: {e}")
        return None


def prepare_data_parallel(dataset, cache_dir="cached", maxlen=50, label_encoder=None):
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[INFO] Processing {len(dataset)} videos using {cpu_count()} CPU cores...")

    args = [(item, cache_dir, maxlen) for item in dataset]

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.starmap(_process_video, args), total=len(dataset), desc="Extracting/Caching"))

    sequences = []
    labels = []

    for result in results:
        if result is None:
            continue
        path, gloss = result
        try:
            sequences.append(np.load(path))
            labels.append(gloss)
        except Exception as e:
            print(f"[WARNING] Skipping cached file {path}: {e}")

    padded = pad_sequences(sequences, maxlen=maxlen, dtype='float32', padding='post')

    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
    else:
        y = label_encoder.transform(labels)

    return padded, y, label_encoder
