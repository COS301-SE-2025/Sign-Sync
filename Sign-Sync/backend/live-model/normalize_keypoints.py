import os
import json
import numpy as np

def normalize_keypoints(keypoints):
    keypoints = np.array(keypoints).reshape((21, 3))
    wrist = keypoints[0]
    centered = keypoints - wrist
    scale = np.linalg.norm(centered[9])  # use middle finger MCP as scale reference
    normalized = centered / scale if scale != 0 else centered
    return normalized.flatten().tolist()

for file in os.listdir():
    if file.startswith("keypoints_") and file.endswith(".json"):
        with open(file, "r") as f:
            data = json.load(f)

        normalized_data = []
        for sample in data:
            norm_kps = normalize_keypoints(sample["keypoints"])
            normalized_data.append({
                "label": sample["label"],
                "keypoints": norm_kps
            })

        output_file = f"normalized_{file}"
        with open(output_file, "w") as f:
            json.dump(normalized_data, f, indent=2)

        print(f"Normalized and saved to {output_file}")