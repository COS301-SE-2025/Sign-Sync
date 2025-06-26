from sklearn.model_selection import train_test_split
import json
import os

def load_dataset(json_path, video_folder, test_size=0.1, min_instances=2):
    with open(json_path, 'r') as file:
        data = json.load(file)

    train_set = []
    test_set = []

    for entry in data:
        gloss = entry['gloss']
        instances = entry['instances']

        if len(instances) < min_instances:
            continue

        train, test = train_test_split(instances, test_size=test_size, random_state=42)

        for instance in train:
            train_set.append({
                "gloss": gloss,
                "video_path": os.path.join(video_folder, instance['video_id'] + ".mp4"),
                "bbox": instance.get('bbox', None)
            })

        for instance in test:
            test_set.append({
                "gloss": gloss,
                "video_path": os.path.join(video_folder, instance['video_id'] + ".mp4"),
                "bbox": instance.get('bbox', None)
            })

    return train_set, test_set
