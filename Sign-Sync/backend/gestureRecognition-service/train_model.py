import load
import prepare
import LSTM_model
import time
from tqdm import tqdm
import pickle
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    start = time.time()
    print("Started training script")

    train_set, test_set = load.load_dataset("data/WLASL_v0.3.json", "data/videos")

    x_train, y_train, label_enc = prepare.prepare_data_parallel(train_set, maxlen=50)
    # Prepare validation data
  


    print("Sample input shape:", x_train.shape)
    y_train = np.array(y_train).flatten()
    print("Sample label distribution:", np.bincount(y_train))
    print("y_train shape:", y_train.shape)
    print("Unique labels:", np.unique(y_train))

    print("First input sequence:", x_train[0])

    label_counts = pd.Series(y_train).value_counts()
    print(label_counts.describe())  # mean, min, etc.

    # 🔧 New: Filter rare classes (less than 4 samples)
    min_samples = 4
    keep_classes = label_counts[label_counts >= min_samples].index
    mask = np.isin(y_train, keep_classes)
    x_train = x_train[mask]
    y_train = y_train[mask]

    # Reindex labels to ensure they are contiguous
    label_map = {old: new for new, old in enumerate(sorted(keep_classes))}
    y_train = np.array([label_map[y] for y in y_train])

    # Update num_classes for model building
    num_classes = len(label_map)

    x_val, y_val, _ = prepare.prepare_data_parallel(test_set, maxlen=50, label_encoder=label_enc)
    y_val = np.array(y_val).flatten()

    # Filter and remap y_val to only include classes seen in training
    val_mask = np.isin(y_val, keep_classes)
    x_val = x_val[val_mask]
    y_val = y_val[val_mask]
    y_val = np.array([label_map[y] for y in y_val])

    model = LSTM_model.build_model((50, x_train.shape[2]), num_classes)
    print("[INFO] Started training")

    if not os.path.exists("model"):
        os.makedirs("model")

    checkpoint = ModelCheckpoint("model/gesture_model.keras", save_best_only=True, monitor="val_loss", mode="min")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(x_val, y_val), 
        callbacks=[checkpoint],
        verbose=1
    )

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("[INFO] Ended training")

    if not os.path.exists("labels"):
        os.makedirs("labels")

    with open("labels/label_encoder.pkl", "wb") as f:
        pickle.dump(label_enc, f)



# 🔒 This is required for Windows multiprocessing
if __name__ == "__main__":
    main()
