import load
import prepare
import LSTM_model
import time
from tqdm import tqdm
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


def main():
    start = time.time()
    print("Started training script")

    train_set, test_set = load.load_dataset("data/WLASL_v0.3.json", "data/videos")

    x_train, y_train, label_enc = prepare.prepare_data_parallel(train_set, maxlen=50)
    x_val, y_val, _ = prepare.prepare_data_parallel(test_set, maxlen=50, label_encoder=label_enc)

    model = LSTM_model.build_model((50, x_train.shape[2]), len(label_enc.classes_))
    print("[INFO] Started training")

    if not os.path.exists("model"):
        os.makedirs("model")

    checkpoint = ModelCheckpoint("model/gesture_model.keras", save_best_only=True, monitor="val_loss", mode="min")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint],
        verbose=1
    )

    print("[INFO] Ended training")

    if not os.path.exists("labels"):
        os.makedirs("labels")

    with open("labels/label_encoder.pkl", "wb") as f:
        pickle.dump(label_enc, f)


# 🔒 This is required for Windows multiprocessing
if __name__ == "__main__":
    main()
