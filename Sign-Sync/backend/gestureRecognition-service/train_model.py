
import load
import prepare
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import augment
from tensorflow.keras import Sequential, layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Masking, LSTM, Dense, SpatialDropout1D, BatchNormalization
from tensorflow.keras.regularizers import l2
from collections import defaultdict
from collections import Counter
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


def main():
    print("Started training script")

    train_set, test_set = load.load_dataset("data/WLASL_v0.3.json", "data/videos")

    train_glosses = {e['gloss'] for e in train_set}
    test_glosses  = {e['gloss'] for e in test_set}
    print("Only in test:", test_glosses - train_glosses)

    top_n = 200
    all_glosses = [item['gloss'] for item in train_set]
    gloss_freq = Counter(all_glosses)
    top_glosses = set(g for g, _ in gloss_freq.most_common(top_n))


    train_set = [item for item in train_set if item['gloss'] in top_glosses]
    test_set = [item for item in test_set if item['gloss'] in top_glosses]

    train_glosses = set(item['gloss'] for item in train_set)
    test_set = [item for item in test_set if item['gloss'] in train_glosses]

    # keypoints are extracted, normalized and then padded to 50 frames
    x_train, y_train, label_enc = prepare.prepare_data_parallel(train_set, maxlen=50)
    print("Classes: ", label_enc.classes_)
    
    x_val, y_val, _ = prepare.prepare_data_parallel(test_set, False, maxlen=50, label_encoder=label_enc)
    
    train_classes = set(np.unique(y_train))
    val_classes = set(np.unique(y_val))
    shared_classes = sorted(train_classes & val_classes)

    if len(shared_classes) == 0:
        exit("No shared classes between train and validation sets after augmentation.")

    print(f"Keeping only {len(shared_classes)} shared classes.")

    label_map = {old: new for new, old in enumerate(shared_classes)}

    inv_label_map = {v: k for k, v in label_map.items()}

    index_to_gloss = {str(new_idx): label_enc.inverse_transform([old_idx])[0] for new_idx, old_idx in inv_label_map.items()}

    with open("label_map.json", "w") as f:
        json.dump(index_to_gloss, f)


    def filter_and_remap(x, y):
        mask = np.isin(y, shared_classes)
        x = x[mask]
        y = np.array([label_map[yi] for yi in y[mask]])
        return x, y

    x_train, y_train = filter_and_remap(x_train, y_train)
    x_val, y_val = filter_and_remap(x_val, y_val)

    y_train = np.array(y_train).flatten()
    y_val = np.array(y_val).flatten()


    class_to_samples = defaultdict(list)
    for xi, yi in zip(x_train, y_train):
        class_to_samples[yi].append(np.array(xi, dtype=np.float32))

    augmentations_per_sample = 10
    target_total_per_class = 2000

    new_x = []
    new_y = []

    for yi, samples in class_to_samples.items():
        augmented = []

        for sample in samples:
            augmented.append(sample)

            for _ in range(augmentations_per_sample):
                aug = augment.augment_keypoints_3d(
                    sample,
                    seed=None,
                    global_dropout=False,
                    intensity=np.random.uniform(0.8, 1.2)
                )
                augmented.append(aug)

        num_fill = max(0, target_total_per_class - len(augmented))
        for i in range(num_fill):
            sample = samples[i % len(samples)]
            aug = augment.augment_keypoints_3d(
                sample,
                seed=None,
                global_dropout=False,
                intensity=np.random.uniform(0.8, 1.2)
            )
            augmented.append(aug)

        new_x.extend(augmented[:target_total_per_class])
        new_y.extend([yi] * target_total_per_class)


    x_train = np.stack(new_x)
    y_train = np.array(new_y)


    class_to_val_samples = defaultdict(list)
    for xi, yi in zip(x_val, y_val):
        class_to_val_samples[yi].append(np.array(xi, dtype=np.float32))

    augment_val_per_sample = 7
    target_val_per_class = 500

    new_val_x = []
    new_val_y = []

    for yi, samples in class_to_val_samples.items():
        augmented = []

        for sample in samples:
            augmented.append(sample)

            for _ in range(augment_val_per_sample):
                aug = augment.augment_keypoints_3d(
                    sample,
                    seed=None,
                    global_dropout=False,
                    intensity=np.random.uniform(0.8, 1.2)
                )
                augmented.append(aug)

        num_fill = max(0, target_val_per_class - len(augmented))
        for i in range(num_fill):
            sample = samples[i % len(samples)]
            aug = augment.augment_keypoints_3d(
                sample,
                seed=None,
                global_dropout=False,
                intensity=np.random.uniform(0.8, 1.2)
            )
            augmented.append(aug)

        new_val_x.extend(augmented[:target_val_per_class])
        new_val_y.extend([yi] * target_val_per_class)


    x_val = np.stack(new_val_x)
    y_val = np.array(new_val_y)


    train_classes = set(np.unique(y_train))
    val_classes = set(np.unique(y_val))
    if train_classes != val_classes:
        print("Mismatch after augmentation.")
        print("Classes in training but missing in validation:", train_classes - val_classes)
        print("Classes in validation but missing in training:", val_classes - train_classes)
        exit("Exiting due to class label mismatch after augmentation.")

    print("Train set class distribution:")
    print(pd.Series(y_train).value_counts())

    print("\nValidation set class distribution:")
    print(pd.Series(y_val).value_counts())


    num_classes = len(label_map)
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)

    from sklearn.utils import class_weight

    weights = class_weight.compute_class_weight('balanced',
                                                classes=np.unique(y_train),
                                                y=y_train)
    class_weights = dict(enumerate(weights))

    model = Sequential([

    layers.Masking(mask_value=0.0,
                   input_shape=(x_train.shape[1], x_train.shape[2])),

    layers.Conv1D(
        filters=32, kernel_size=5, padding='causal', dilation_rate=1,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    ),
    layers.BatchNormalization(),
    layers.SpatialDropout1D(0.3),

    layers.Conv1D(
        filters=32, kernel_size=5, padding='causal', dilation_rate=2,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    ),
    layers.BatchNormalization(),
    layers.SpatialDropout1D(0.3),

    layers.Conv1D(
        filters=32, kernel_size=5, padding='causal', dilation_rate=4,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    ),
    layers.BatchNormalization(),
    layers.SpatialDropout1D(0.3),

    layers.LSTM(
        units=64,
        return_sequences=False,
        dropout=0.3,
        recurrent_dropout=0.4
    ),

    layers.Dense(
        units=64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    ),
    layers.Dropout(0.6),

    layers.Dense(num_classes, activation='softmax')
])

    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', factor=0.7, patience=2, verbose=1
    )

    checkpoint = ModelCheckpoint(
    filepath="better_model_val_acccccc.keras", 
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
    )

    callbacks = [early_stop, lr_scheduler, checkpoint]


    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    model.fit(
        x_train, y_train_cat,
        validation_data=(x_val, y_val_cat),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights
    )

    model.save("save_model.keras")

    train_loss, train_acc = model.evaluate(
    x_train, y_train_cat,
    batch_size=32,
    verbose=1
    )
    print(f"Train accuracy (no dropout/bn training): {train_acc:.3f}")

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_enc, f)


    print("--------------------------------------")
    print("X_Val: ", x_val[0])
    y_pred_cat = model.predict(x_val)
    y_pred = np.argmax(y_pred_cat, axis=1)

    print("Classification Report:\n")
    print(classification_report(y_val, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_val, y_pred))

if __name__ == "__main__":
    main()
