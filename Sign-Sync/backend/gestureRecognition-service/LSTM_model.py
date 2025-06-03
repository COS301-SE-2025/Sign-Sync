from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.regularizers import l2

def build_model(input_shape, num_classes):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.00005)),
        LSTM(64, kernel_regularizer=l2(0.00005)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.00005)),
        Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.00005))
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
