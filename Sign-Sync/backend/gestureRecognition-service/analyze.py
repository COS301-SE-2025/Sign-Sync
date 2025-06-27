import pickle

with open("label_encoder.pkl", "rb") as f:
    label_enc = pickle.load(f)

indices = [16, 24, 30, 33, 3, 4, 5, 20, 25, 26, 31, 34, 0, 2, 7, 9, 12, 27, 22, 10,
           18, 1, 19, 8, 32, 6, 13, 15, 17, 21, 23, 28, 29, 35, 36, 11, 14]

glosses = label_enc.inverse_transform(indices)

for idx, gloss in zip(indices, glosses):
    print(f"Class {idx}: {gloss}")
