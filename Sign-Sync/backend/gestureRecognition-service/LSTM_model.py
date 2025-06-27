# === LSTM_model.py ===
# Replaced with HMM model manager

from hmmlearn import hmm
from collections import defaultdict
import numpy as np

class HMMGlossClassifier:
    def __init__(self, n_components=2, covariance_type='diag', n_iter=300, verbose=False):
        self.models = {}
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter

    def fit(self, x_train, y_train):
        label_to_sequences = defaultdict(list)

        for seq, label in zip(x_train, y_train):
            label_to_sequences[label].append(seq)

        for label, sequences in label_to_sequences.items():
            X = np.vstack(sequences)
            lengths = [len(seq) for seq in sequences]

            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                tol=1e-2,
                verbose=self.verbose,
                n_iter=self.n_iter,
                random_state=42
            )
            model.fit(X, lengths)
            self.models[label] = model

    def predict(self, x_test):
        predictions = []
        for seq in x_test:
            best_score = float('-inf')
            best_label = None
            for label, model in self.models.items():
                try:
                    score = model.score(seq)
                    if score > best_score:
                        best_score = score
                        best_label = label
                except:
                    continue
            predictions.append(best_label)
        return np.array(predictions)

    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return np.mean(y_pred == y_test)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.models, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            self.models = pickle.load(f)


def build_model():
    return HMMGlossClassifier()
