# train_signsync_bigru.py
import os, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score

DATA_DIR = Path("preprocessed")
USE_AUG  = False  # True -> use train_aug2x.npz; False -> use train.npz
TRAIN_PATH = DATA_DIR / ("train_aug2x.npz" if USE_AUG else "train.npz")
VAL_PATH   = DATA_DIR / "val.npz"
TEST_PATH  = DATA_DIR / "test.npz"
LABEL_MAP  = DATA_DIR / "label_map.json"

# Training hyperparams
BATCH_SIZE   = 64
LR           = 3e-4
EPOCHS       = 40
PATIENCE     = 8
WEIGHT_DECAY = 1e-5
GRU_HIDDEN   = 192
GRU_LAYERS   = 2
DROPOUT      = 0.25
USE_MASK     = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def load_npz_dataset(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)   # [N,T,J,C]
    y = d["y"].astype(np.int64)     # [N]
    return X, y

def to_loaders(train_path, val_path, batch_size):
    Xtr, ytr = load_npz_dataset(train_path)
    Xva, yva = load_npz_dataset(val_path)

    # optionally drop the mask channel at input time
    if USE_MASK:
        # keep all C channels (XYZ + mask)
        pass
    else:
        Xtr = Xtr[..., :3]
        Xva = Xva[..., :3]

    # reshape to [N, T, F], where F = J*C
    Ntr, T, J, C = Xtr.shape
    Nva = Xva.shape[0]
    F = J * C
    Xtr = Xtr.reshape(Ntr, T, F)
    Xva = Xva.reshape(Nva, T, F)

    # torch tensors
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva)
    yva_t = torch.from_numpy(yva)

    train_ds = TensorDataset(Xtr_t, ytr_t)
    val_ds   = TensorDataset(Xva_t, yva_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    meta = {"T": T, "J": J, "C": C, "F": F, "num_classes": int(ytr.max() + 1)}
    return train_dl, val_dl, meta, ytr

class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden=192, layers=2, num_classes=50, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2*hidden),
            nn.Dropout(dropout),
            nn.Linear(2*hidden, num_classes),
        )

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.gru(x)        # [B, T, 2H]
        h = out[:, -1, :]           # last time step
        logits = self.head(h)       # [B, K]
        return logits

def class_weights_from_labels(y):
    # inverse freq weighting
    counts = np.bincount(y, minlength=int(y.max()+1)).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)

def train_one_epoch(model, dl, criterion, opt):
    model.train()
    total_loss = 0.0
    total_n = 0
    for X, y in dl:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        opt.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
    return total_loss / max(1, total_n)

@torch.no_grad()
def evaluate(model, dl, criterion):
    model.eval()
    total_loss = 0.0
    total_n = 0
    preds, gts = [], []
    for X, y in dl:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(X)
        loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
        gts.append(y.cpu().numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    gts = np.concatenate(gts) if gts else np.array([])
    acc = accuracy_score(gts, preds) if preds.size else 0.0
    f1  = f1_score(gts, preds, average="macro") if preds.size else 0.0
    return total_loss / max(1, total_n), acc, f1, preds, gts

def main():
    print(f"Device: {DEVICE}")
    train_dl, val_dl, meta, ytr = to_loaders(TRAIN_PATH, VAL_PATH, BATCH_SIZE)
    num_classes = meta["num_classes"]
    input_dim = meta["F"]

    print(f"Input: T={meta['T']}  J={meta['J']}  C={meta['C']}  F={input_dim}  classes={num_classes}")

    model = BiGRUClassifier(
        input_dim=input_dim,
        hidden=GRU_HIDDEN,
        layers=GRU_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(DEVICE)

    # weighted CE for class imbalance
    weights = class_weights_from_labels(ytr)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_epoch = -1
    ckpt_path = CKPT_DIR / "bigru_best.pt"
    patience = PATIENCE
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_dl, criterion, opt)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_dl, criterion)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  val_f1={va_f1:.4f}  ({dt:.1f}s)")

        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "meta": meta}, ckpt_path)
            patience = PATIENCE
            print(f"  ↳ new best, saved to {ckpt_path}")
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val_loss={best_val:.4f})")
                break

    # ---------- Final test ----------
    # reload best
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    # test loader (reuse preprocessing)
    Xte, yte = load_npz_dataset(TEST_PATH)
    if not USE_MASK:
        Xte = Xte[..., :3]
    Nte, T, J, C = Xte.shape
    Xte = Xte.reshape(Nte, T, J*C)
    test_dl = DataLoader(TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)), batch_size=BATCH_SIZE, shuffle=False)

    test_loss, test_acc, test_f1, preds, gts = evaluate(model, test_dl, criterion)
    print("\n=== TEST RESULTS ===")
    print(f"loss={test_loss:.4f}  acc={test_acc:.4f}  macro-F1={test_f1:.4f}")

    # human-readable label names
    with open(LABEL_MAP, "r") as f:
        label_map = json.load(f)
    id2lab = {v: k for k, v in label_map.items()}
    target_names = [id2lab[i] for i in range(len(id2lab))]
    print("\nPer-class report (macro avg at bottom):")
    print(classification_report(gts, preds, target_names=target_names, digits=3))

if __name__ == "__main__":
    main()
