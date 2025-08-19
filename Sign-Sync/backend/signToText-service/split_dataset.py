import csv, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

DATA_ROOT = Path("dataset/raw")
OUT_DIR   = Path("splits")
SEED      = 42

FNAME_RE = re.compile(r"^sample_(?P<id>\d{3})(?:_(?P<signer>[^.]+))?\.npz$")

rng = np.random.default_rng(SEED)

def find_samples():
    rows = []
    for word_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        label = word_dir.name
        for p in sorted(word_dir.glob("sample_*.npz")):
            m = FNAME_RE.match(p.name)
            if not m:
                print(f"[WARN] Skipping unexpected filename: {p}", file=sys.stderr)
                continue
            signer = m.group("signer") or "unknown"
            rows.append((str(p), label, signer))
    return rows

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = find_samples()
    if not rows:
        print(f"[ERR] No samples found under {DATA_ROOT}")
        sys.exit(1)

    # Group by (word, signer)
    buckets = defaultdict(list)
    for path, label, signer in rows:
        buckets[(label, signer)].append((path, label, signer))

    train, val, test = [], [], []

    for (label, signer), items in sorted(buckets.items()):
        if len(items) != 30:
            print(f"[ERR] {label}/{signer} has {len(items)} samples, expected 30.")
            sys.exit(1)
        idx = rng.permutation(len(items))
        items = [items[i] for i in idx]

        train.extend(items[:20])
        val.extend(items[20:25])
        test.extend(items[25:])

    def write_csv(fname, data):
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "signer"])
            w.writerows(data)

    write_csv(OUT_DIR / "train.csv", train)
    write_csv(OUT_DIR / "val.csv", val)
    write_csv(OUT_DIR / "test.csv", test)

    print(f"✅ Done! CSVs saved in '{OUT_DIR}'")
    print(f"Train: {len(train)} clips | Val: {len(val)} | Test: {len(test)}")

if __name__ == "__main__":
    main()
