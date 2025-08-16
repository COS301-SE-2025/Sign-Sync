# augment_signsync.py
import math, sys
from pathlib import Path
import numpy as np

IN_PATH        = Path("preprocessed/train.npz")
OUT_PATH       = Path("preprocessed/train_aug2x.npz")  # originals + augmented
AUG_FACTOR     = 1
SEED           = 2025

POSE_J = 15
LH_J   = 21
RH_J   = 21
J      = POSE_J + LH_J + RH_J

# Channel layout:
#  - channels 0:3 -> XYZ (normalized)
#  - channel 3    -> mask (1 present, 0 missing)
C_XYZ = 3
C     = 4

# Augmentation strengths (tweak if needed)
TIME_WARP_KNOTS      = 4      
TIME_WARP_STRENGTH   = 0.12   
FRAME_DROPOUT_RATIO  = 0.06   
JITTER_STD           = 0.015  
ROT_Z_DEG            = 5.0    
OCCLUSION_SPAN_RATIO = (0.05, 0.15)  
OCCLUSION_PROB       = 0.50   
# --------------------------------------------
rng = np.random.default_rng(SEED)

def load_train():
    if not IN_PATH.exists():
        print(f"[ERR] Missing {IN_PATH}. Run preprocessing first.", file=sys.stderr)
        sys.exit(1)
    data = np.load(IN_PATH, allow_pickle=True)
    X = data["X"]     # [N,T,J,C]
    y = data["y"]     # [N]
    signers = data["signers"]  # [N]
    paths = data["paths"]      # [N]
    assert X.ndim == 4 and X.shape[2] == J and X.shape[3] == C, f"Unexpected X shape: {X.shape}"
    return X.astype(np.float32), y.astype(np.int64), signers, paths

def save_out(X, y, signers, paths):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, X=X.astype(np.float32), y=y, signers=signers, paths=paths)
    print(f"✅ Saved augmented train to {OUT_PATH} | X={X.shape}, y={y.shape}")

# ---------- helpers ----------
def piecewise_linear_warp(T, knots, strength):
    """
    Build a monotonically increasing time map t' = f(t) over [0,1] with small random deviations.
    Returns sample positions in [0,T-1] for resampling.
    """
    # control points in [0,1]
    xs = np.linspace(0.0, 1.0, knots, dtype=np.float32)
    ys = xs + rng.normal(0.0, strength, size=knots).astype(np.float32)
    ys[0], ys[-1] = 0.0, 1.0
    # monotonic fix (cumulative max)
    ys = np.maximum.accumulate(ys)
    if ys[-1] < 1e-6:
        ys[-1] = 1.0
    ys /= ys[-1]

    # Now evaluate inverse map at uniform positions to get sample positions
    u = np.linspace(0.0, 1.0, T, dtype=np.float32)
    # invert via linear search over xs (knots small; acceptable cost)
    t_idx = np.empty_like(u)
    k = 0
    for i, ui in enumerate(u):
        while k < knots-2 and ys[k+1] < ui:
            k += 1
        # interpolate between (xs[k], ys[k]) -> (xs[k+1], ys[k+1])
        y0, y1 = ys[k], ys[k+1]
        x0, x1 = xs[k], xs[k+1]
        if y1 - y0 < 1e-6:
            t_idx[i] = x0
        else:
            a = (ui - y0) / (y1 - y0)
            t_idx[i] = x0 + a * (x1 - x0)
    return t_idx * (T - 1)

def linear_resample_time(X_seq, t_idx):
    """
    X_seq: [T,J,C], t_idx: [T] positions in [0, T-1]
    Returns [T,J,C]
    """
    T = X_seq.shape[0]
    t0 = np.clip(np.floor(t_idx).astype(np.int32), 0, T-1)
    t1 = np.clip(t0 + 1, 0, T-1)
    a  = (t_idx - t0).astype(np.float32)[:, None, None]  # [T,1,1]
    return (1 - a) * X_seq[t0] + a * X_seq[t1]

def random_time_warp(X_seq):
    T = X_seq.shape[0]
    t_idx = piecewise_linear_warp(T, TIME_WARP_KNOTS, TIME_WARP_STRENGTH)
    return linear_resample_time(X_seq, t_idx)

def random_frame_dropout_resample(X_seq):
    """
    Randomly drop ~p frames then resample back to T.
    Properly handles mask channel (averaged through interpolation).
    """
    T = X_seq.shape[0]
    keep = np.ones(T, dtype=bool)
    num_drop = int(round(T * FRAME_DROPOUT_RATIO))
    if num_drop > 0:
        # avoid removing first/last to keep boundaries stable
        idx = np.arange(1, T-1)
        if idx.size > 0:
            drop_idx = rng.choice(idx, size=min(num_drop, idx.size), replace=False)
            keep[drop_idx] = False
    compact = X_seq[keep]  # [T',J,C]
    # resample back to T along time
    t_idx = np.linspace(0.0, compact.shape[0]-1, T, dtype=np.float32)
    return linear_resample_time(compact, t_idx)

def random_jitter_xyz(X_seq):
    noise = rng.normal(0.0, JITTER_STD, size=X_seq[..., :C_XYZ].shape).astype(np.float32)
    out = X_seq.copy()
    out[..., :C_XYZ] += noise
    return out

def random_small_z_rotation(X_seq):
    # Rotate XY by a tiny angle; leave Z as-is; mask unchanged
    theta = math.radians(rng.uniform(-ROT_Z_DEG, ROT_Z_DEG))
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    out = X_seq.copy()
    # [T,J,3] @ [3,3]^T
    xyz = out[..., :C_XYZ]
    out[..., :C_XYZ] = xyz @ R.T
    return out

def apply_hand_occlusion(X_seq):
    """
    Zero-out one hand for a contiguous span; set mask=0 for those joints in that span.
    """
    T = X_seq.shape[0]
    out = X_seq.copy()
    # choose left or right
    hand = "left" if rng.random() < 0.5 else "right"
    if hand == "left":
        j_start = POSE_J
        j_end   = POSE_J + LH_J
    else:
        j_start = POSE_J + LH_J
        j_end   = POSE_J + LH_J + RH_J

    span = int(round(T * rng.uniform(*OCCLUSION_SPAN_RATIO)))
    if span <= 0:
        return out
    t0 = rng.integers(0, max(1, T - span))
    t1 = t0 + span

    # zero xyz and mask=0 for those hand joints in [t0:t1)
    out[t0:t1, j_start:j_end, :C_XYZ] = 0.0
    out[t0:t1, j_start:j_end, 3]      = 0.0
    return out
