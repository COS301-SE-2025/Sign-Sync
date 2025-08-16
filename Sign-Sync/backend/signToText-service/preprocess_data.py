import json, csv, math, sys
from pathlib import Path
import numpy as np

SPLITS_DIR = Path("splits")
DATA_DIR   = Path(".")
OUT_DIR    = Path("preprocessed")
TARGET_FPS = 25
RECORD_SEC = 3.0
T          = int(TARGET_FPS * RECORD_SEC)
SEED       = 1337

POSE_KEEP = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
USE_ROTATION_NORM = True
ADD_MASK_CHANNEL  = True

rng = np.random.default_rng(SEED)

def load_csv(fname):
    rows = []
    with open(fname, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["path"], row["label"], row["signer"]))
    return rows

def load_npz_clip(path):
    data = np.load(path)
    # Expect keys: pose_xyzw [t,33,4], left_xyz [t,21,3], right_xyz [t,21,3]
    pose_xyzw = data.get("pose_xyzw", None)   # may be None
    left_xyz  = data.get("left_xyz", None)
    right_xyz = data.get("right_xyz", None)
    return pose_xyzw, left_xyz, right_xyz

def linear_resample(arr, target_len):
    """Resample a [T0, ...] array to [target_len, ...] via linear interpolation on the time axis."""
    T0 = arr.shape[0]
    if T0 == target_len:
        return arr
    # build original and target indices in [0, 1]
    x0 = np.linspace(0.0, 1.0, T0, dtype=np.float32)
    x1 = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    # vectorized interpolation along time for the last dims flattened
    flat = arr.reshape(T0, -1)
    out_flat = np.empty((target_len, flat.shape[1]), dtype=arr.dtype)
    for d in range(flat.shape[1]):
        out_flat[:, d] = np.interp(x1, x0, flat[:, d])
    return out_flat.reshape((target_len,) + arr.shape[1:])

def torso_center_and_scale(pose_xyzw_frame):
    if pose_xyzw_frame is None:
        return np.zeros(3, dtype=np.float32), 1.0

    def get(idx):
        return pose_xyzw_frame[idx, :3]  # x,y,z (normalized image coords)

    # indices
    L_SH, R_SH = 11, 12
    L_HP, R_HP = 23, 24

    # presence check: if a landmark row is all zeros, treat as missing
    def present(idx):
        return not np.allclose(pose_xyzw_frame[idx, :3], 0.0, atol=1e-8)

    have_sh = present(L_SH) and present(R_SH)
    have_hp = present(L_HP) and present(R_HP)

    if have_sh and have_hp:
        center = 0.25*(get(L_SH)+get(R_SH)+get(L_HP)+get(R_HP))
    elif have_sh:
        center = 0.5*(get(L_SH)+get(R_SH))
    elif have_hp:
        center = 0.5*(get(L_HP)+get(R_HP))
    else:
        center = np.zeros(3, dtype=np.float32)

    scale = 1.0
    if have_sh:
        scale = np.linalg.norm(get(L_SH) - get(R_SH)) + 1e-6
    elif have_hp:
        scale = np.linalg.norm(get(L_HP) - get(R_HP)) + 1e-6

    # clamp absurdly small scale to avoid blow-ups
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0
    return center.astype(np.float32), float(scale)

def zrot_matrix(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

def rotation_theta_from_shoulders(pose_xyzw_frame):
    """Angle to rotate so that shoulders lie horizontally (right shoulder to +x)."""
    L_SH, R_SH = 11, 12
    if pose_xyzw_frame is None:
        return 0.0
    LS = pose_xyzw_frame[L_SH, :3]
    RS = pose_xyzw_frame[R_SH, :3]
    if np.allclose(LS, 0.0) or np.allclose(RS, 0.0):
        return 0.0
    v = RS - LS  # vector from left to right shoulder
    # angle between v (projected to xy) and +x axis
    theta = math.atan2(v[1], v[0])  # radians
    return -theta  # rotate by -theta to align with x-axis

def frame_mask(arr_xyz):
    """Return [J] mask (1 if landmark is nonzero anywhere in 3D, else 0)."""
    return (np.linalg.norm(arr_xyz, axis=-1) > 0).astype(np.float32)

def process_clip(path):
    pose_xyzw, left_xyz, right_xyz = load_npz_clip(path)

    # Basic presence checks
    if pose_xyzw is None or left_xyz is None or right_xyz is None:
        raise ValueError(f"Missing arrays in {path}. Expected pose_xyzw, left_xyz, right_xyz.")

    # Keep selected pose joints, drop visibility (keep XYZ)
    pose_xyz = pose_xyzw[:, POSE_KEEP, :3]  # [t, P, 3]
    # Hands as-is
    lh_xyz   = left_xyz   # [t, 21, 3]
    rh_xyz   = right_xyz  # [t, 21, 3]

    # Resample each stream to T
    pose_xyz = linear_resample(pose_xyz, T)
    lh_xyz   = linear_resample(lh_xyz,   T)
    rh_xyz   = linear_resample(rh_xyz,   T)

    full_pose_xyzw = linear_resample(pose_xyzw, T)  # [T,33,4]

    out_seq = []
    out_mask = []

    for t in range(T):
        center, scale = torso_center_and_scale(full_pose_xyzw[t])

        theta = rotation_theta_from_shoulders(full_pose_xyzw[t]) if USE_ROTATION_NORM else 0.0
        Rz = zrot_matrix(theta)

        # concat joints for this frame: [P+21+21, 3]
        frame_xyz = np.concatenate([pose_xyz[t], lh_xyz[t], rh_xyz[t]], axis=0).astype(np.float32)

        # center & scale
        frame_xyz = (frame_xyz - center[None, :]) / scale

        # rotate around Z (affects x,y; z remains)
        if USE_ROTATION_NORM:
            frame_xyz = frame_xyz @ Rz.T

        # mask (before zeroing anything)
        m = frame_mask(frame_xyz)

        out_seq.append(frame_xyz)
        out_mask.append(m)

    X = np.stack(out_seq, axis=0)               # [T, J, 3]
    M = np.stack(out_mask, axis=0)              # [T, J]

    if ADD_MASK_CHANNEL:
        X = np.concatenate([X, M[..., None]], axis=-1)  # [T, J, 4]

    return X  # [T, J, C]

def build_label_map(labels):
    uniq = sorted(set(labels))
    return {lab: i for i, lab in enumerate(uniq)}

def compute_mean_std(train_X, n_xyz_channels=3):
    """
    train_X: [N, T, J, C]
    Compute mean/std over XYZ channels only (exclude mask).
    Returns mean, std shaped [1,1,J,3] for broadcasting.
    """
    xyz = train_X[..., :n_xyz_channels]  # [N,T,J,3]
    mean = xyz.mean(axis=(0,1), keepdims=True)      # [1,1,J,3]
    std  = xyz.std(axis=(0,1), keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

def apply_norm(X, mean, std, n_xyz_channels=3):
    Y = X.copy()
    Y[..., :n_xyz_channels] = (Y[..., :n_xyz_channels] - mean) / std
    return Y
