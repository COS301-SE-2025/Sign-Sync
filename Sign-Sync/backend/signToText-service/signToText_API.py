import math, json, uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- Config ----------------
CKPT_PATH   = Path("checkpoints/bigru_best.pt")
LABELS_PATH = Path("preprocessed/label_map.json")
NORM_PATH   = Path("preprocessed/normalization_params.npz")

T            = 75
POSE_KEEP    = [0,2,5,7,8,9,10,11,12,13,14,15,16,23,24]
POSE_J, LH_J, RH_J = 15, 21, 21
J = POSE_J + LH_J + RH_J

EMA_ALPHA        = 0.6
PROB_THRESHOLD   = 0.80
HOLD_FRAMES      = 10
MIN_VALID_JOINT_FRAC_PER_FRAME = 0.25
MIN_VALID_JOINT_FRAC_WINDOW    = 0.35

# New: frames required to "release" after a commit before we allow re-commit
RELEASE_FRAMES = 12  # ~1s at 80ms send cadence (tune 8–18)

# --------------- Model -----------------
class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden=192, layers=2, num_classes=50, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(2*hidden),
            nn.Dropout(dropout),
            nn.Linear(2*hidden, num_classes),
        )
    def forward(self, x):  # [B,T,F]
        y, _ = self.gru(x)
        h = y[:, -1, :]
        return self.head(h)

device = "cuda" if torch.cuda.is_available() else "cpu"

with LABELS_PATH.open("r") as f:
    _m = json.load(f)
id2lab: Dict[int, str] = {v: k for k, v in _m.items()}
lab2id: Dict[str, int] = {v: k for k, v in id2lab.items()}
NUM_CLASSES = len(id2lab)

norm = np.load(NORM_PATH, allow_pickle=True)
MEAN = np.asarray(norm["mean"]).squeeze()
STD  = np.asarray(norm["std"]).squeeze()
C_NO_MASK = MEAN.shape[-1]

ckpt = torch.load(CKPT_PATH, map_location=device)
meta = ckpt.get("meta", {})
EXPECTED_F = meta.get("F", J*4)
HIDDEN     = meta.get("hidden", 192)
LAYERS     = meta.get("layers", 2)
DROPOUT    = meta.get("dropout", 0.25)

model = BiGRUClassifier(EXPECTED_F, HIDDEN, LAYERS, NUM_CLASSES, DROPOUT).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# ------------- Utils -------------
def zrot(theta):
    c,s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)

def torso_center_and_scale(P33_xyzw):
    L_SH,R_SH,L_HP,R_HP = 11,12,23,24
    get = lambda i: P33_xyzw[i,:3]
    pres = lambda i: not np.allclose(P33_xyzw[i,:3], 0.0, atol=1e-8)
    have_sh = pres(L_SH) and pres(R_SH)
    have_hp = pres(L_HP) and pres(R_HP)
    if have_sh and have_hp:
        center = 0.25*(get(L_SH)+get(R_SH)+get(L_HP)+get(R_HP))
    elif have_sh:
        center = 0.5*(get(L_SH)+get(R_SH))
    elif have_hp:
        center = 0.5*(get(L_HP)+get(R_HP))
    else:
        center = np.zeros(3, np.float32)
    if have_sh:
        scale = np.linalg.norm(get(L_SH)-get(R_SH)) + 1e-6
    elif have_hp:
        scale = np.linalg.norm(get(L_HP)-get(R_HP)) + 1e-6
    else:
        scale = 1.0
    if not np.isfinite(scale) or scale < 1e-3: scale = 1.0
    return center.astype(np.float32), float(scale)

def theta_from_shoulders(P33_xyzw):
    L_SH,R_SH=11,12
    LS,RS = P33_xyzw[L_SH,:3], P33_xyzw[R_SH,:3]
    if np.allclose(LS,0) or np.allclose(RS,0): return 0.0
    v = RS - LS
    return -math.atan2(v[1], v[0])

def add_velocity(seq_TJC):
    xyz = seq_TJC[...,:3]
    dxyz = np.zeros_like(xyz)
    dxyz[1:] = xyz[1:] - xyz[:-1]
    mask = seq_TJC[...,-1:]
    return np.concatenate([xyz,dxyz,mask], axis=-1)

def resample_time(arr_TJC, target_T=T):
    T0 = arr_TJC.shape[0]
    if T0 == target_T: return arr_TJC
    x0 = np.linspace(0.0,1.0,T0, dtype=np.float32)
    x1 = np.linspace(0.0,1.0,target_T, dtype=np.float32)
    flat = arr_TJC.reshape(T0, -1)
    out = np.empty((target_T, flat.shape[1]), dtype=arr_TJC.dtype)
    for d in range(flat.shape[1]):
        out[:,d] = np.interp(x1, x0, flat[:,d])
    return out.reshape(target_T, *arr_TJC.shape[1:])

def adapt_features(X_TJC):
    T0,J0,C0 = X_TJC.shape
    F0 = J0*C0
    if F0 == EXPECTED_F:
        return X_TJC.reshape(1,T0,EXPECTED_F)
    if C0==7 and EXPECTED_F==J*4:
        X4 = np.concatenate([X_TJC[...,:3], X_TJC[...,-1:]], axis=-1)
        return X4.reshape(1,T0,EXPECTED_F)
    if C0==4 and EXPECTED_F==J*7:
        zeros = np.zeros_like(X_TJC[...,:3])
        X7 = np.concatenate([X_TJC[...,:3], zeros, X_TJC[...,-1:]], axis=-1)
        return X7.reshape(1,T0,EXPECTED_F)
    raise ValueError("Channel mismatch.")

# ------------- State -------------
@dataclass
class SessionState:
    buffer: deque = field(default_factory=lambda: deque(maxlen=T))
    ema_logits: Optional[np.ndarray] = None
    last_top1: Optional[int] = None
    stable_run: int = 0
    last_committed: Optional[int] = None
    sentence: str = ""
    # New: refractory lock to avoid duplicate commits of same word while held
    refractory: bool = False
    release_run: int = 0

sessions: Dict[str, SessionState] = {}

# ------------- API -------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class StartResp(BaseModel):
    session_id: str
    model: str
    expected_F: int
    T: int
    J: int
    channels: str
    labels: List[str]

@app.post("/v1/session/start", response_model=StartResp)
def start_session():
    sid = uuid.uuid4().hex
    sessions[sid] = SessionState()
    channels = "xyz+mask" if EXPECTED_F == J*4 else "xyz+dxyz+mask"
    return StartResp(
        session_id=sid, model="bigru", expected_F=EXPECTED_F, T=T, J=J,
        channels=channels, labels=[id2lab[i] for i in range(NUM_CLASSES)]
    )

@app.post("/v1/session/stop")
def stop_session(data: Dict[str,str]):
    sid = data.get("session_id")
    sessions.pop(sid, None)
    return {"ok": True}

@app.websocket("/v1/stream/{session_id}")
async def ws_stream(ws: WebSocket, session_id: str):
    await ws.accept()
    state = sessions.get(session_id)
    if state is None:
        await ws.send_json({"type":"error","msg":"invalid session"})
        await ws.close(); return

    try:
        while True:
            msg = await ws.receive_json()

            # ---- control messages ----
            if isinstance(msg, dict) and ("type" in msg):
                if msg["type"] == "clear_sentence":
                    state.sentence = ""
                    state.last_committed = None
                    state.refractory = False
                    state.release_run = 0
                    await ws.send_json({"type":"sentence","text":""})
                    continue
                if msg["type"] == "undo":
                    if state.sentence.strip():
                        words = state.sentence.strip().split()
                        words.pop()
                        state.sentence = " ".join(words) + (" " if words else "")
                        state.last_committed = lab2id.get(words[-1], None) if words else None
                        # after undo: temporarily require a release before re-commit
                        state.refractory = True
                        state.release_run = 0
                    await ws.send_json({"type":"sentence","text":state.sentence})
                    continue
                

            # ---- inputs ----
            P33 = np.zeros((33,4), np.float32)
            if "pose33" in msg and msg["pose33"]:
                P33 = np.array(msg["pose33"], dtype=np.float32).reshape(33,4)
            LH = np.zeros((21,3), np.float32)
            if "left21" in msg and msg["left21"]:
                LH = np.array(msg["left21"], dtype=np.float32).reshape(21,3)
            RH = np.zeros((21,3), np.float32)
            if "right21" in msg and msg["right21"]:
                RH = np.array(msg["right21"], dtype=np.float32).reshape(21,3)

            pose_xyz = P33[POSE_KEEP, :3]
            center, scale = torso_center_and_scale(P33)
            theta = theta_from_shoulders(P33)
            Rz = zrot(theta)

            f = np.concatenate([pose_xyz, LH, RH], axis=0).astype(np.float32)
            f = (f - center[None,:]) / scale
            f = f @ Rz.T

            mask = (np.linalg.norm(f, axis=-1) > 0).astype(np.float32)
            if float(mask.mean()) < MIN_VALID_JOINT_FRAC_PER_FRAME:
                await ws.send_json({"type":"prediction","topk":[],"stable":False,"idle":True})
                # idle counts toward release if we are refractory
                if state.refractory:
                    state.release_run += 1
                    if state.release_run >= RELEASE_FRAMES:
                        state.refractory = False
                continue

            frame = np.concatenate([f, mask[:,None]], axis=-1)
            state.buffer.append(frame)

            if len(state.buffer) < int(T*0.6):
                await ws.send_json({"type":"prediction","topk":[],"stable":False,"filling":True})
                continue

            window = np.stack(state.buffer, axis=0)
            if window.shape[0] != T:
                window = resample_time(window, T)

            if float((window[..., -1] > 0).mean()) < MIN_VALID_JOINT_FRAC_WINDOW:
                await ws.send_json({"type":"prediction","topk":[],"stable":False,"idle":True})
                if state.refractory:
                    state.release_run += 1
                    if state.release_run >= RELEASE_FRAMES:
                        state.refractory = False
                continue

            if C_NO_MASK == 6:
                window = add_velocity(window)

            feat = window[...,:C_NO_MASK]
            mask_ch = window[...,-1:]
            feat = (feat - MEAN) / (STD + 1e-6)
            X_TJC = np.concatenate([feat, mask_ch], axis=-1).astype(np.float32)
            Xb = adapt_features(X_TJC)

            with torch.no_grad():
                logits = model(torch.from_numpy(Xb).to(device))
                logits_np = logits.cpu().numpy()[0]

            if state.ema_logits is None:
                state.ema_logits = logits_np
            else:
                state.ema_logits = EMA_ALPHA*state.ema_logits + (1-EMA_ALPHA)*logits_np

            probs = torch.softmax(torch.tensor(state.ema_logits), dim=-1).numpy()
            topk_idx = probs.argsort()[-3:][::-1]
            top1 = int(topk_idx[0]); p1 = float(probs[top1])

            is_stable_now = (p1 >= PROB_THRESHOLD)
            same_as_committed = (state.last_committed is not None and top1 == state.last_committed)

            await ws.send_json({
                "type":"prediction",
                "topk":[{"label": id2lab[i], "p": float(probs[i])} for i in topk_idx],
                "stable": is_stable_now
            })

            if is_stable_now:
                # stability accounting
                if state.last_top1 == top1:
                    state.stable_run += 1
                else:
                    state.stable_run = 1
                    state.last_top1 = top1

                if state.refractory:
                    # Only advance release if we’re NOT holding the same committed word
                    if not same_as_committed:
                        state.release_run += 1
                        if state.release_run >= RELEASE_FRAMES:
                            state.refractory = False
                    # prevent runaway stable_run while refractory
                    if state.stable_run > HOLD_FRAMES:
                        state.stable_run = HOLD_FRAMES
                else:
                    # allowed to commit
                    if state.stable_run >= HOLD_FRAMES:
                        if (state.last_committed is None) or (top1 != state.last_committed):
                            word = id2lab[top1]
                            state.sentence += word + " "
                            state.last_committed = top1
                            state.stable_run = 0

                            # enter refractory: require a release before next commit
                            state.refractory = True
                            state.release_run = 0

                            await ws.send_json({"type":"word_event","label":word,"confidence":p1})
                            await ws.send_json({"type":"sentence","text":state.sentence})
                        else:
                            # guard: same as committed, do not re-emit
                            state.stable_run = 0
            else:
                # unstable/idle → counts toward release if refractory
                if state.refractory:
                    state.release_run += 1
                    if state.release_run >= RELEASE_FRAMES:
                        state.refractory = False
                state.stable_run = 0
                state.last_top1 = None

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except:
            pass
