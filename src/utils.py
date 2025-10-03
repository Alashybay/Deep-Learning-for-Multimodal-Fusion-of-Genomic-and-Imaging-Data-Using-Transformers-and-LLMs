import time
from typing import Dict
import numpy as np

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def accuracy(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> float:
    y_pred = (y_prob >= thr).astype(int)
    return float((y_pred == y_true).mean())

def fmt_losses(ep: int, epochs: int, losses: Dict[str, float]) -> str:
    parts = [f"{k}={losses[k]:.4f}" for k in ["inter","early","img","gen"]]
    return f"Epoch {ep:02d}/{epochs} | " + " | ".join(parts)
