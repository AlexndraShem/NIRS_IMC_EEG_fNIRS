"""Инференс FBShallow без torch (чистый numpy) + экспорт весов в .npz.

    python -m src.dl.np_infer export <tag>_<fold>      # data/models/dl/<..>.pt -> <..>.npz
    python -m src.dl.np_infer check  <tag>_<fold>      # сверка numpy vs torch и время одного окна

Свёртки temporal (без bias) и spatial (без bias) линейны и идут подряд, поэтому сливаются в одно
ядро K[s, c, τ] = Σ_t W_sp[s, t, c]·W_t[t, τ]; дальше BN (eval) → x² → среднее по n_sub
подокнам → log → fc. Один вызов = одна матричная операция (T' × C·k) @ (C·k × n_s).
"""
from __future__ import annotations

import sys
import time

import numpy as np

from src.dl.train import MODEL_DIR


def export(name):
    return export_to(MODEL_DIR / f"{name}.pt", MODEL_DIR / f"{name}.npz")


def export_to(src, dst):
    import torch
    from pathlib import Path
    ck = torch.load(src, map_location="cpu", weights_only=False)
    s = {k: v.numpy().astype(np.float64) for k, v in ck["state"].items()}
    cfg = ck["cfg"]
    assert cfg["model"] == "fbshallow"
    wt = s["temporal.weight"][:, 0, 0, :]                 # (n_t, k)
    ws = s["spatial.weight"][:, :, :, 0]                   # (n_s, n_t, C)
    K = np.einsum("stc,tk->sck", ws, wt)                   # (n_s, C, k)
    g = s["bn.weight"] / np.sqrt(s["bn.running_var"] + 1e-5)
    b = s["bn.bias"] - s["bn.running_mean"] * g
    n_sub = cfg["kw"].get("n_sub", 4)
    out = dict(K=K.reshape(len(K), -1).T.astype(np.float32), k=K.shape[2], bn_g=g, bn_b=b, n_sub=n_sub,
               fc_w=s["fc.weight"], fc_b=s["fc.bias"], L=cfg["L"], norm=cfg["norm"], stats=cfg["stats"],
               bands=np.array(cfg["bands"], float).reshape(-1, 2))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    return dst


class NpFBShallow:
    def __init__(self, path):
        z = np.load(path)
        self.K, self.k, self.n_sub = z["K"], int(z["k"]), int(z["n_sub"])
        self.g, self.b = z["bn_g"].astype(np.float32), z["bn_b"].astype(np.float32)
        self.fc_w, self.fc_b = z["fc_w"], z["fc_b"]
        self.L = int(z["L"])

    def forward(self, x):
        """x (C, L) нормированный → (logits (3,), emb (n_s·n_sub,))."""
        x = np.ascontiguousarray(x, np.float32)
        U = np.lib.stride_tricks.sliding_window_view(x, self.k, axis=1)       # (C, T', k)
        U = U.transpose(1, 0, 2).reshape(U.shape[1], -1)                      # (T', C·k)
        y = (U @ self.K) * self.g + self.b                                    # (T', n_s)
        y = y * y
        T = len(y)
        emb = np.stack([y[(i * T) // self.n_sub: -((-(i + 1) * T) // self.n_sub)].mean(0)
                        for i in range(self.n_sub)], 1)                      # (n_s, n_sub)
        emb = np.log(np.maximum(emb, 1e-6)).reshape(-1).astype(np.float64)
        return self.fc_w @ emb + self.fc_b, emb


    def embed_many(self, x, starts):
        """Эмбеддинги многих окон длины L одного непрерывного куска x (C, n): свёртка считается
        один раз по всему куску (окна перекрываются), средние по подокнам — через cumsum."""
        x = np.ascontiguousarray(x, np.float32)
        U = np.lib.stride_tricks.sliding_window_view(x, self.k, axis=1)
        y = (U.transpose(1, 0, 2).reshape(U.shape[1], -1) @ self.K) * self.g + self.b
        cs = np.vstack([np.zeros((1, y.shape[1])), np.cumsum((y * y).astype(np.float64), 0)])
        T = self.L - self.k + 1
        out = []
        for i in range(self.n_sub):
            a, b = (i * T) // self.n_sub, -((-(i + 1) * T) // self.n_sub)
            st = np.asarray(starts)
            out.append((cs[st + b] - cs[st + a]) / (b - a))                   # (n_win, n_s)
        emb = np.log(np.maximum(np.stack(out, 2), 1e-6)).reshape(len(starts), -1)
        return emb @ self.fc_w.T + self.fc_b, emb


def check(name):
    import torch
    from src.dl.models import build
    ck = torch.load(MODEL_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    C = len(cfg["bands"]) * 21 if cfg["bands"] else 21
    m = build("fbshallow", C=C, T=cfg["L"], **cfg["kw"]); m.load_state_dict(ck["state"]); m.eval()
    npm = NpFBShallow(export(name))
    x = np.random.randn(C, cfg["L"]).astype(np.float32)
    with torch.no_grad():
        lt, et = m(torch.from_numpy(x)[None])
    ln, en = npm.forward(x)
    print("max|Δlogits|", np.abs(lt.numpy()[0] - ln).max(), "max|Δemb|", np.abs(et.numpy()[0] - en).max())
    for _ in range(5):
        npm.forward(x)
    t = []
    for _ in range(200):
        t0 = time.perf_counter(); npm.forward(x); t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e3
    print(f"numpy forward C={C} L={cfg['L']}: mean {t.mean():.2f} ms, p95 {np.percentile(t, 95):.2f}, max {t.max():.2f} ms")
    print("K size (КБ):", npm.K.nbytes / 1024)


if __name__ == "__main__":
    {"export": export, "check": check}[sys.argv[1]](sys.argv[2])
