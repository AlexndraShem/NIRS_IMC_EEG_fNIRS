"""Компактные ЭЭГ-сети (125 Гц, вход (B, C, T)): ShallowConvNet, EEGNet, Deep4-lite.
Каждая сеть отдаёт (logits, emb): emb — вход последнего линейного слоя (для адаптационной головы)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Shallow(nn.Module):
    """ShallowConvNet (Schirrmeister 2017), пересчитан на 125 Гц: temporal k=13, pool 37/7."""

    def __init__(self, C=21, T=250, n_f=40, k=13, pool=37, stride=7, drop=0.5, n_cls=3):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_f, (1, k))
        self.spatial = nn.Conv2d(n_f, n_f, (C, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_f)
        self.pool = nn.AvgPool2d((1, pool), (1, stride))
        self.drop = nn.Dropout(drop)
        with torch.no_grad():
            n_emb = self._feat(torch.zeros(1, C, T)).shape[1]
        self.fc = nn.Linear(n_emb, n_cls)

    def _feat(self, x):
        x = self.spatial(self.temporal(x[:, None]))
        x = self.bn(x)
        x = self.pool(x * x)
        x = torch.log(torch.clamp(x, min=1e-6))
        return x.flatten(1)

    def forward(self, x):
        e = self._feat(x)
        return self.fc(self.drop(e)), e


class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern 2018) на 125 Гц: temporal k=32, pool 4 и 8, sep k=16."""

    def __init__(self, C=21, T=250, F1=8, D=2, F2=16, k1=32, k2=16, drop=0.25, n_cls=3):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.b2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, k2), padding=(0, k2 // 2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            n_emb = self._feat(torch.zeros(1, C, T)).shape[1]
        self.fc = nn.Linear(n_emb, n_cls)

    def _feat(self, x):
        return self.b2(self.b1(x[:, None])).flatten(1)

    def forward(self, x):
        e = self._feat(x)
        return self.fc(e), e


class FBShallow(nn.Module):
    """«Фильтр-банковый» Shallow: больше временных фильтров с длинным ядром (k=33 ≈ 0.26 с),
    пространственная свёртка, лог-мощность по всему окну в нескольких подокнах."""

    def __init__(self, C=21, T=250, n_t=24, n_s=48, k=33, n_sub=4, drop=0.5, n_cls=3):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_t, (1, k), bias=False)
        self.spatial = nn.Conv2d(n_t, n_s, (C, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_s)
        self.n_sub = n_sub
        self.drop = nn.Dropout(drop)
        with torch.no_grad():
            n_emb = self._feat(torch.zeros(1, C, T)).shape[1]
        self.fc = nn.Linear(n_emb, n_cls)

    def _feat(self, x):
        x = self.bn(self.spatial(self.temporal(x[:, None])))[:, :, 0]      # (B, n_s, T')
        x = F.adaptive_avg_pool1d(x * x, self.n_sub)
        return torch.log(torch.clamp(x, min=1e-6)).flatten(1)

    def forward(self, x):
        e = self._feat(x)
        return self.fc(self.drop(e)), e


def build(name, C=21, T=250, **kw):
    return {"shallow": Shallow, "eegnet": EEGNet, "fbshallow": FBShallow}[name](C=C, T=T, **kw)
