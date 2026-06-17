"""Funciones de pérdida para super-resolución.

- L1 / L2 (MSE): pérdidas de distorsión estándar (las que maximizan PSNR).
- Charbonnier: sqrt((x-y)^2 + eps^2), variante suave de L1 muy usada en SR (EDSR/LapSRN).
- VGGPerceptualLoss: compara features de una VGG19 pre-entrenada en vez de píxeles. Favorece la
  nitidez percibida aunque baje el PSNR (eje distorsión-vs-percepción). Como trabajamos en canal Y,
  replicamos Y a 3 canales y normalizamos con las stats de ImageNet antes de pasar por la VGG.

NOTA Kaggle: la VGG19 pre-entrenada requiere descarga -> activar Internet ON en el notebook.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, sr, hr):
        return torch.mean(torch.sqrt((sr - hr) ** 2 + self.eps2))


class VGGPerceptualLoss(nn.Module):
    """MSE entre features VGG19 de sr y hr. layer_idx=35 ~ relu5_4 (alto nivel)."""

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    def __init__(self, layer_idx=35):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:layer_idx].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self._STD).view(1, 3, 1, 1))

    def _prep(self, y):
        y = y.clamp(0, 1).repeat(1, 3, 1, 1)      # Y(1ch) -> 3ch
        return (y - self.mean) / self.std

    def forward(self, sr, hr):
        return F.mse_loss(self.vgg(self._prep(sr)), self.vgg(self._prep(hr)))


def build_loss(name):
    """name: 'l1' | 'l2' | 'charbonnier'. (perceptual se maneja aparte por su peso/combinación)."""
    name = name.lower()
    if name == "l1":
        return nn.L1Loss()
    if name == "l2":
        return nn.MSELoss()
    if name == "charbonnier":
        return CharbonnierLoss()
    raise ValueError(name)
