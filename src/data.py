"""Dataset de entrenamiento para super-resolución (canal Y).

Convención de entrenamiento (coherente con eval.py y con FSRCNN):
- Se trabaja SOLO sobre el canal Y (luminancia), normalizado a [0,1] (Y_matlab/255).
- Para cada imagen HR de entrenamiento se genera su par LR con bicubic-downscale
  estilo MATLAB (imresize_matlab con escala 1/scale). Esto replica la degradación
  con la que se evalúa, evitando el "train/test mismatch".
- El modelo (FSRCNN) recibe el parche LR (pequeño) y produce el parche HR
  (lr_size*scale). La variante residual suma internamente el bicubic del LR.

El Dataset muestrea parches aleatorios (con flips/rot90 de data augmentation) y su
longitud es `patches_per_epoch`, independiente del nº de imágenes.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from eval import imread_rgb, modcrop, rgb2y
from baselines import imresize_matlab


def _load_y_pairs(image_dir, scale):
    """Carga todas las imágenes de `image_dir` como pares (Y_hr, Y_lr) float [0,1]."""
    paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.bmp"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
    )
    pairs = []
    for p in paths:
        hr_rgb = modcrop(imread_rgb(p), scale)
        y_hr = rgb2y(hr_rgb) / 255.0                       # [0,1], tamaño HR
        y_lr = imresize_matlab(y_hr, 1.0 / scale)          # bicubic-downscale MATLAB
        pairs.append((y_hr.astype(np.float32), y_lr.astype(np.float32)))
    return pairs


def _augment(lr, hr, rng):
    """Flips horizontal/vertical y rot90 aplicados consistentemente a lr y hr."""
    if rng.random() < 0.5:
        lr, hr = lr[:, ::-1], hr[:, ::-1]
    if rng.random() < 0.5:
        lr, hr = lr[::-1, :], hr[::-1, :]
    if rng.random() < 0.5:
        lr, hr = np.rot90(lr), np.rot90(hr)
    return np.ascontiguousarray(lr), np.ascontiguousarray(hr)


class SRPatchDataset(Dataset):
    """Parches aleatorios (LR, HR) del canal Y para entrenar.

    lr_size: lado del parche LR. El parche HR resultante es lr_size*scale.
    """

    def __init__(self, image_dir, scale, lr_size=24, patches_per_epoch=8000,
                 augment=True, seed=0):
        self.scale = scale
        self.lr_size = lr_size
        self.hr_size = lr_size * scale
        self.patches_per_epoch = patches_per_epoch
        self.augment = augment
        self.pairs = _load_y_pairs(image_dir, scale)
        if not self.pairs:
            raise RuntimeError(f"No se encontraron imágenes en {image_dir}")
        # Solo imágenes lo bastante grandes para muestrear un parche LR completo
        self.pairs = [pr for pr in self.pairs
                      if pr[1].shape[0] >= lr_size and pr[1].shape[1] >= lr_size]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.patches_per_epoch

    def __getitem__(self, idx):
        y_hr, y_lr = self.pairs[self.rng.integers(len(self.pairs))]
        h, w = y_lr.shape
        ly = self.rng.integers(0, h - self.lr_size + 1)
        lx = self.rng.integers(0, w - self.lr_size + 1)
        lr = y_lr[ly:ly + self.lr_size, lx:lx + self.lr_size]
        hy, hx = ly * self.scale, lx * self.scale
        hr = y_hr[hy:hy + self.hr_size, hx:hx + self.hr_size]
        if self.augment:
            lr, hr = _augment(lr, hr, self.rng)
        # (1, H, W) tensores
        return (torch.from_numpy(lr[None].copy()),
                torch.from_numpy(hr[None].copy()))
