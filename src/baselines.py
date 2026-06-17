"""Baselines de upsampling: nearest, bilinear y bicubic.

El bicubic es el baseline de referencia que el modelo debe superar, y el que define
el "gate" de la Fase 1. Para reproducir los números canónicos de la literatura
(p.ej. Set5 2x ≈ 33.66 dB) hay que usar el bicubic ESTILO MATLAB (kernel a=-0.5),
no el de OpenCV (a=-0.75). Por eso incluimos un `imresize` portado de MATLAB.

nearest y bilinear se hacen con cv2 (son solo baselines extra, no el gate).
"""
import numpy as np
import cv2


# ----------------------------------------------------------------------------
# imresize estilo MATLAB (kernel bicúbico a=-0.5). Port estándar usado en repos de SR.
# ----------------------------------------------------------------------------
def _cubic(x):
    ax = np.abs(x)
    ax2 = ax ** 2
    ax3 = ax ** 3
    f = (1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1)
    f += (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2))
    return f


def _contributions(in_length, out_length, scale, kernel_width):
    # Antialiasing solo al reducir (scale < 1); al ampliar no se ensancha el kernel.
    if scale < 1:
        kernel_width = kernel_width / scale
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    p = int(np.ceil(kernel_width)) + 2
    ind = left[:, None] + np.arange(p)[None, :]
    if scale < 1:
        weights = scale * _cubic(scale * (u[:, None] - ind))
    else:
        weights = _cubic(u[:, None] - ind)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    ind = np.minimum(np.maximum(ind - 1, 0), in_length - 1).astype(np.int64)
    # Descartar columnas con peso totalmente nulo
    keep = np.any(weights != 0, axis=0)
    return weights[:, keep], ind[:, keep]


def _resize_along_dim(img, dim, weights, indices):
    img = np.moveaxis(img, dim, 0)
    out = np.tensordot(weights, img[indices], axes=([1], [1]))
    # tensordot deja eje de salida adelante; reordenar
    out = np.einsum("ij...->i...", out) if False else out
    out = np.moveaxis(out, 0, dim)
    return out


def imresize_matlab(img, scale):
    """imresize estilo MATLAB con kernel bicúbico (a=-0.5). img: float [0,255] o [0,1].

    scale: factor (>1 amplía, <1 reduce). Devuelve float del mismo dtype lógico.
    """
    img = img.astype(np.float64)
    h, w = img.shape[:2]
    oh, ow = int(round(h * scale)), int(round(w * scale))
    kernel_width = 4.0

    wy, iy = _contributions(h, oh, scale, kernel_width)
    wx, ix = _contributions(w, ow, scale, kernel_width)

    # Resize vertical
    out = np.zeros((oh, w) + img.shape[2:], dtype=np.float64)
    for c in range(oh):
        out[c] = np.tensordot(wy[c], img[iy[c]], axes=([0], [0]))
    # Resize horizontal
    out2 = np.zeros((oh, ow) + img.shape[2:], dtype=np.float64)
    for c in range(ow):
        out2[:, c] = np.tensordot(wx[c], out[:, ix[c]], axes=([0], [1]))
    return out2


def upsample_to(lr_rgb, out_hw, method):
    """Amplía lr_rgb (uint8 RGB) al tamaño out_hw=(H,W) con el método dado.

    method: 'nearest' | 'bilinear' | 'bicubic'. 'bicubic' usa imresize estilo MATLAB.
    Devuelve uint8 RGB.
    """
    H, W = out_hw
    h, w = lr_rgb.shape[:2]
    if method == "nearest":
        out = cv2.resize(lr_rgb, (W, H), interpolation=cv2.INTER_NEAREST)
    elif method == "bilinear":
        out = cv2.resize(lr_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    elif method == "bicubic":
        scale = H / h  # asumimos ratio entero igual en ambos ejes
        out = imresize_matlab(lr_rgb, scale)
        out = out[:H, :W]
        out = np.clip(np.round(out), 0, 255).astype(np.uint8)
        return out
    else:
        raise ValueError(method)
    return out
