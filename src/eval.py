"""Evaluación de super-resolución: conversión de color, PSNR/SSIM sobre canal Y,
y loaders de Set5/Set14 (versiones LR/HR de SelfExSR de Jia-Bin Huang).

Convención (la del paper SRCNN y la que pide el spec):
- Las métricas se calculan SOLO sobre el canal Y (luminancia) en espacio YCbCr.
- Se usa la conversión rgb2ycbcr "estilo MATLAB" (rango limitado 16-235), que es la
  que usan los papers de SR. Usar la conversión full-range de cv2/skimage da PSNR
  distinto y NO reproduce los números canónicos.
- Antes de medir se recorta un borde de `scale` píxeles (shave) en cada lado, porque
  los bordes son artefactos del upsampling y se descartan por convención.
- PSNR se calcula con valor pico 255 aunque Y viva en [16,235].
"""
import os
import glob
import cv2
import numpy as np
from skimage.metrics import structural_similarity as _ssim

DATASETS_ROOT = os.path.join(os.path.dirname(__file__), "..", "datasets", "SelfExSR", "data")


def imread_rgb(path):
    """Lee una imagen como uint8 RGB (cv2 lee en BGR)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def modcrop(img, scale):
    """Recorta la imagen para que alto y ancho sean divisibles por `scale`."""
    h, w = img.shape[:2]
    h, w = h - (h % scale), w - (w % scale)
    return img[:h, :w, ...]


def rgb2y(img):
    """RGB uint8 -> canal Y (luminancia) estilo MATLAB rgb2ycbcr, en float [16,235].

    Y = 16 + (65.481*R + 128.553*G + 24.966*B) / 255   con R,G,B en [0,255].
    """
    img = img.astype(np.float64)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return y


def shave(img, border):
    """Recorta `border` píxeles en cada lado."""
    if border <= 0:
        return img
    return img[border:-border, border:-border, ...]


def psnr_y(sr_rgb, hr_rgb, scale):
    """PSNR sobre canal Y, con shave de `scale` píxeles. sr/hr: uint8 RGB del mismo tamaño."""
    ys = shave(rgb2y(sr_rgb), scale)
    yh = shave(rgb2y(hr_rgb), scale)
    mse = np.mean((ys - yh) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


def ssim_y(sr_rgb, hr_rgb, scale):
    """SSIM sobre canal Y, con shave de `scale` píxeles."""
    ys = shave(rgb2y(sr_rgb), scale)
    yh = shave(rgb2y(hr_rgb), scale)
    return float(_ssim(ys, yh, data_range=255.0))


# --- Conversión YCbCr completa (estilo MATLAB), para reconstruir RGB del modelo ---
_M = np.array([[65.481, 128.553, 24.966],
               [-37.797, -74.203, 112.0],
               [112.0, -93.786, -18.214]]) / 255.0
_OFF = np.array([16.0, 128.0, 128.0])
_MINV = np.linalg.inv(_M)


def rgb2ycbcr(img):
    """RGB uint8 -> (Y, Cb, Cr) float estilo MATLAB. Y∈[16,235], Cb/Cr∈[16,240].

    Implementación elementwise (sin matmul) para evitar warnings espurios de BLAS.
    """
    f = img.astype(np.float64)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y = _OFF[0] + (_M[0, 0] * r + _M[0, 1] * g + _M[0, 2] * b)
    cb = _OFF[1] + (_M[1, 0] * r + _M[1, 1] * g + _M[1, 2] * b)
    cr = _OFF[2] + (_M[2, 0] * r + _M[2, 1] * g + _M[2, 2] * b)
    return y, cb, cr


def ycbcr2rgb(y, cb, cr):
    """(Y, Cb, Cr) float estilo MATLAB -> RGB uint8 (elementwise)."""
    yy = np.nan_to_num(y) - _OFF[0]
    cbb = np.nan_to_num(cb) - _OFF[1]
    crr = np.nan_to_num(cr) - _OFF[2]
    r = _MINV[0, 0] * yy + _MINV[0, 1] * cbb + _MINV[0, 2] * crr
    g = _MINV[1, 0] * yy + _MINV[1, 1] * cbb + _MINV[1, 2] * crr
    b = _MINV[2, 0] * yy + _MINV[2, 1] * cbb + _MINV[2, 2] * crr
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def sr_from_model(model, lr_rgb, scale, device="cpu"):
    """Reconstruye SR RGB: Y lo predice el modelo; Cb/Cr se agrandan con bicubic.

    Devuelve (sr_rgb_uint8, y_sr_float) — y_sr sirve para PSNR directo sin round-trip RGB.
    """
    import torch
    from baselines import upsample_to

    y, cb, cr = rgb2ycbcr(lr_rgb)
    out_hw = (lr_rgb.shape[0] * scale, lr_rgb.shape[1] * scale)
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy((y / 255.0)[None, None].astype(np.float32)).to(device)
        y_sr = model(inp).clamp(0, 1).cpu().numpy()[0, 0] * 255.0
    # Cb/Cr a tamaño HR por bicubic (sobre el LR RGB, canal por canal vía upsample)
    lr_up = upsample_to(lr_rgb, out_hw, "bicubic")
    _, cb_sr, cr_sr = rgb2ycbcr(lr_up)
    # Recortar por si el modelo devuelve 1px de diferencia
    h = min(y_sr.shape[0], cb_sr.shape[0])
    w = min(y_sr.shape[1], cb_sr.shape[1])
    y_sr, cb_sr, cr_sr = y_sr[:h, :w], cb_sr[:h, :w], cr_sr[:h, :w]
    sr_rgb = ycbcr2rgb(y_sr, cb_sr, cr_sr)
    return sr_rgb, y_sr


def _se_forward(x, k, flip):
    """Transformación geométrica: espejo horizontal (opcional) + rot90 k veces."""
    if flip:
        x = x[:, ::-1]
    return np.rot90(x, k)


def _se_inverse(x, k, flip):
    """Deshace _se_forward: rot90 -k y luego el espejo (orden inverso)."""
    x = np.rot90(x, -k)
    if flip:
        x = x[:, ::-1]
    return x


def sr_y_self_ensemble(model, lr_rgb, scale, device="cpu"):
    """Y super-resuelto con SELF-ENSEMBLE: promedia la salida del modelo sobre las 8
    transformaciones geométricas (4 rotaciones x espejo on/off). Cada orientación se
    equivoca distinto; al promediar, los errores se cancelan -> Y más limpio (+0.1-0.3 dB),
    sin reentrenar. Solo es cómputo de inferencia (8 pasadas por imagen).
    """
    import torch

    y, _, _ = rgb2ycbcr(lr_rgb)
    y = y / 255.0
    model.eval()
    acc = None
    for flip in (False, True):
        for k in range(4):
            t = np.ascontiguousarray(_se_forward(y, k, flip)).astype(np.float32)
            with torch.no_grad():
                out = model(torch.from_numpy(t[None, None]).to(device)).clamp(0, 1)
            out = out.cpu().numpy()[0, 0]
            out = _se_inverse(out, k, flip)
            acc = out if acc is None else acc + out
    return (acc / 8.0) * 255.0


def sr_from_model_se(model, lr_rgb, scale, device="cpu"):
    """Como sr_from_model pero con self-ensemble en Y (Cb/Cr siguen por bicubic)."""
    from baselines import upsample_to

    y_sr = sr_y_self_ensemble(model, lr_rgb, scale, device)
    out_hw = (lr_rgb.shape[0] * scale, lr_rgb.shape[1] * scale)
    lr_up = upsample_to(lr_rgb, out_hw, "bicubic")
    _, cb_sr, cr_sr = rgb2ycbcr(lr_up)
    h = min(y_sr.shape[0], cb_sr.shape[0])
    w = min(y_sr.shape[1], cb_sr.shape[1])
    y_sr, cb_sr, cr_sr = y_sr[:h, :w], cb_sr[:h, :w], cr_sr[:h, :w]
    return ycbcr2rgb(y_sr, cb_sr, cr_sr), y_sr


def _psnr_arr(y1, y2, scale):
    a, b = shave(y1, scale), shave(y2, scale)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)


def eval_model(model, name, scale, device="cpu", self_ensemble=False):
    """PSNR/SSIM medio de un modelo en Set5/Set14, directo sobre canal Y (con shave).

    self_ensemble=True promedia 8 transformaciones geométricas (más lento, más PSNR).
    """
    items = load_set(name, scale)
    psnrs, ssims = [], []
    for it in items:
        if self_ensemble:
            y_sr = sr_y_self_ensemble(model, it["lr"], scale, device)
        else:
            _, y_sr = sr_from_model(model, it["lr"], scale, device)
        y_hr = rgb2y(it["hr"])[:y_sr.shape[0], :y_sr.shape[1]]
        psnrs.append(_psnr_arr(y_sr, y_hr, scale))
        ssims.append(float(_ssim(shave(y_sr, scale), shave(y_hr, scale), data_range=255.0)))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def load_set(name, scale):
    """Carga un dataset de evaluación. Devuelve lista de dicts {name, lr, hr} en uint8 RGB.

    name: 'Set5' o 'Set14'. Usa las imágenes LR (downsample exacto) y HR del repo SelfExSR.
    """
    folder = os.path.join(DATASETS_ROOT, name, f"image_SRF_{scale}")
    hr_paths = sorted(glob.glob(os.path.join(folder, f"*_SRF_{scale}_HR.png")))
    items = []
    for hp in hr_paths:
        lp = hp.replace("_HR.png", "_LR.png")
        hr = modcrop(imread_rgb(hp), scale)
        lr = imread_rgb(lp)
        items.append({"name": os.path.basename(hp), "lr": lr, "hr": hr})
    return items
