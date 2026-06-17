"""GATE de la Fase 1: validar que el pipeline de evaluación es correcto.

Calcula el PSNR/SSIM del baseline bicúbico (estilo MATLAB) sobre Set5 y Set14, en el
canal Y con shave. Si el medidor está bien, el PSNR del bicúbico debe coincidir con
los valores canónicos de la literatura (±~0.1 dB):

    Set5  : 2x ~33.66 | 3x ~30.39 | 4x ~28.42
    Set14 : 2x ~30.24 | 3x ~27.55 | 4x ~26.00

Si estos números NO dan, NO avanzar a entrenar: cualquier PSNR del modelo sería basura.
"""
import numpy as np
from eval import load_set, psnr_y, ssim_y
from baselines import upsample_to

CANON = {
    ("Set5", 2): 33.66, ("Set5", 3): 30.39, ("Set5", 4): 28.42,
    ("Set14", 2): 30.24, ("Set14", 3): 27.55, ("Set14", 4): 26.00,
}


def eval_bicubic(name, scale):
    items = load_set(name, scale)
    psnrs, ssims = [], []
    for it in items:
        sr = upsample_to(it["lr"], it["hr"].shape[:2], "bicubic")
        psnrs.append(psnr_y(sr, it["hr"], scale))
        ssims.append(ssim_y(sr, it["hr"], scale))
    return float(np.mean(psnrs)), float(np.mean(ssims))


if __name__ == "__main__":
    print(f"{'dataset':8} {'sf':>3} {'PSNR':>7} {'canon':>7} {'dPSNR':>7} {'SSIM':>7}")
    for name in ["Set5", "Set14"]:
        for sf in [2, 3, 4]:
            p, s = eval_bicubic(name, sf)
            c = CANON[(name, sf)]
            flag = "OK" if abs(p - c) <= 0.15 else "<-- REVISAR"
            print(f"{name:8} {sf:>3} {p:7.2f} {c:7.2f} {p - c:+7.2f} {s:7.4f}  {flag}")
