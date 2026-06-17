"""Fase 3/4 — genera toda la evidencia a partir de los pesos en outputs/<tag>/best.pth.

Auto-descubre TODOS los modelos entrenados y produce:
  1) Tabla PSNR/SSIM: baselines + cada modelo, en los test sets pedidos (Set5/Set14/BSD100/Urban100).
     -> outputs/tables.md   (las ablaciones de pérdida y arquitectura aparecen como filas)
  2) Paneles cualitativos (full + zoom) -> outputs/figures/panel_*.png
  3) Panel de percepción: Bicubic / mejor-PSNR / GAN / GT -> outputs/figures/perception_x{f}.png
  4) Curvas de convergencia de todos los modelos de un factor -> outputs/figures/conv_x{f}.png

Uso:
    python make_figures.py --scales 2 4 --testsets Set5 Set14 BSD100 Urban100
"""
import os
import re
import json
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval import (load_set, psnr_y, ssim_y, rgb2y, sr_from_model, eval_model, shave, _psnr_arr)
from baselines import upsample_to
from models import build_model

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIGDIR = os.path.join(OUT, "figures")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BASELINES = ["nearest", "bilinear", "bicubic"]


def discover(scale):
    """[(tag, arch)] de modelos entrenados para un factor."""
    out = []
    for d in sorted(glob.glob(os.path.join(OUT, f"*_x{scale}_*"))):
        if os.path.exists(os.path.join(d, "best.pth")):
            tag = os.path.basename(d)
            out.append((tag, tag.split("_x")[0]))
    return out


def load_trained(arch, scale, tag):
    m = build_model(arch, scale).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(OUT, tag, "best.pth"), map_location=DEVICE))
    m.eval()
    return m


def eval_baseline(name, scale, method):
    items = load_set(name, scale)
    ps, ss = [], []
    for it in items:
        sr = upsample_to(it["lr"], it["hr"].shape[:2], method)
        hr = it["hr"][:sr.shape[0], :sr.shape[1]]
        ps.append(psnr_y(sr, hr, scale)); ss.append(ssim_y(sr, hr, scale))
    return float(np.mean(ps)), float(np.mean(ss))


def build_tables(scales, testsets):
    lines = ["# Resultados — PSNR / SSIM (canal Y, con shave)\n"]
    for scale in scales:
        lines.append(f"\n## Factor x{scale}\n")
        head = "| Método | " + " | ".join(f"{t} PSNR | {t} SSIM" for t in testsets) + " |"
        lines.append(head)
        lines.append("|---" * (1 + 2 * len(testsets)) + "|")
        rows = []
        for method in BASELINES:
            cells = []
            for t in testsets:
                p, s = eval_baseline(t, scale, method)
                cells += [f"{p:.2f}", f"{s:.4f}"]
            rows.append((method, cells, False))
        for tag, arch in discover(scale):
            m = load_trained(arch, scale, tag)
            cells = []
            for t in testsets:
                p, s = eval_model(m, t, scale, DEVICE)
                cells += [f"{p:.2f}", f"{s:.4f}"]
            rows.append((tag, cells, True))
        for name, cells, bold in rows:
            nm = f"**{name}**" if bold else name
            vals = " | ".join(f"**{c}**" if bold else c for c in cells)
            lines.append(f"| {nm} | {vals} |")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write(txt + "\n")
    print(txt)


def _center_crop(img, c):
    h, w = img.shape[:2]
    y, x = (h - c) // 2, (w - c) // 2
    return img[y:y + c, x:x + c]


def _sr_and_psnr(tag, arch, scale, it):
    m = load_trained(arch, scale, tag)
    sr, y_sr = sr_from_model(m, it["lr"], scale, DEVICE)
    y_hr = rgb2y(it["hr"])[:y_sr.shape[0], :y_sr.shape[1]]
    return sr, _psnr_arr(y_sr, y_hr, scale)


def make_panel(dataset, scale, idx, tags, crop=80, fname=None):
    """Panel 2 filas (full + zoom). Columnas: Bicubic | <tags...> | GT."""
    items = load_set(dataset, scale)
    it = items[idx]; hr = it["hr"]
    cols = []
    bic = upsample_to(it["lr"], hr.shape[:2], "bicubic")
    cols.append(("Bicubic", bic, psnr_y(bic, hr[:bic.shape[0], :bic.shape[1]], scale)))
    for tag, arch in tags:
        sr, p = _sr_and_psnr(tag, arch, scale, it)
        cols.append((tag.split(f"_x{scale}")[0], sr, p))
    cols.append(("Ground Truth", hr, None))

    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4))
    for j, (title, img, p) in enumerate(cols):
        h = min(img.shape[0], hr.shape[0]); w = min(img.shape[1], hr.shape[1]); img = img[:h, :w]
        axes[0, j].imshow(img); axes[0, j].axis("off")
        axes[0, j].set_title(title if p is None else f"{title}\n{p:.2f} dB", fontsize=10)
        axes[1, j].imshow(_center_crop(img, min(crop, h, w))); axes[1, j].axis("off")
    fig.suptitle(f"{dataset} x{scale} — {it['name'].replace('_HR.png','')}", fontsize=12)
    fig.tight_layout(); os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, fname or f"panel_{dataset}_x{scale}_{idx}.png")
    fig.savefig(out, dpi=120); plt.close(fig); print("ok ->", out)


def make_convergence(scale):
    fig, ax = plt.subplots(figsize=(6, 4)); found = False
    for tag, arch in discover(scale):
        hp = os.path.join(OUT, tag, "history.json")
        if not os.path.exists(hp):
            continue
        d = json.load(open(hp))
        if "val_psnr" in d and d.get("epoch"):
            ax.plot(d["epoch"], d["val_psnr"], "-o", ms=3, label=tag.split(f"_x{scale}")[0]); found = True
    if not found:
        plt.close(fig); return
    ax.set_xlabel("epoch"); ax.set_ylabel("Set5 PSNR (dB)")
    ax.set_title(f"Convergencia x{scale}"); ax.legend(fontsize=8); fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, f"conv_x{scale}.png"); fig.savefig(out, dpi=120); plt.close(fig)
    print("ok ->", out)


def make_perception_panel(scale, idx=0, dataset="Set14"):
    """Panel del tradeoff: Bicubic | mejor distorsión (mayor PSNR no-GAN) | GAN | GT."""
    tags = discover(scale)
    gan = [(t, a) for t, a in tags if "gan" in t]
    nogan = [(t, a) for t, a in tags if "gan" not in t]
    if not gan or not nogan:
        return
    # mejor distorsión por PSNR final
    def psnr_of(tag):
        hp = os.path.join(OUT, tag, "history.json")
        return json.load(open(hp))["final"]["Set5"]["psnr"] if os.path.exists(hp) else -1
    best = max(nogan, key=lambda ta: psnr_of(ta[0]))
    make_panel(dataset, scale, idx, [best, gan[0]], fname=f"perception_x{scale}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--testsets", nargs="+", default=["Set5", "Set14"])
    ap.add_argument("--panels", nargs="+", type=int, default=[0, 2])
    args = ap.parse_args()
    print(f"device={DEVICE}")
    build_tables(args.scales, args.testsets)
    for scale in args.scales:
        make_convergence(scale)
        tags = discover(scale)
        for idx in args.panels:
            make_panel("Set5", scale, idx, tags)
        make_perception_panel(scale)


if __name__ == "__main__":
    main()
