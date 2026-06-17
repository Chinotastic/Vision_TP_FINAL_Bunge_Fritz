"""Eje 3 — eficiencia: nº de parámetros y tiempo de inferencia por imagen.

Para cada modelo entrenado en outputs/<tag>/ mide:
  - parámetros (capacidad del modelo),
  - latencia media por imagen de Set5 (ms) y FPS, sobre el device disponible,
  - PSNR de Set5 (leído de history.json).
Genera outputs/bench.md y un scatter PSNR-vs-latencia (outputs/figures/speed_x{scale}.png),
que conecta con la motivación de la tarea (calidad vs velocidad, estilo DLSS).
"""
import os
import re
import json
import glob
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import build_model
from eval import load_set, rgb2ycbcr

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIGDIR = os.path.join(OUT, "figures")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def discover(scale):
    """Tags entrenados para un factor dado -> [(tag, arch)]."""
    found = []
    for d in sorted(glob.glob(os.path.join(OUT, f"*_x{scale}_*"))):
        if not os.path.exists(os.path.join(d, "best.pth")):
            continue
        tag = os.path.basename(d)
        arch = tag.split("_x")[0]
        found.append((tag, arch))
    return found


@torch.no_grad()
def measure(arch, scale, tag, lr_y_list, warmup=3, runs=10):
    m = build_model(arch, scale).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(OUT, tag, "best.pth"), map_location=DEVICE))
    m.eval()
    n_params = sum(p.numel() for p in m.parameters())
    # latencia media por imagen
    times = []
    for _ in range(warmup):
        m(lr_y_list[0])
    for _ in range(runs):
        t0 = time.perf_counter()
        for x in lr_y_list:
            m(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / len(lr_y_list))
    return n_params, float(np.mean(times) * 1000)  # ms/imagen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", type=int, default=[2, 4])
    args = ap.parse_args()
    print(f"device={DEVICE}")
    lines = ["# Eficiencia — parámetros, latencia, FPS\n"]
    for scale in args.scales:
        items = load_set("Set5", scale)
        lr_y = []
        for it in items:
            y, _, _ = rgb2ycbcr(it["lr"])
            lr_y.append(torch.from_numpy((y / 255.0)[None, None].astype(np.float32)).to(DEVICE))
        rows = []
        for tag, arch in discover(scale):
            n, ms = measure(arch, scale, tag, lr_y)
            hp = os.path.join(OUT, tag, "history.json")
            psnr = json.load(open(hp))["final"]["Set5"]["psnr"] if os.path.exists(hp) else float("nan")
            rows.append((tag, n, ms, 1000.0 / ms, psnr))
        lines.append(f"\n## Factor x{scale} (device={DEVICE})\n")
        lines.append("| Modelo | Params | ms/img | FPS | Set5 PSNR |")
        lines.append("|---|---|---|---|---|")
        for tag, n, ms, fps, psnr in rows:
            lines.append(f"| {tag} | {n:,} | {ms:.2f} | {fps:.1f} | {psnr:.2f} |")
        # scatter PSNR vs latencia
        if rows:
            fig, ax = plt.subplots(figsize=(6, 4))
            for tag, n, ms, fps, psnr in rows:
                ax.scatter(ms, psnr, s=40)
                ax.annotate(tag.split(f"_x{scale}")[0], (ms, psnr),
                            textcoords="offset points", xytext=(5, 4), fontsize=8)
            ax.set_xlabel(f"latencia (ms/imagen, {DEVICE})"); ax.set_ylabel("Set5 PSNR (dB)")
            ax.set_title(f"Calidad vs velocidad — x{scale}")
            fig.tight_layout(); os.makedirs(FIGDIR, exist_ok=True)
            fig.savefig(os.path.join(FIGDIR, f"speed_x{scale}.png"), dpi=120); plt.close(fig)
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "bench.md"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
