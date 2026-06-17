"""Entrenamiento adversarial estilo SRGAN (canal Y) — eje distorsión vs. percepción.

Generador = una de nuestras redes (FSRCNN/EDSR). Discriminador aprende a distinguir HR real
de la salida del generador. La pérdida del generador combina:
    G_loss = contenido (L1 + perceptual VGG) + adv_weight * adversarial
El resultado típico: PSNR/SSIM algo MÁS BAJOS que el modelo L1, pero imágenes más nítidas
(texturas plausibles). Es el contraste central del póster.

Buena práctica (SRGAN): inicializar el generador desde un modelo ya pre-entrenado con L1
(--init outputs/<tag>/best.pth) y recién ahí afinar adversarialmente.

Uso:
    python train_gan.py --gen edsr --scale 4 --init ../outputs/edsr_x4_l1/best.pth --epochs 150
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import SRPatchDataset
from models import build_model, Discriminator
from losses import VGGPerceptualLoss
from eval import eval_model

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="edsr", choices=["srcnn", "fsrcnn", "fsrcnn_res", "edsr"])
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--train-dir", default=os.path.join("..", "datasets", "T91"))
    ap.add_argument("--init", default=None, help="pesos pre-entrenados del generador (L1)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-size", type=int, default=24)
    ap.add_argument("--patches-per-epoch", type=int, default=8000)
    ap.add_argument("--adv-weight", type=float, default=1e-3)
    ap.add_argument("--perc-weight", type=float, default=0.05)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    device = get_device()
    tag = args.tag or f"{args.gen}_x{args.scale}_gan"
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{tag}] device={device}")

    ds = SRPatchDataset(args.train_dir, scale=args.scale, lr_size=args.lr_size,
                        patches_per_epoch=args.patches_per_epoch)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True)

    G = build_model(args.gen, args.scale).to(device)
    if args.init and os.path.exists(args.init):
        G.load_state_dict(torch.load(args.init, map_location=device))
        print(f"generador inicializado desde {args.init}")
    D = Discriminator().to(device)

    l1 = nn.L1Loss()
    perc = VGGPerceptualLoss().to(device)
    bce = nn.BCEWithLogitsLoss()
    optG = torch.optim.Adam(G.parameters(), lr=args.lr)
    optD = torch.optim.Adam(D.parameters(), lr=args.lr)

    history = {"epoch": [], "g_loss": [], "d_loss": [], "val_psnr": []}

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        gsum = dsum = 0.0
        for lr_b, hr_b in dl:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            sr = G(lr_b)
            real_lbl = torch.ones(hr_b.size(0), 1, device=device)
            fake_lbl = torch.zeros(hr_b.size(0), 1, device=device)

            # --- Discriminador ---
            optD.zero_grad()
            d_real = D(hr_b)
            d_fake = D(sr.detach().clamp(0, 1))
            d_loss = bce(d_real, real_lbl) + bce(d_fake, fake_lbl)
            d_loss.backward()
            optD.step()

            # --- Generador ---
            optG.zero_grad()
            content = l1(sr, hr_b) + args.perc_weight * perc(sr, hr_b)
            adv = bce(D(sr.clamp(0, 1)), real_lbl)
            g_loss = content + args.adv_weight * adv
            g_loss.backward()
            optG.step()

            gsum += g_loss.item(); dsum += d_loss.item()

        if epoch % args.val_every == 0 or epoch == args.epochs:
            vp, vs = eval_model(G, "Set5", args.scale, device)
            history["epoch"].append(epoch)
            history["g_loss"].append(gsum / len(dl))
            history["d_loss"].append(dsum / len(dl))
            history["val_psnr"].append(vp)
            print(f"ep {epoch:4d}  G {gsum/len(dl):.4f}  D {dsum/len(dl):.4f}  "
                  f"Set5 PSNR {vp:.3f} SSIM {vs:.4f}")

    # SRGAN: guardamos el generador FINAL (no el de mejor PSNR; el objetivo es perceptual)
    torch.save(G.state_dict(), os.path.join(out_dir, "best.pth"))
    results = {}
    for name in ["Set5", "Set14"]:
        p, s = eval_model(G, name, args.scale, device)
        results[name] = {"psnr": p, "ssim": s}
        print(f"FINAL {name} x{args.scale}: PSNR {p:.3f}  SSIM {s:.4f}")
    history["final"] = results
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
