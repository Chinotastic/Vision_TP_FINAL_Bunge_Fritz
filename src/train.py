"""Entrenamiento de un modelo de super-resolución (canal Y).

Uso (local, smoke test rápido en CPU):
    python train.py --model fsrcnn --scale 2 --epochs 2 --patches-per-epoch 500

Uso (Kaggle, entrenamiento real en GPU):
    python train.py --model fsrcnn --scale 2 --epochs 200 --batch 64

Guarda en outputs/<tag>/: best.pth, history.json y curve.png. Reporta PSNR/SSIM
final en Set5 y Set14. Registra para el informe la curva (loss train + PSNR val).
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import SRPatchDataset
from models import build_model
from losses import build_loss, VGGPerceptualLoss
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
    ap.add_argument("--model", default="fsrcnn",
                    choices=["srcnn", "fsrcnn", "fsrcnn_res", "edsr"])
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--train-dir", default=os.path.join("..", "datasets", "T91"))
    ap.add_argument("--loss", default="l1",
                    choices=["l1", "l2", "charbonnier", "perceptual"])
    ap.add_argument("--perc-weight", type=float, default=0.05,
                    help="peso de la pérdida perceptual (cuando --loss perceptual)")
    ap.add_argument("--init", default=None,
                    help="pesos previos para inicializar (p.ej. afinar perceptual desde L1)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-size", type=int, default=24)
    ap.add_argument("--patches-per-epoch", type=int, default=8000)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    device = get_device()
    tag = args.tag or f"{args.model}_x{args.scale}_{args.loss}"
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{tag}] device={device}")

    ds = SRPatchDataset(args.train_dir, scale=args.scale, lr_size=args.lr_size,
                        patches_per_epoch=args.patches_per_epoch)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True)

    model = build_model(args.model, args.scale).to(device)
    if args.init and os.path.exists(args.init):
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"inicializado desde {args.init}")
    # Pérdida: distorsión simple, o perceptual = contenido L1 + peso * VGG
    if args.loss == "perceptual":
        content_crit = nn.L1Loss()
        perc_crit = VGGPerceptualLoss().to(device)

        def crit(sr, hr):
            return content_crit(sr, hr) + args.perc_weight * perc_crit(sr, hr)
    else:
        crit = build_loss(args.loss)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}  imgs_train={len(ds.pairs)}")

    history = {"epoch": [], "train_loss": [], "val_psnr": [], "params": n_params}
    best_psnr = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for lr_b, hr_b in dl:
            lr_b, hr_b = lr_b.to(device), hr_b.to(device)
            opt.zero_grad()
            loss = crit(model(lr_b), hr_b)
            loss.backward()
            opt.step()
            running += loss.item()
        train_loss = running / len(dl)

        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_psnr, val_ssim = eval_model(model, "Set5", args.scale, device)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_psnr"].append(val_psnr)
            tag_best = ""
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
                tag_best = "  <- best"
            print(f"ep {epoch:4d}  loss {train_loss:.5f}  Set5 PSNR {val_psnr:.3f} "
                  f"SSIM {val_ssim:.4f}{tag_best}")

    # Eval final con los mejores pesos
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pth"), map_location=device))
    results = {}
    for name in ["Set5", "Set14"]:
        p, s = eval_model(model, name, args.scale, device)
        results[name] = {"psnr": p, "ssim": s}
        print(f"FINAL {name} x{args.scale}: PSNR {p:.3f}  SSIM {s:.4f}")
    history["final"] = results
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Curva para el informe
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(history["epoch"], history["train_loss"], "C0-o", ms=3, label="train loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="C0")
        ax2 = ax1.twinx()
        ax2.plot(history["epoch"], history["val_psnr"], "C1-s", ms=3, label="val PSNR (Set5)")
        ax2.set_ylabel("Set5 PSNR (dB)", color="C1")
        plt.title(tag)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "curve.png"), dpi=120)
        print(f"curva -> {os.path.join(out_dir, 'curve.png')}")
    except Exception as e:
        print("no se pudo graficar:", e)


if __name__ == "__main__":
    main()
