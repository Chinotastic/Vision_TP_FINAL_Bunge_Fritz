"""Figura de curvas de entrenamiento vs validación (pedido del tutor).

Genera figures/curvas_train_val_x2.png: dos paneles (pérdida de entrenamiento y
PSNR de validación vs época) para los tres modelos entrenados en IDENTICAS
condiciones sobre T91 a factor x2: EDSR, FSRCNN y FSRCNN-residual.

Sirve para dos cosas que pidio el tutor:
  - Entender por que EDSR (mucho mas grande) no le gana al FSRCNN-residual con un
    dataset chico: su PSNR de validacion es ruidoso/inestable y su mejor pico no
    supera al residual, pese a no bajar mas la perdida de entrenamiento.
  - Comparar FSRCNN vs FSRCNN-residual bajo las mismas condiciones: el residual
    tiene menor perdida y mayor PSNR en todo el recorrido -> aprender el residuo
    (HR - bicubic) es una optimizacion mas facil.

Uso: python3 src/make_curves.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path("outputs")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

# tag -> (etiqueta para la leyenda, color, ancho de linea)
MODELS = {
    "fsrcnn_x2_l1":     ("FSRCNN (12.8k)",          "#888888", 1.8),
    "edsr_x2_l1":       ("EDSR (777k)",             "#d1495b", 1.8),
    "fsrcnn_res_x2_l1": ("FSRCNN-residual (12.8k)", "#2e6fdb", 2.4),
}


def load(tag):
    h = json.load(open(OUT / tag / "history.json"))
    return h["epoch"], h["train_loss"], h["val_psnr"]


def main():
    fig, (ax_loss, ax_psnr) = plt.subplots(1, 2, figsize=(11, 4.2))

    for tag, (label, color, lw) in MODELS.items():
        ep, tl, vp = load(tag)
        ax_loss.plot(ep, tl, color=color, lw=lw, label=label, marker="o", ms=3)
        ax_psnr.plot(ep, vp, color=color, lw=lw, label=label, marker="o", ms=3)

    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoca")
    ax_loss.set_ylabel("Perdida de entrenamiento (L1, escala log)")
    ax_loss.set_title("Entrenamiento")
    ax_loss.grid(True, which="both", alpha=0.3)
    ax_loss.legend(fontsize=8)

    ax_psnr.set_xlabel("Epoca")
    ax_psnr.set_ylabel("PSNR de validacion en Set5 (dB)")
    ax_psnr.set_title("Validacion")
    ax_psnr.grid(True, alpha=0.3)
    ax_psnr.legend(fontsize=8, loc="lower right")
    # acercar el eje a la zona de interes (los modelos buenos viven en 35-37 dB)
    ax_psnr.set_ylim(33.5, 37.2)

    fig.suptitle("Curvas de entrenamiento y validacion (x2, entrenados en T91, misma config)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG / "curvas_train_val_x2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("guardado:", out)


if __name__ == "__main__":
    main()
