"""Arquitecturas de super-resolución (canal Y).

Interfaz unificada: TODOS los modelos reciben un parche LR del canal Y de forma
(B, 1, h, w) y devuelven la versión HR (B, 1, h*scale, w*scale). Así un solo
`train.py` / `data.py` sirve para las tres arquitecturas.

- SRCNN: agranda el LR con bicubic y lo "afila" con 3 convs. Opera en espacio HR (caro).
- FSRCNN: opera en LR (barato) y agranda al final con una deconvolución aprendida.
- FSRCNN-residual (NUESTRA MEJORA): FSRCNN que predice el RESIDUO sobre el bicubic
  del LR. Como el bicubic ya da ~90%, la red aprende solo la corrección -> converge
  más rápido y suele mejorar el PSNR. Cambio mínimo, gran argumento para el informe.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _bicubic_up(x, scale):
    """Upsample bicubic diferenciable (torch). Para residual base / entrada de SRCNN."""
    return F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)


class SRCNN(nn.Module):
    """Dong et al. 2015. 3 convoluciones sobre la imagen ya bicubic-upsampleada."""

    def __init__(self, scale, d=64, s=32):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(1, d, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(d, s, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(s, 1, 5, padding=2),
        )

    def forward(self, x):
        x = _bicubic_up(x, self.scale)
        return self.net(x)


class FSRCNN(nn.Module):
    """Dong et al. 2016. Feature -> shrink -> mapping(m) -> expand -> deconv.

    Defaults del paper: d=56, s=12, m=4. Deconv 9x9 stride=scale da salida exacta h*scale.
    Si `residual=True`, suma el bicubic del LR (variante propia, estilo VDSR).
    """

    def __init__(self, scale, d=56, s=12, m=4, residual=False):
        super().__init__()
        self.scale = scale
        self.residual = residual

        layers = [nn.Conv2d(1, d, 5, padding=2), nn.PReLU(d)]          # feature extraction
        layers += [nn.Conv2d(d, s, 1), nn.PReLU(s)]                    # shrinking
        for _ in range(m):                                            # mapping
            layers += [nn.Conv2d(s, s, 3, padding=1), nn.PReLU(s)]
        layers += [nn.Conv2d(s, d, 1), nn.PReLU(d)]                    # expanding
        self.body = nn.Sequential(*layers)
        self.deconv = nn.ConvTranspose2d(
            d, 1, 9, stride=scale, padding=4, output_padding=scale - 1
        )

    def forward(self, x):
        out = self.deconv(self.body(x))
        if self.residual:
            out = out + _bicubic_up(x, self.scale)
        return out


class _ResBlock(nn.Module):
    """Bloque residual estilo EDSR: conv-relu-conv + skip (sin BatchNorm)."""

    def __init__(self, nf, res_scale=1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class _Upsampler(nn.Sequential):
    """Upsample con sub-pixel convolution (PixelShuffle). Soporta 2, 3, 4."""

    def __init__(self, scale, nf):
        layers = []
        if scale in (2, 3):
            layers += [nn.Conv2d(nf, nf * scale * scale, 3, padding=1), nn.PixelShuffle(scale)]
        elif scale == 4:
            for _ in range(2):
                layers += [nn.Conv2d(nf, nf * 4, 3, padding=1), nn.PixelShuffle(2)]
        else:
            raise ValueError(f"escala no soportada: {scale}")
        super().__init__(*layers)


class EDSR(nn.Module):
    """EDSR-lite (Lim et al. 2017): head -> N resblocks -> upsample (pixel-shuffle) -> tail.

    Versión liviana: nf=64, n_resblocks=8. Opera en LR, sin BatchNorm. Más capacidad que
    FSRCNN -> mayor PSNR, a costa de más params/cómputo (interesante para el eje velocidad).
    """

    def __init__(self, scale, nf=64, n_resblocks=8, res_scale=1.0):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(1, nf, 3, padding=1)
        body = [_ResBlock(nf, res_scale) for _ in range(n_resblocks)]
        body += [nn.Conv2d(nf, nf, 3, padding=1)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(_Upsampler(scale, nf), nn.Conv2d(nf, 1, 3, padding=1))

    def forward(self, x):
        f = self.head(x)
        f = f + self.body(f)          # long skip connection
        return self.tail(f)


class Discriminator(nn.Module):
    """Discriminador estilo SRGAN/PatchGAN para canal Y (entrada HR-size, 1 canal).

    Pila de convs con stride que reduce la resolución, y AdaptiveAvgPool al final para ser
    agnóstico al tamaño del parche. Salida: un logit por imagen (real vs generada).
    """

    def __init__(self, nf=64):
        super().__init__()

        def block(cin, cout, stride):
            return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                    nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True)]

        layers = [nn.Conv2d(1, nf, 3, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        layers += block(nf, nf, 2)
        layers += block(nf, nf * 2, 1) + block(nf * 2, nf * 2, 2)
        layers += block(nf * 2, nf * 4, 1) + block(nf * 4, nf * 4, 2)
        layers += block(nf * 4, nf * 8, 1) + block(nf * 8, nf * 8, 2)
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(nf * 8, nf * 16), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nf * 16, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, scale):
    """name: 'srcnn' | 'fsrcnn' | 'fsrcnn_res' | 'edsr'."""
    name = name.lower()
    if name == "srcnn":
        return SRCNN(scale)
    if name == "fsrcnn":
        return FSRCNN(scale, residual=False)
    if name == "fsrcnn_res":
        return FSRCNN(scale, residual=True)
    if name == "edsr":
        return EDSR(scale)
    raise ValueError(name)
