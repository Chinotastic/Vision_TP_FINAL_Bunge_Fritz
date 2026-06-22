# Super-Resolución de Imagen Única — TPF Visión Artificial (I308, UdeSA)

Trabajo práctico final: reconstrucción de imágenes de alta resolución a partir de versiones
de baja resolución, implementando y comparando modelos de la familia SRCNN/FSRCNN, con una
mejora propia y un análisis de calidad vs. costo.

## Resultado principal

Nuestro mejor modelo es **FSRCNN-residual** (la red predice el residuo sobre el bicubic),
opcionalmente con **self-ensemble** en inferencia. Supera los targets de la consigna
(Set5: 34 / 31 / 29 dB para x2 / x3 / x4) con solo ~13k parámetros:

| Factor | Bicubic | FSRCNN-res | FSRCNN-res + self-ensemble | Target |
|---|---|---|---|---|
| x2 | 33.68 | 36.74 | **37.05** | 34 |
| x3 | 30.41 | 32.94 | **33.13** | 31 |
| x4 | 28.43 | 30.66 | **30.90** | 29 |

(PSNR en dB, canal Y, sobre Set5.) Comparado con EDSR (~70× más parámetros), el modelo
chico iguala o supera la calidad a una fracción del costo. Ver `outputs/comparacion_final.md`.

## Estructura

```
src/                 código fuente (modelos, entrenamiento, evaluación)
  data.py            dataset de patches (RGB->Y, generación de LR bicubic estilo MATLAB)
  baselines.py       upsampling nearest/bilinear/bicubic (bicubic MATLAB, a=-0.5)
  models.py          SRCNN, FSRCNN, FSRCNN-residual (mejora propia), EDSR, Discriminator
  losses.py          L1, L2, Charbonnier, pérdida perceptual VGG19
  train.py           loop de entrenamiento (un modelo / factor / pérdida)
  train_gan.py       variante SRGAN-lite (adversarial)
  eval.py            PSNR/SSIM sobre canal Y con shave; self-ensemble; loaders Set5/Set14/...
  metrics_check.py   gate: valida que el PSNR del bicubic coincide con los valores canónicos
  make_figures.py    genera tablas + paneles cualitativos + curvas de convergencia
  bench.py           eficiencia: parámetros, latencia (ms/img), FPS
notebooks/
  resultados.ipynb        notebook definitivo por fases: gate, tablas, figuras,
                          comparación calidad-vs-costo y demo (lee outputs/)
outputs/             pesos entrenados (best.pth), curvas, tablas (.md) y figuras crudas
figures/             figuras renombradas para el informe (curvas, cualitativos, eficiencia)
docs/                consignas del TP (PDFs)
datasets/            T91 (train) + Set5/Set14/BSD100/Urban100 (test) — no versionado
CONTEXT_LOG.md       bitácora de desarrollo por fases (qué hicimos / esperábamos / obtuvimos)
```

## Cómo correrlo

Requisitos: Python 3, PyTorch, OpenCV, scikit-image, numpy, matplotlib.

```bash
pip install torch torchvision opencv-python scikit-image numpy matplotlib
```

**Ver resultados** (no entrena, solo lee `outputs/`):

```bash
cd src
python make_figures.py --scales 2 3 4 --testsets Set5 Set14 BSD100 Urban100
python bench.py --scales 2 3 4
```

o abrir `notebooks/resultados.ipynb` (recorre todo el análisis por fases).

**Validar la métrica** (gate, debe pasar antes de confiar en cualquier número):

```bash
cd src && python metrics_check.py
```

**Reentrenar un modelo** (ejemplo: FSRCNN-residual x2):

```bash
cd src
python train.py --model fsrcnn_res --scale 2 --epochs 150 --batch 64
# otros: --model {srcnn,fsrcnn,fsrcnn_res,edsr}  --loss {l1,l2,charbonnier,perceptual}
# para entrenar EDSR con un dataset grande: --train-dir <DIV2K_HR> --lr 1e-4 --epochs 300
```

## Convención de evaluación

Las métricas se calculan **solo sobre el canal Y** (luminancia) en YCbCr estilo MATLAB
(rango 16–235), recortando un borde de `scale` píxeles (shave). Es la convención de los
papers de super-resolución; medir en RGB o sin shave da PSNR distinto y no reproduce los
valores canónicos. El bicubic usado es el de MATLAB (a=-0.5), validado en `metrics_check.py`.

## Modelos entrenados

- **Arquitecturas** (x2/x3/x4): SRCNN, FSRCNN, FSRCNN-residual, EDSR.
- **Ablación de pérdida** (FSRCNN x2): L1, L2, Charbonnier.
- **Percepción vs. distorsión** (EDSR x4): pérdida perceptual VGG, SRGAN.
- **Efecto del dataset** (EDSR x2/x3/x4): entrenado con T91 vs. DIV2K.
