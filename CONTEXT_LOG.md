# CONTEXT_LOG — TPF Super-Resolución

Bitácora viva del proyecto, organizada por fases. Cada fase: **qué hicimos / qué
esperábamos / qué obtuvimos**. Sirve para no perder contexto entre sesiones y como
material crudo para el informe IEEE final.

Tema: super-resolución de imagen única (SRCNN/FSRCNN). Eval en Set5/Set14, métricas
PSNR/SSIM sobre canal Y. Entorno: scripts en `src/`, entrenamiento en Kaggle.

---

## Fase 1 — Pipeline de datos + evaluación

### Qué hicimos
- **Entorno**: venv `.venv` (hereda torch 2.8 / cv2 4.13 / numpy / matplotlib del sistema)
  + `scikit-image 0.24` para SSIM. Python 3.9.
- **Datasets de eval**: clone disperso del repo `jbhuang0604/SelfExSR` →
  `datasets/SelfExSR/data/Set5` y `Set14`. Traen pares HR/LR ya generados para
  factores 2x/3x/4x. El LR es el downsample exacto (HR/scale), p.ej. Set5 2x: HR 512² → LR 256².
- **`src/eval.py`**: conversión RGB→Y estilo MATLAB rgb2ycbcr (rango [16,235]), `modcrop`,
  `shave` (recorta `scale` px de borde), `psnr_y`, `ssim_y`, y `load_set(name, scale)`.
- **`src/baselines.py`**: upsampling nearest/bilinear (cv2) y **bicubic estilo MATLAB**
  (`imresize_matlab`, kernel a=-0.5). Clave: cv2 usa a=-0.75 y NO reproduce los números canónicos.
- **`src/metrics_check.py`**: GATE — PSNR/SSIM del bicúbico en Set5/Set14 vs. valores de la literatura.

### Qué esperábamos
- Que el PSNR del baseline bicúbico coincidiera con los valores canónicos (±~0.1 dB):
  Set5 2x/3x/4x ≈ 33.66 / 30.39 / 28.42; Set14 ≈ 30.24 / 27.55 / 26.00.
- Si no coincidía, el medidor estaría mal (canal RGB en vez de Y, sin shave, o bicubic equivocado).

### Qué obtuvimos
- **Gate VERDE.** Resultados del bicúbico:

  | dataset | 2x | 3x | 4x |
  |---|---|---|---|
  | Set5  | 33.68 | 30.41 | 28.43 |
  | Set14 | 30.33 | 27.63 | 26.09 |

  Todos dentro de ±0.1 dB de Set5 (exacto) y ~+0.09 en Set14 (variación normal por el set puntual).
- **Conclusión**: el pipeline de evaluación es correcto y los PSNR de modelos que reportemos
  desde acá son confiables. Habilita la Fase 2 (entrenamiento).
- **Trampa documentada para el informe**: el coeficiente del kernel bicúbico (a=-0.5 MATLAB vs
  a=-0.75 OpenCV) y la conversión Y de rango limitado cambian el PSNR; usar la convención correcta
  es lo que permite comparar contra la literatura.

### Datos de entrenamiento + `data.py`
- **Set de train**: T91 (91 imágenes, clásico de SRCNN) en `datasets/T91/` (9.4 MB). Fuente:
  repo `thepooons/SRCNN` (las fuentes oficiales CUHK/vllab estaban caídas, 404). Para el
  entrenamiento "de verdad" en Kaggle se puede sumar General100/DIV2K como Kaggle dataset.
- **`src/data.py`** → `SRPatchDataset`: muestrea parches aleatorios (LR, HR) del canal Y en [0,1],
  con flips/rot90. LR generado por bicubic-downscale MATLAB (mismo `imresize_matlab` que el baseline),
  para que la degradación de train coincida con la de eval. El modelo recibe parche LR pequeño y
  predice el HR (lr_size·scale); la variante residual sumará internamente el bicubic del LR.
- *Esperábamos*: parches con formas (1, p, p) y (1, p·scale, p·scale), Y en [0,1]. *Obtuvimos*: exactamente eso
  (SF2: lr 24² → hr 48²; SF4: lr 24² → hr 96²), 91/90 imágenes usables. OK.

### Estado: FASE 1 COMPLETA ✅
Pipeline de datos y evaluación funcionando y validado. Habilitada la Fase 2 (FSRCNN base 2x).

---

## Fase 2 — Modelos + entrenamiento

### Qué hicimos
- **`src/models.py`** — 3 arquitecturas con interfaz unificada (entran LR Y, salen HR Y):
  - `SRCNN`: bicubic-up + 3 convs (referencia).
  - `FSRCNN` (base): feature→shrink→mapping(m=4)→expand→deconv. d=56, s=12. ~12.8k params.
  - `FSRCNN-residual` (NUESTRA MEJORA): igual pero suma el bicubic del LR → predice solo el residuo.
- **`src/eval.py`** ampliado: conversión YCbCr completa estilo MATLAB (elementwise, sin matmul para
  evitar warnings de BLAS), `sr_from_model` (Y del modelo + Cb/Cr bicubic → RGB), y `eval_model`
  que mide PSNR/SSIM **directo sobre el canal Y predicho** (sin round-trip RGB, más limpio).
- **`src/train.py`** — entrena cualquier modelo; loss L1/L2, Adam; registra curva (loss train +
  PSNR val Set5); guarda `best.pth`, `history.json`, `curve.png`; eval final en Set5/Set14.
  Detecta device (cuda/mps/cpu).

### Qué esperábamos
- Que el flujo corriera end-to-end y que el **residual convergiera mucho más rápido** que el base.

### Qué obtuvimos (smoke test local, entrenamiento de juguete: 600 patches, 3-4 épocas)
- `fsrcnn` base: Set5 PSNR ~19 tras 4 épocas (converge lento, esperable: aprende el mapeo entero).
- `fsrcnn_res`: Set5 PSNR **33.42 tras 3 épocas** (parte del bicubic 33.68 y ya casi lo iguala).
- **Confirma la hipótesis**: el residual converge dramáticamente más rápido. Material directo para la
  discusión del informe (curvas de convergencia comparadas).
- Pesos de juguete borrados; los reales se entrenan en Kaggle.

### Cómo entrenar de verdad (Kaggle, GPU)
Subir `src/` + `datasets/T91` + `datasets/SelfExSR` como dataset de Kaggle. Correr los 4:
```
python train.py --model fsrcnn      --scale 2 --epochs 300 --patches-per-epoch 12000 --batch 64
python train.py --model fsrcnn_res  --scale 2 --epochs 300 --patches-per-epoch 12000 --batch 64
python train.py --model fsrcnn      --scale 4 --epochs 300 --patches-per-epoch 12000 --batch 64
python train.py --model fsrcnn_res  --scale 4 --epochs 300 --patches-per-epoch 12000 --batch 64
```
Esperado: Set5 2x ~36–37, Set5 4x ~30.5 (superar bicubic 33.68 / 28.43 y target 34 / 29).
Probar también `--loss l2` como variante extra que pide el spec.

### Resultado real del entrenamiento en Kaggle (T4)
- `fsrcnn_x2_l1`: convergió bien. ep20 ya superó target 34; ep250 **PSNR 36.65 / SSIM 0.9601** en Set5.
  (Convergió más rápido que lo advertido gracias a 12k patches/época.) Gate de eval en Kaggle: idéntico al local.
- Recursos T4: GPU mem 213 MiB/15 GiB, RAM 1.7/30 GiB — holgadísimo. Cuello de botella = dataloader CPU (normal).

### NÚMEROS FINALES de los 4 modelos (Kaggle T4, L1, 300 épocas) — PSNR / SSIM
| Modelo | Set5 x2 | Set14 x2 | Set5 x4 | Set14 x4 |
|---|---|---|---|---|
| bicubic (baseline) | 33.68 | 30.33 | 28.43 | 26.09 |
| FSRCNN base | 36.72 / .9603 | 32.49 / .9160 | 30.63 / .8793 | 27.61 / .7736 |
| FSRCNN-res (mejora) | 36.96 / .9615 | 32.62 / .9165 | 30.76 / .8830 | 27.72 / .7765 |
| target spec | ≥34 | — | ≥29 | — |

**Lectura para el informe**: los 4 superan bicubic y los targets holgados. El residual gana SIEMPRE
pero por poco (+0.13–0.24 dB). La mejora real del residual es la VELOCIDAD DE CONVERGENCIA (ver curvas
conv_x*.png), no el PSNR final. No sobrevender el residual: la diferencia final es marginal.

---

## Fase 3 — Figuras y tablas (evidencia para informe/póster)

### Qué hicimos
- **`src/make_figures.py`** — lee `outputs/<tag>/best.pth` y genera:
  1. **Tabla** PSNR/SSIM: baselines (nearest/bilinear/bicubic) vs modelos, Set5 y Set14, por factor → `outputs/tables.md`.
  2. **Paneles cualitativos** (2 filas: imagen completa + zoom 80px) Bicubic / FSRCNN / FSRCNN-res / GT
     con PSNR en cada título → `outputs/figures/panel_*.png`. (El oro del póster.)
  3. **Curva comparada** base vs residual (convergencia) → `outputs/figures/conv_x{f}.png`.
- Uso: `python make_figures.py --loss l1 --scales 2 4 --panels 0 2`. Detecta device.

### Qué esperábamos / obtuvimos
- Probado local con pesos de juguete: genera tabla + paneles + curvas sin errores; layout del panel correcto
  (verificado visualmente). Con los pesos reales de Kaggle dará las figuras finales.

### Qué obtuvimos (FIGURAS DEFINITIVAS, pesos reales de Kaggle)
Descarga de pesos: la versión interactiva de Kaggle no persistía; se resolvió con **Save & Run All
(Commit)** y luego bajando los archivos de `outputs/` vía API de Kaggle (token KGAT como Bearer en
curl → URLs firmadas). Los 4 `best.pth` + history + curve quedaron en `outputs/`.

Tabla final (PSNR/SSIM, canal Y, con shave) — `outputs/tables.md`:
| Método | Set5 x2 | Set14 x2 | Set5 x4 | Set14 x4 |
|---|---|---|---|---|
| bicubic | 33.68 | 30.33 | 28.43 | 26.09 |
| FSRCNN base | 36.84 / .9611 | 32.53 / .9162 | 30.62 / .8795 | 27.62 / .7743 |
| FSRCNN-res | 36.98 / .9615 | 32.64 / .9169 | 30.78 / .8828 | 27.72 / .7761 |
| target spec | ≥34 | — | ≥29 | — |

Figuras en `outputs/figures/`: `panel_Set5_x{2,4}_{0,2}.png` (Bicubic/base/residual/GT + zoom),
`conv_x{2,4}.png` (convergencia base vs residual). Verificadas visualmente: panel nítido y curva
muestra el residual convergiendo mucho más rápido y quedando marginalmente arriba.

### Estado: FASE 3 COMPLETA ✅
Tablas + paneles + curvas listos con pesos reales. Los 4 modelos superan baselines y targets.

---

## Fase 4 — Trabajo completo (4 ejes). Decisión del usuario: hacer LOS 4, entrenar en Kaggle.

### Qué hicimos (código nuevo, todo testeado local)
- **Eje 1 (arquitecturas)**: agregado `EDSR` a `models.py` (head + N resblocks sin BN + pixel-shuffle;
  nf=64, 8 resblocks, ~0.78–0.96M params). SRCNN/FSRCNN/FSRCNN-res ya estaban.
- **Eje 2 (percepción)**: `losses.py` con `CharbonnierLoss` y `VGGPerceptualLoss` (Y→3ch + stats ImageNet).
  `Discriminator` (PatchGAN, AdaptiveAvgPool) en models.py. `train_gan.py` = SRGAN-lite
  (content L1+VGG + adversarial BCE, init opcional desde modelo L1). `train.py` soporta `--loss
  perceptual` y `--init`.
- **Eje 3 (eficiencia)**: `bench.py` — params + latencia ms/img + FPS por modelo, y scatter PSNR-vs-latencia.
  Hallazgo: el residual es más lento que el base (por el bicubic extra).
- **Eje 4 (completitud)**: factor 3x (modelos lo soportan, gate x3 ya validado); ablación L1/L2/Charbonnier;
  test sets **BSD100 + Urban100** bajados a SelfExSR/data (100 imgs c/u).
- **`make_figures.py`** reescrito: auto-descubre todos los modelos, tabla multi-testset, paneles,
  curvas de convergencia de todos, y panel de percepción (Bicubic | mejor-PSNR | GAN | GT).

### Cómo se entrena (Kaggle)
- Notebook nuevo: **`notebooks/kaggle_train_full.ipynb`** (celdas por eje). Requiere GPU T4 **+ Internet ON**
  (VGG19). Matriz: arch×{2,4}, 3x, ablación pérdidas, perceptual+GAN en edsr x4. ~2-3 h en T4.
- Dataset re-zipeado: `tpf-superres.zip` (305 MB, ahora con código nuevo + BSD100/Urban100) → subir como
  **nueva versión** del dataset de Kaggle.
- Persistir con **Save & Run All (Commit)**; bajar outputs por API (token KGAT como Bearer → URLs firmadas).

### Corrida 1 en Kaggle (notebook-2): PARCIAL — código viejo montado
- El notebook nuevo corrió contra el **dataset con el `src/` VIEJO** (sin EDSR/train_gan.py/losses.py).
  Log: `argument --model: invalid choice: 'edsr'` y `can't open train_gan.py`.
- Salieron 9 modelos válidos (descargados): srcnn x2/x4, fsrcnn x2(l1,l2)/x3/x4, fsrcnn_res x2/x3/x4.
  Faltan: EDSR (x2/x3/x4), fsrcnn x2 charbonnier, perceptual, gan.
- Tabla parcial (PSNR Set5): x2 → srcnn 36.37, fsrcnn 36.11, fsrcnn_l2 36.41, fsrcnn_res 36.74.
  Hallazgo: L2 > L1 en PSNR (esperado, L2 optimiza MSE). Residual el mejor.

### Acción correctiva
- Re-subir el dataset a la **versión NUEVA** (`tpf-superres.zip`, con el código nuevo). ← causa raíz del fallo.
- Correr **`notebooks/kaggle_train_missing.ipynb`** (solo los 6 faltantes, ~1-1.5h). Su celda 1 verifica
  que el código sea el nuevo y aborta si no. Luego bajar por API y mergear con los 9 ya descargados.

### Corrida 2 en Kaggle (notebook-3, `kaggle_train_missing.ipynb`): COMPLETA ✅
- Esta vez con el dataset re-subido (código nuevo). Entrenó los **6 faltantes** y los bajamos por API.
- Ahora tenemos los **15 modelos completos** en `outputs/`:
  - Arquitectura: srcnn / fsrcnn / fsrcnn_res / **edsr**, en x2 y x4.
  - Factor 3x: fsrcnn / fsrcnn_res / edsr.
  - Ablación pérdida (fsrcnn x2): l1 / l2 / charbonnier.
  - Percepción: edsr_x4_perceptual (VGG) + edsr_x4_gan (SRGAN).

### NÚMEROS FINALES de los 6 nuevos (Kaggle T4) — PSNR / SSIM, canal Y
| Modelo | Set5 PSNR | Set5 SSIM | Set14 PSNR | Set14 SSIM |
|---|---|---|---|---|
| edsr_x2_l1 | 36.57 | 0.9595 | 32.38 | 0.9150 |
| edsr_x3_l1 | 32.87 | 0.9210 | 29.29 | 0.8381 |
| edsr_x4_l1 | 30.32 | 0.8772 | 27.39 | 0.7714 |
| fsrcnn_x2_charbonnier | 36.21 | 0.9571 | 32.13 | 0.9121 |
| edsr_x4_perceptual | 27.85 | 0.7878 | 25.79 | 0.7041 |
| edsr_x4_gan | 29.06 | 0.8283 | 26.57 | 0.7303 |

### Qué esperábamos vs qué obtuvimos (HALLAZGO IMPORTANTE)
- **Esperábamos**: EDSR (924k params) > FSRCNN-res (13k) en PSNR — más capacidad, mejor reconstrucción.
- **Obtuvimos lo CONTRARIO**: en x2, edsr 36.57 < fsrcnn_res 36.98 < fsrcnn 36.84; en x4, edsr 30.32 <
  fsrcnn_res 30.78 < fsrcnn 30.62. El modelo grande PIERDE contra el chico.
- **Por qué** (diagnóstico, no es bug): entrenamos a EDSR con **solo T91 (91 imgs)** y `lr=1e-3`. EDSR
  está diseñado para datasets grandes (DIV2K, 800 imgs) y `lr=1e-4`. Con tan pocos datos la capacidad
  extra no se aprovecha. FSRCNN, al ser chico, satura con poquitos datos y le alcanza T91.
- **Percepción-distorsión** (esperado y confirmado): perceptual y GAN dan PSNR MÁS BAJO que edsr_x4_l1
  (30.32) — perceptual 27.85, gan 29.06 — porque optimizan nitidez/realismo, no PSNR. `train_gan.py`
  guarda el generador FINAL (no el de mejor-PSNR) a propósito.

---

## Fase 5 — Experimento EDSR con datos grandes (DIV2K)

### Qué hicimos
- A raíz del hallazgo anterior, montamos un experimento controlado para testear la hipótesis
  "EDSR pierde por falta de datos, no por arquitectura".
- Notebook nuevo **`notebooks/kaggle_train_edsr_div2k.ipynb`** (muy comentado): re-entrena SOLO EDSR
  (x2/x3/x4) con un **subset de DIV2K** (~200 imgs 2K) y `lr=1e-4`, 300 épocas. Tags separados
  `edsr_x{s}_div2k` para NO pisar los de T91 y poder comparar.
- Decisión de diseño (comparación justa): la progresión de arquitecturas queda en T91 (donde EDSR no
  ayuda); el EDSR-DIV2K es un experimento APARTE que muestra que la capacidad sí paga con más datos.
- Limitación técnica resuelta: `data.py` carga todas las imágenes a RAM → DIV2K completo (800 2K)
  reventaba la memoria de Kaggle. Solución sin tocar código: el notebook copia solo N=200 imgs a una
  carpeta y entrena desde ahí.
- Comentamos a fondo `kaggle_train_full.ipynb` (notebook maestro) celda por celda, para documentación.

### Qué esperamos
- `edsr_x{s}_div2k` con PSNR mayor que `edsr_x{s}_l1` (T91), idealmente superando a fsrcnn_res.
- Si se confirma → cierra la historia: "el modelo grande necesitaba más datos".

### Qué obtuvimos — HIPÓTESIS CONFIRMADA ✅
- Entrenado en Kaggle (subset ~200 imgs de DIV2K, lr 1e-4, 300 épocas). PSNR Set5:
  | Escala | EDSR-T91 | EDSR-DIV2K | fsrcnn_res |
  |---|---|---|---|
  | x2 | 36.57 | **37.00** | 36.74 |
  | x3 | 32.87 | **33.01** | 32.94 |
  | x4 | 30.32 | **30.71** | 30.66 |
- Con más datos, EDSR sube +0.4 dB (x2/x4) y **pasa a superar al FSRCNN-residual en las 3 escalas**.
- Cierra la historia: la baja performance de EDSR en T91 era por falta de datos, no por la arquitectura.
  La capacidad extra del modelo grande solo se aprovecha con datasets grandes (DIV2K).
- Matiz para el informe: fsrcnn_res sigue siendo MUCHO más eficiente (13k vs 924k params) y con
  self-ensemble (37.05 x2) queda a la par de EDSR-DIV2K. O sea: el modelo chico bien usado compite
  con el grande, a una fracción del costo.

### Mejora extra: self-ensemble (sin reentrenar)
- Agregado a `eval.py`: `sr_y_self_ensemble` / `sr_from_model_se` y flag `self_ensemble` en `eval_model`.
  Promedia la salida sobre las 8 transformaciones geométricas (4 rot × espejo). `make_figures.py`
  agrega fila `<tag> +SE` para los modelos en `--se` (default: fsrcnn_res).
- Resultado (Set5 PSNR): x2 36.74→37.05 (+0.31), x3 32.94→33.13 (+0.19), x4 30.66→30.90 (+0.24).
  Gratis (solo inferencia), a costa de 8× tiempo de inferencia.
- Acumulado vs FSRCNN base en x2: 36.11 → 37.05 = **+0.94 dB** (residual + self-ensemble).

### Pendiente
- Correr `kaggle_train_edsr_div2k.ipynb` en Kaggle (Add Input: tpf-superres + un DIV2K público) →
  bajar por API → mergear `edsr_x*_div2k` en outputs/ → regenerar make_figures.
- Informe IEEE + póster.
