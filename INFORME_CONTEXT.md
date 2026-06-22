# Contexto para redactar el informe (IEEE, LaTeX)

> Documento de contexto para un LLM que escribe el informe final en **formato IEEE
> Conference Proceedings (LaTeX)**. Contiene: (1) las cláusulas obligatorias de la cátedra,
> (2) todo el contenido del trabajo (método, datos, resultados), (3) las figuras disponibles
> con su nombre de archivo y qué representan, y (4) avisos sobre comparaciones que convendría
> agregar. **No inventar números**: usar los de este documento y los de `outputs/*.md`.

---

## 1. Requisitos OBLIGATORIOS del informe (de la consigna)

- **Formato:** IEEE Conference Proceedings, **máximo 8 páginas A4**.
- **Mínimo 5 secciones** sugeridas: Introducción, Método, Conjuntos de datos, Resultados, Discusión.
- **Explicar los métodos**: detallar las arquitecturas de redes neuronales usadas.
- **Detallar el/los conjuntos de datos** sobre los que se trabajó (con enlaces/referencias).
- **Exponer métricas de precisión** (PSNR/SSIM) **e imágenes de los resultados**.
- Si se usan implementaciones existentes, **explicar fuentes, detalles y particularidades**.
- **Proponer al menos UNA modificación/mejora propia** (en este trabajo: FSRCNN-residual + self-ensemble).
- El entregable global incluye además: código fuente, enlaces a datasets y póster (el póster va aparte).

Checklist de cumplimiento (ya cubierto por el material):
- [x] ≥2 factores de escala → tenemos x2, x3, x4.
- [x] ≥2 variantes por factor → SRCNN/FSRCNN/FSRCNN-res/EDSR + ablaciones.
- [x] Evaluación en Set5/Set14 (+ BSD100/Urban100) con PSNR y SSIM sobre canal Y.
- [x] Superar baselines (nearest/bilinear/bicubic) y targets (Set5 34/31/29).
- [x] Curvas de entrenamiento.
- [x] Resultados cualitativos (imágenes).
- [x] Mejora propia.

---

## 2. Identificación del trabajo

- **Tema:** Super-Resolución de Imagen Única (Single Image Super-Resolution, SISR).
- **Materia:** I308 Visión Artificial, Universidad de San Andrés.
- **Autores:** Axel Fritz, [Bunge] — grupo 6. (Completar nombre del compañero e emails.)
- **Título sugerido (evitar genéricos):** algo que refleje la extensión, p.ej.
  *"Super-resolución eficiente: ¿alcanza un modelo chico? FSRCNN-residual + self-ensemble vs EDSR"*.

---

## 3. Problema

Dada una imagen de baja resolución (LR), reconstruir su versión de alta resolución (HR).
Es un problema mal condicionado (se perdió información al achicar). Se entrena con pares (LR, HR)
generados achicando imágenes HR con bicubic estilo MATLAB, y se evalúa recuperando la HR.

---

## 4. Conjuntos de datos (sección "Conjuntos de datos")

**Entrenamiento:**
- **T91** (91 imágenes) — set canónico de SRCNN/FSRCNN. De cada imagen se extraen 8000 patches
  aleatorios por época, con data augmentation (flips + rot90). Es el set principal.
- **DIV2K** (subset ~200 imágenes HR 2K) — usado SOLO para reentrenar EDSR, que necesita más datos.
  Referencia: DIV2K (NTIRE 2017). En Kaggle: `soumikrakshit/div2k-high-resolution-images`.

**Evaluación (test):** Set5, Set14, BSD100, Urban100 — versiones LR/HR del repositorio
**SelfExSR de Jia-Bin Huang** (https://github.com/jbhuang0604/SelfExSR). Factores 2/3/4
(Urban100 no provee x3).

**Convención de evaluación (importante, va en Método o Datos):** métricas calculadas SOLO sobre
el **canal Y** (luminancia) en YCbCr estilo MATLAB (rango 16–235); se recorta un borde de `scale`
píxeles (shave); el bicubic usado es el de MATLAB (a=−0.5). Validado en `metrics_check.py` contra
los valores canónicos publicados (gate de la métrica).

---

## 5. Método (sección "Método")

**Arquitecturas (de menor a mayor capacidad):**
- **SRCNN** — 3 convoluciones sobre la imagen ya interpolada (bicubic). Opera en HR (lento).
- **FSRCNN** — opera en LR y agranda al final con deconvolución. Rápido. ~13k parámetros. *Base.*
- **FSRCNN-residual** *(MEJORA PROPIA)* — predice el **residuo** sobre el bicubic:
  `salida = bicubic(LR) + red(LR)`. La red solo aprende a corregir el bicubic → converge mejor y
  sube PSNR. Mismo tamaño que FSRCNN (~13k params). **Es el mejor modelo del trabajo.**
- **EDSR** — red profunda con ResBlocks (sin BatchNorm) + PixelShuffle. ~0.78–0.96M parámetros.

**Funciones de pérdida** (ablación): L1, L2 (MSE), Charbonnier; y para percepción:
perceptual (VGG19) y SRGAN (adversarial).

**Mejora propia (2 componentes):**
1. **Conexión residual** sobre el bicubic (arriba).
2. **Self-ensemble** en inferencia: se super-resuelve la imagen en sus 8 orientaciones
   (4 rotaciones × espejo), se deshacen las transformaciones y se promedia. Cancela errores →
   +0.2–0.3 dB sin reentrenar, a costa de ~8× el tiempo de inferencia.

**Hiperparámetros de entrenamiento:** optimizador Adam; lr 1e-3 (1e-4 para EDSR-DIV2K);
150 épocas (300 para EDSR-DIV2K; 120 para perceptual/GAN); batch 64 (16 para EDSR);
patch LR 24×24; 8000 patches/época. GPU Kaggle T4.

---

## 6. Resultados (sección "Resultados") — números clave

PSNR (dB) en **Set5**, canal Y. Tablas completas (todos los testsets/SSIM) en `outputs/tables.md`.

| Factor | Bicubic (baseline) | SRCNN | FSRCNN | FSRCNN-res | **FSRCNN-res + SE** | EDSR (T91) | EDSR (DIV2K) | Target |
|---|---|---|---|---|---|---|---|---|
| x2 | 33.68 | 36.37 | 36.11 | 36.74 | **37.05** | 36.57 | 37.00 | 34 |
| x3 | 30.41 | — | 32.19 | 32.94 | **33.13** | 32.87 | 33.01 | 31 |
| x4 | 28.43 | 30.27 | 30.09 | 30.66 | **30.90** | 30.32 | 30.71 | 29 |

Ablación de pérdida (FSRCNN x2, Set5): L1 36.11 · L2 36.41 · Charbonnier 36.21.
Percepción (EDSR x4, Set5): perceptual 27.85 · SRGAN 29.06 (PSNR menor a propósito).

Eficiencia (Set5, device=mps), de `outputs/bench.md` y `outputs/comparacion_final.md`:
- FSRCNN-res: ~13k params; ~0.9 ms/img (x2).
- FSRCNN-res + SE: ~13k params; ~21 ms/img (x2) — 8 pasadas.
- EDSR: ~0.78–0.92M params; ~15–18 ms/img (x2).

**Headline:** la mejora propia (residual + self-ensemble) lleva el FSRCNN base de 36.11 a 37.05 dB
en x2 (**+0.94 dB**), superando a EDSR (70× más parámetros) en las tres escalas.

---

## 7. Figuras disponibles (carpeta `figures/`)

Usar estos nombres exactos al insertar en LaTeX. Sugerencia de caption entre comillas.

| Archivo | Qué representa | Sección | Caption sugerido |
|---|---|---|---|
| `figures/convergencia_x2.png` (y `_x3`, `_x4`) | PSNR de validación (Set5) vs época, todos los modelos del factor | Resultados | "Curvas de convergencia (PSNR Set5) para el factor ×N." |
| `figures/curvas_train_val_x2.png` | 2 paneles (pérdida de train L1 + PSNR de val) vs época para EDSR/FSRCNN/FSRCNN-res en T91, misma config. EDSR ruidoso e inestable y sin ganarle al residual pese a 60× params; el residual con menor pérdida y mayor PSNR todo el recorrido | Discusión | "Curvas de entrenamiento y validación (×2, T91): el residual domina estable, EDSR es inestable." Responde al pedido del tutor (curvas train+val + ablación FSRCNN vs residual en idénticas condiciones). |
| `figures/cualitativo_Set5_x2_ej1.png` … `_x4_ej2` | Comparación visual LR \| bicubic \| modelos \| GT, con PSNR (2 ejemplos por factor) | Resultados | "Resultados cualitativos en Set5 ×N: bicubic vs modelos vs ground truth." |
| `figures/percepcion_distorsion_x4.png` | Bicubic \| mejor-PSNR \| SRGAN \| GT — trade-off | Discusión | "Trade-off distorsión–percepción (×4): el modelo GAN es más nítido pese a menor PSNR." |
| `figures/eficiencia_x2.png` (y `_x3`, `_x4`) | Scatter calidad (PSNR) vs latencia (ms/img) | Resultados/Discusión | "Calidad vs costo computacional por modelo (×N)." |

Tablas en Markdown listas para transcribir a LaTeX:
- `outputs/tables.md` — PSNR/SSIM completo (baselines + modelos, 4 testsets).
- `outputs/bench.md` — params / latencia / FPS.
- `outputs/comparacion_final.md` — modelo chico vs EDSR (calidad + costo).

---

## 8. Discusión (sección "Discusión") — hilos narrativos

1. **Mejora propia funciona:** la conexión residual sube PSNR de forma consistente (gana en las 3
   escalas) sin costo extra de parámetros; el self-ensemble agrega otro empujón gratis.
2. **Capacidad vs datos:** EDSR (modelo grande) con T91 PIERDE contra el FSRCNN-residual; solo al
   reentrenarlo con DIV2K supera al chico básico. La capacidad extra requiere datos.
3. **Calidad vs costo:** el FSRCNN-res + self-ensemble iguala/supera a EDSR-DIV2K con ~70× menos
   parámetros. Trade-off: la versión sin SE es la más rápida; con SE gana calidad pero ~8× más lenta.
4. **Distorsión vs percepción:** las pérdidas perceptual/SRGAN bajan PSNR a propósito a cambio de
   nitidez/realismo — ilustra que PSNR no captura calidad perceptual.

**Trabajo a futuro (para Discusión/póster):** entrenar todos los modelos con DIV2K para una
comparación pareja; probar self-ensemble sobre EDSR; modelos más profundos; métricas perceptuales (LPIPS).

---

## 9. AVISOS — comparaciones/correlaciones que convendría agregar

Estas NO están hechas todavía. Si el informe las necesita para ser más sólido, avisar a los autores
para generarlas (la mayoría son baratas con el código actual):

1. **Comparación contra los números publicados** de SRCNN/FSRCNN/EDSR (tabla "nuestro vs paper").
   Daría credibilidad de reproducción. **No la tenemos** — habría que recopilar los valores de los papers.
2. **Paneles cualitativos en Urban100** (el testset más difícil, con líneas/estructuras). Hoy los
   paneles son solo de Set5. Generables con `make_figures.py --testsets Urban100`.
3. **Panel cualitativo que muestre el efecto del self-ensemble** (con SE vs sin SE, lado a lado).
   No existe; habría que agregar una figura.
4. **Correlación PSNR vs SSIM** o tabla con ambas para todos los modelos (hoy SSIM está en tables.md
   pero no se discute). Útil si se quiere argumentar consistencia entre métricas.
5. **Resultados de EDSR-DIV2K en Set14/BSD100/Urban100** (hoy el foco de la comparación es Set5).
   Ya están en `outputs/tables.md`; faltaría destacarlos si se quiere generalizar la conclusión.
6. **Ablación de self-ensemble por nº de transformaciones** (2/4/8) para justificar el costo. No hecha.

> Si el redactor decide que alguna de estas es necesaria, debe pedirla explícitamente; varias se
> generan en minutos reusando `src/make_figures.py` y `src/eval.py`.
