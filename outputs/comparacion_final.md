# Comparación final — calidad vs costo

### Factor x2  (device=mps)

| Modelo | Dataset | Params | PSNR Set5 | SSIM | ms/img | FPS |
|---|---|---|---|---|---|---|
| FSRCNN-res (nuestro) | T91 | 12,809 | 36.74 | 0.9605 | 0.97 | 1032 |
| FSRCNN-res + self-ensemble | T91 | 12,809 | 37.05 | 0.9620 | 19.17 | 52 |
| EDSR (mismo dataset) | T91 | 776,705 | 36.57 | 0.9595 | 15.44 | 65 |
| EDSR (dataset grande) | DIV2K | 776,705 | 37.00 | 0.9611 | 14.67 | 68 |

### Factor x3  (device=mps)

| Modelo | Dataset | Params | PSNR Set5 | SSIM | ms/img | FPS |
|---|---|---|---|---|---|---|
| FSRCNN-res (nuestro) | T91 | 12,809 | 32.94 | 0.9224 | 0.35 | 2867 |
| FSRCNN-res + self-ensemble | T91 | 12,809 | 33.13 | 0.9248 | 15.79 | 63 |
| EDSR (mismo dataset) | T91 | 961,345 | 32.87 | 0.9210 | 8.55 | 117 |
| EDSR (dataset grande) | DIV2K | 961,345 | 33.01 | 0.9226 | 8.57 | 117 |

### Factor x4  (device=mps)

| Modelo | Dataset | Params | PSNR Set5 | SSIM | ms/img | FPS |
|---|---|---|---|---|---|---|
| FSRCNN-res (nuestro) | T91 | 12,809 | 30.66 | 0.8811 | 0.35 | 2887 |
| FSRCNN-res + self-ensemble | T91 | 12,809 | 30.90 | 0.8850 | 10.80 | 93 |
| EDSR (mismo dataset) | T91 | 924,417 | 30.32 | 0.8772 | 6.89 | 145 |
| EDSR (dataset grande) | DIV2K | 924,417 | 30.71 | 0.8810 | 6.90 | 145 |
