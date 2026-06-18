# Eficiencia — parámetros, latencia, FPS


## Factor x2 (device=mps)

| Modelo | Params | ms/img | FPS | Set5 PSNR |
|---|---|---|---|---|
| edsr_x2_div2k | 776,705 | 16.52 | 60.5 | 37.00 |
| edsr_x2_l1 | 776,705 | 17.14 | 58.3 | 36.57 |
| fsrcnn_res_x2_l1 | 12,809 | 1.77 | 563.4 | 36.74 |
| fsrcnn_x2_charbonnier | 12,809 | 1.00 | 1002.4 | 36.21 |
| fsrcnn_x2_l1 | 12,809 | 1.00 | 1001.5 | 36.11 |
| fsrcnn_x2_l2 | 12,809 | 1.45 | 690.5 | 36.41 |
| srcnn_x2_l1 | 57,281 | 0.90 | 1113.4 | 36.37 |

## Factor x3 (device=mps)

| Modelo | Params | ms/img | FPS | Set5 PSNR |
|---|---|---|---|---|
| edsr_x3_div2k | 961,345 | 10.90 | 91.7 | 33.01 |
| edsr_x3_l1 | 961,345 | 10.40 | 96.2 | 32.87 |
| fsrcnn_res_x3_l1 | 12,809 | 1.84 | 543.5 | 32.94 |
| fsrcnn_x3_l1 | 12,809 | 0.64 | 1561.3 | 32.19 |

## Factor x4 (device=mps)

| Modelo | Params | ms/img | FPS | Set5 PSNR |
|---|---|---|---|---|
| edsr_x4_div2k | 924,417 | 8.06 | 124.1 | 30.71 |
| edsr_x4_gan | 924,417 | 7.75 | 129.1 | 29.06 |
| edsr_x4_l1 | 924,417 | 7.70 | 129.9 | 30.32 |
| edsr_x4_perceptual | 924,417 | 7.68 | 130.2 | 27.85 |
| fsrcnn_res_x4_l1 | 12,809 | 1.97 | 507.5 | 30.66 |
| fsrcnn_x4_l1 | 12,809 | 0.37 | 2690.5 | 30.09 |
| srcnn_x4_l1 | 57,281 | 0.13 | 7803.5 | 30.27 |
