# Eficiencia — parámetros, latencia, FPS


## Factor x2 (device=mps)

| Modelo | Params | ms/img | FPS | Set5 PSNR |
|---|---|---|---|---|
| fsrcnn_res_x2_l1 | 12,809 | 1.96 | 509.5 | 36.98 |
| fsrcnn_x2_l1 | 12,809 | 1.23 | 810.5 | 36.84 |

## Factor x4 (device=mps)

| Modelo | Params | ms/img | FPS | Set5 PSNR |
|---|---|---|---|---|
| fsrcnn_res_x4_l1 | 12,809 | 2.21 | 452.6 | 30.78 |
| fsrcnn_x4_l1 | 12,809 | 0.48 | 2099.9 | 30.62 |
