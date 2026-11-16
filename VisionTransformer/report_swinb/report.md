# Aggregated results — swinb

Source: per-run result JSONs (single source of truth). Memory in GiB (1024³ B), sizes in MiB. Peak memory = peak **allocated** by PyTorch during train+val, fp32, batch 64 (excludes CUDA context). `peak − static` = peak allocated minus (params + grads + AdamW state); it is not pure activation memory (for full FT it understates activations by about the gradient size). Per-batch timings use epochs ≥ 2. Every metric column is mean ± std over seeds (sample std, n−1); medians/min/max are in stats.json.

## ⚠ Assertion failures

- runs come from 2 different source versions ['bad188ad14e4', 'be09787d03af']; do not pool them without checking what changed

## Table 1 — Full fine-tuning, learning-rate sweep

| experiment | lr | impl | n | test acc % | best ep | stop ep | epoch train s | fwd ms/batch | bwd ms/batch | opt ms/batch | peak alloc GiB | peak − static GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| swinb_baseline_lr1e-5 | 1e-05 | library | 5 | 91.26 ± 0.23 | 3.2 ± 0.4 | 8.2 ± 0.4 | 485.3 ± 0.3 | 245.76 ± 0.02 | 435.83 ± 0.23 | 6.51 ± 0.24 | 14.121 ± 0.000 | 12.827 ± 0.000 |
| swinb_baseline_lr1e-4 | 0.0001 | library | 5 | 88.65 ± 0.44 | 2.2 ± 0.4 | 7.2 ± 0.4 | 485.7 ± 0.7 | 245.80 ± 0.10 | 436.01 ± 0.44 | 6.63 ± 0.35 | 14.121 ± 0.000 | 12.827 ± 0.000 |

## Table 2 — LoRA configurations (lr 1e-4)

| experiment | lr | impl | n | test acc % | best ep | stop ep | epoch train s | fwd ms/batch | bwd ms/batch | opt ms/batch | peak alloc GiB | peak − static GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| swinb_lora_attn_r8 | 0.0001 | library | 5 | 91.74 ± 0.31 | 7.4 ± 2.1 | 12.4 ± 2.1 | 378.0 ± 0.1 | 263.39 ± 0.06 | 271.06 ± 0.06 | 1.36 ± 0.05 | 9.549 ± 0.000 | 9.215 ± 0.000 |
| swinb_lora_mlp_r8 | 0.0001 | library | 5 | 91.69 ± 0.20 | 4.4 ± 0.5 | 9.4 ± 0.5 | 385.3 ± 0.2 | 273.39 ± 0.02 | 271.60 ± 0.14 | 1.08 ± 0.08 | 12.007 ± 0.000 | 11.667 ± 0.000 |
| swinb_lora_attn_mlp_r8 | 0.0001 | library | 5 | 91.70 ± 0.22 | 3.8 ± 0.8 | 8.8 ± 0.8 | 428.5 ± 0.4 | 289.91 ± 0.17 | 315.41 ± 0.38 | 2.16 ± 0.13 | 13.290 ± 0.000 | 12.942 ± 0.000 |
| swinb_lora_attn_r8_L0-11 | 0.0001 | library | 5 | 91.36 ± 0.20 | 12.8 ± 1.9 | 17.8 ± 1.9 | 364.5 ± 0.1 | 256.85 ± 0.07 | 258.44 ± 0.10 | 1.20 ± 0.04 | 9.284 ± 0.000 | 8.955 ± 0.000 |
| swinb_lora_attn_r8_L12-23 | 0.0001 | library | 5 | 90.57 ± 0.19 | 7.2 ± 1.6 | 12.2 ± 1.6 | 262.3 ± 0.1 | 253.03 ± 0.03 | 117.15 ± 0.08 | 1.12 ± 0.08 | 3.707 ± 0.000 | 3.376 ± 0.000 |
| swinb_lora_attn_r8_L0_2_4_6_8_10_12_14_16_18_20_22 | 0.0001 | library | 5 | 91.72 ± 0.19 | 9.6 ± 2.2 | 14.6 ± 2.2 | 360.0 ± 0.2 | 254.88 ± 0.27 | 253.92 ± 0.12 | 1.29 ± 0.10 | 9.156 ± 0.000 | 8.827 ± 0.000 |
| swinb_lora_attn_r8_L1_3_5_7_9_11_13_15_17_19_21_23 | 0.0001 | library | 5 | 91.82 ± 0.29 | 8.6 ± 1.9 | 13.6 ± 1.9 | 348.7 ± 0.1 | 255.67 ± 0.01 | 237.18 ± 0.07 | 1.19 ± 0.03 | 8.180 ± 0.000 | 7.851 ± 0.000 |

## Scratch − library, paired per seed

Each cell: mean difference ± std of the per-seed differences (95% t-CI half-width, n seeds). A CI that contains 0 means **no detectable difference at this sample size**; it is not evidence of equivalence. With n=3 the t critical value is 4.30, so intervals are wide.

| experiment | Δ test acc (pp) | Δ stop epoch | Δ epoch train (s) | Δ fwd (ms) | Δ bwd (ms) | Δ peak alloc (GiB) |
|---|---|---|---|---|---|---|

### Per-seed accuracy differences

| experiment | seed | library acc % | scratch acc % | scratch − library (pp) |
|---|---|---|---|---|

## LoRA − full fine-tuning (lr 1e-4 baseline), paired per seed, same implementation

| experiment | impl | Δ test acc (pp) | Δ peak alloc (GiB) | Δ epoch train (s) | Δ fwd (ms) | Δ bwd (ms) | Δ opt (ms) |
|---|---|---|---|---|---|---|---|
| swinb_lora_attn_mlp_r8 | library | +3.05 ± 0.29 (CI95 ±0.36, n=5) | -0.83 ± 0.00 (CI95 ±0.00, n=5) | -57.2 ± 1.0 (CI95 ±1.3, n=5) | +44.11 ± 0.22 (CI95 ±0.27, n=5) | -120.60 ± 0.73 (CI95 ±0.90, n=5) | -4.47 ± 0.46 (CI95 ±0.57, n=5) |
| swinb_lora_attn_r8 | library | +3.10 ± 0.32 (CI95 ±0.40, n=5) | -4.57 ± 0.00 (CI95 ±0.00, n=5) | -107.6 ± 0.6 (CI95 ±0.7, n=5) | +17.59 ± 0.08 (CI95 ±0.10, n=5) | -164.95 ± 0.38 (CI95 ±0.48, n=5) | -5.27 ± 0.30 (CI95 ±0.38, n=5) |
| swinb_lora_attn_r8_L0-11 | library | +2.72 ± 0.28 (CI95 ±0.35, n=5) | -4.84 ± 0.00 (CI95 ±0.00, n=5) | -121.2 ± 0.6 (CI95 ±0.8, n=5) | +11.05 ± 0.09 (CI95 ±0.12, n=5) | -177.57 ± 0.40 (CI95 ±0.49, n=5) | -5.43 ± 0.34 (CI95 ±0.42, n=5) |
| swinb_lora_attn_r8_L0_2_4_6_8_10_12_14_16_18_20_22 | library | +3.07 ± 0.48 (CI95 ±0.60, n=5) | -4.96 ± 0.00 (CI95 ±0.00, n=5) | -125.6 ± 0.9 (CI95 ±1.1, n=5) | +9.08 ± 0.34 (CI95 ±0.42, n=5) | -182.09 ± 0.55 (CI95 ±0.68, n=5) | -5.34 ± 0.38 (CI95 ±0.47, n=5) |
| swinb_lora_attn_r8_L12-23 | library | +1.92 ± 0.42 (CI95 ±0.53, n=5) | -10.41 ± 0.00 (CI95 ±0.00, n=5) | -223.4 ± 0.6 (CI95 ±0.8, n=5) | +7.23 ± 0.08 (CI95 ±0.09, n=5) | -318.86 ± 0.38 (CI95 ±0.48, n=5) | -5.51 ± 0.33 (CI95 ±0.41, n=5) |
| swinb_lora_attn_r8_L1_3_5_7_9_11_13_15_17_19_21_23 | library | +3.18 ± 0.52 (CI95 ±0.64, n=5) | -5.94 ± 0.00 (CI95 ±0.00, n=5) | -137.0 ± 0.7 (CI95 ±0.9, n=5) | +9.86 ± 0.09 (CI95 ±0.11, n=5) | -198.83 ± 0.47 (CI95 ±0.58, n=5) | -5.44 ± 0.37 (CI95 ±0.46, n=5) |
| swinb_lora_mlp_r8 | library | +3.05 ± 0.29 (CI95 ±0.36, n=5) | -2.11 ± 0.00 (CI95 ±0.00, n=5) | -100.4 ± 0.7 (CI95 ±0.9, n=5) | +27.59 ± 0.11 (CI95 ±0.13, n=5) | -164.41 ± 0.48 (CI95 ±0.60, n=5) | -5.55 ± 0.38 (CI95 ±0.48, n=5) |

## Trainable parameters and per-task artifact size

Artifact = head + adapters (LoRA) or the full weights (baseline), fp32, the same for both implementations.

| experiment | trainable elements | artifact MiB (analytic, fp32) | measured adapter export MiB |
|---|---|---|---|
| swinb_baseline_lr1e-4 | 86,845,724 | 331.29 | - |
| swinb_baseline_lr1e-5 | 86,845,724 | 331.29 | - |
| swinb_lora_attn_mlp_r8 | 1,642,596 | 6.27 | 6.34 |
| swinb_lora_attn_r8 | 680,036 | 2.59 | 2.64 |
| swinb_lora_attn_r8_L0-11 | 335,972 | 1.28 | 1.30 |
| swinb_lora_attn_r8_L0_2_4_6_8_10_12_14_16_18_20_22 | 391,268 | 1.49 | 1.52 |
| swinb_lora_attn_r8_L12-23 | 446,564 | 1.70 | 1.73 |
| swinb_lora_attn_r8_L1_3_5_7_9_11_13_15_17_19_21_23 | 391,268 | 1.49 | 1.52 |
| swinb_lora_mlp_r8 | 1,065,060 | 4.06 | 4.09 |

### Profiler acceptance checks (85 checks: 84 pass, 1 warn, 0 fail)

The profiler tables below count as evidence only if these hold. They compare profile_memory.py against the independent in-loop measurements and against analytic byte counts.

| scope | check | status | detail |
|---|---|---|---|
| swinb_baseline_lr1e-4/library | profiler runtime overhead agrees with the in-loop probe's (same GPU runtime state) | WARN | 9.1 vs 18.1 MiB |


### Inference memory and latency (fresh process, no training; measured by profile_memory.py)

| experiment | impl | variant | params | resident weights GiB | b1 peak alloc GiB | b1 p50 ms | b64 peak alloc GiB | b64 p50 ms |
|---|---|---|---|---|---|---|---|---|
| swinb_baseline_lr1e-4 | library | fp32_unmerged | 86,845,724 | 0.3329 | 0.3577 | 8.70 | 1.805 | 243.0 |
| swinb_lora_attn_mlp_r8 | library | fp32_unmerged | 88,488,320 | 0.3390 | 0.3690 | 13.89 | 2.194 | 287.6 |
| swinb_lora_attn_mlp_r8 | library | fp32_merged | 86,845,724 | 0.3329 | 0.3574 | 8.80 | 1.805 | 243.5 |
| swinb_lora_attn_r8 | library | fp32_unmerged | 87,525,760 | 0.3354 | 0.3602 | 12.22 | 1.807 | 260.9 |
| swinb_lora_attn_r8 | library | fp32_merged | 86,845,724 | 0.3329 | 0.3574 | 8.93 | 1.805 | 243.5 |
| swinb_lora_attn_r8_L0-11 | library | fp32_unmerged | 87,181,696 | 0.3341 | 0.3589 | 10.59 | 1.806 | 254.6 |
| swinb_lora_attn_r8_L0-11 | library | fp32_merged | 86,845,724 | 0.3329 | 0.3574 | 9.21 | 1.805 | 243.7 |
| swinb_lora_attn_r8_L0_2_4_6_8_10_12_14_16_18_20_22 | library | fp32_unmerged | 87,236,992 | 0.3343 | 0.3592 | 10.31 | 1.806 | 252.5 |
| swinb_lora_attn_r8_L0_2_4_6_8_10_12_14_16_18_20_22 | library | fp32_merged | 86,845,724 | 0.3329 | 0.3574 | 9.22 | 1.805 | 243.5 |
| swinb_lora_attn_r8_L12-23 | library | fp32_unmerged | 87,292,288 | 0.3345 | 0.3594 | 10.24 | 1.807 | 250.2 |
| swinb_lora_attn_r8_L12-23 | library | fp32_merged | 86,845,724 | 0.3329 | 0.3574 | 9.23 | 1.805 | 243.6 |
| swinb_lora_attn_r8_L1_3_5_7_9_11_13_15_17_19_21_23 | library | fp32_unmerged | 87,236,992 | 0.3343 | 0.3592 | 10.32 | 1.806 | 252.7 |
| swinb_lora_attn_r8_L1_3_5_7_9_11_13_15_17_19_21_23 | library | fp32_merged | 86,845,724 | 0.3329 | 0.3574 | 9.27 | 1.805 | 243.8 |
| swinb_lora_mlp_r8 | library | fp32_unmerged | 87,910,784 | 0.3368 | 0.3669 | 11.56 | 2.192 | 270.4 |
| swinb_lora_mlp_r8 | library | fp32_merged | 86,845,724 | 0.3329 | 0.3574 | 9.27 | 1.805 | 243.8 |

### Training-step decomposition at batch 64 (measured by profile_memory.py)

| experiment | impl | resident before fwd GiB | activation GiB | grads GiB | optimizer transient GiB | peak GiB |
|---|---|---|---|---|---|---|
| swinb_baseline_lr1e-4 | library | 1.029 | 13.004 | 0.3238 | 0.324 | 14.120 |
| swinb_lora_attn_mlp_r8 | library | 0.395 | 12.739 | 0.0061 | 0.006 | 13.290 |
| swinb_lora_attn_r8 | library | 0.384 | 8.987 | 0.0026 | 0.003 | 9.551 |
| swinb_lora_attn_r8_L0-11 | library | 0.380 | 8.711 | 0.0013 | 0.001 | 9.283 |
| swinb_lora_attn_r8_L0_2_4_6_8_10_12_14_16_18_20_22 | library | 0.381 | 8.583 | 0.0015 | 0.001 | 9.156 |
| swinb_lora_attn_r8_L12-23 | library | 0.382 | 3.144 | 0.0017 | 0.002 | 3.705 |
| swinb_lora_attn_r8_L1_3_5_7_9_11_13_15_17_19_21_23 | library | 0.381 | 7.617 | 0.0015 | 0.001 | 8.178 |
| swinb_lora_mlp_r8 | library | 0.389 | 11.447 | 0.0040 | 0.004 | 12.003 |


### Merge / unmerge (measured by merge_study.py)

Correctness, fp32 (merged vs unmerged logits):

| experiment | adapter | impl | max abs Δlogit | top-1 agreement | library vs scratch (merged) max abs Δ |
|---|---|---|---|---|---|
| swinb_lora_attn_mlp_r8 | trained | library | 1.25e-06 | 1.000 | - |
| swinb_lora_attn_r8 | trained | library | 6.11e-07 | 1.000 | - |
| swinb_lora_attn_r8_L0-11 | trained | library | 1.73e-06 | 1.000 | - |
| swinb_lora_attn_r8_L0_2_4_6_8_10_12_14_16_18_20_22 | trained | library | 8.64e-07 | 1.000 | - |
| swinb_lora_attn_r8_L12-23 | trained | library | 2.38e-07 | 1.000 | - |
| swinb_lora_attn_r8_L1_3_5_7_9_11_13_15_17_19_21_23 | trained | library | 7.15e-07 | 1.000 | - |
| swinb_lora_mlp_r8 | trained | library | 1.57e-06 | 1.000 | - |

Round-trip drift (merge → unmerge):

| experiment | impl | dtype | cycles | max abs ΔW vs pristine | max abs Δlogit vs original |
|---|---|---|---|---|---|
| swinb_lora_attn_mlp_r8 | library | fp32 | 100 | 5.96e-08 | 6.26e-07 |
| swinb_lora_attn_r8 | library | fp32 | 100 | 2.98e-08 | 8.79e-07 |
| swinb_lora_attn_r8_L0-11 | library | fp32 | 100 | 2.98e-08 | 1.79e-06 |
| swinb_lora_attn_r8_L0_2_4_6_8_10_12_14_16_18_20_22 | library | fp32 | 100 | 2.98e-08 | 8.34e-07 |
| swinb_lora_attn_r8_L12-23 | library | fp32 | 100 | 2.98e-08 | 2.09e-07 |
| swinb_lora_attn_r8_L1_3_5_7_9_11_13_15_17_19_21_23 | library | fp32 | 100 | 2.98e-08 | 8.34e-07 |
| swinb_lora_mlp_r8 | library | fp32 | 100 | 5.96e-08 | 1.01e-06 |

Cost:

| experiment | impl / item | merge-all ms | unmerge-all ms | transient / size |
|---|---|---|---|---|
| swinb_lora_attn_mlp_r8 | library | 6.3 | 6.2 | 32.0 |
| swinb_lora_attn_mlp_r8 | reload full state dict | 133.3 | - | 338 MiB file |
| swinb_lora_attn_mlp_r8 | load+apply adapter file | 16.3 | - | 6.33 MiB file |
| swinb_lora_attn_r8 | library | 4.0 | 3.8 | 8.0 |
| swinb_lora_attn_r8 | reload full state dict | 126.6 | - | 334 MiB file |
| swinb_lora_attn_r8 | load+apply adapter file | 10.0 | - | 2.63 MiB file |
| swinb_lora_attn_r8_L0-11 | library | 2.2 | 2.0 | 2.0 |
| swinb_lora_attn_r8_L0-11 | reload full state dict | 108.0 | - | 333 MiB file |
| swinb_lora_attn_r8_L0-11 | load+apply adapter file | 5.4 | - | 1.30 MiB file |
| swinb_lora_attn_r8_L0_2_4_6_8_10_12_14_16_18_20_22 | library | 2.3 | 2.0 | 8.0 |
| swinb_lora_attn_r8_L0_2_4_6_8_10_12_14_16_18_20_22 | reload full state dict | 109.1 | - | 333 MiB file |
| swinb_lora_attn_r8_L0_2_4_6_8_10_12_14_16_18_20_22 | load+apply adapter file | 5.5 | - | 1.51 MiB file |
| swinb_lora_attn_r8_L12-23 | library | 2.2 | 2.0 | 8.0 |
| swinb_lora_attn_r8_L12-23 | reload full state dict | 125.1 | - | 333 MiB file |
| swinb_lora_attn_r8_L12-23 | load+apply adapter file | 5.6 | - | 1.72 MiB file |
| swinb_lora_attn_r8_L1_3_5_7_9_11_13_15_17_19_21_23 | library | 2.4 | 2.2 | 8.0 |
| swinb_lora_attn_r8_L1_3_5_7_9_11_13_15_17_19_21_23 | reload full state dict | 120.9 | - | 333 MiB file |
| swinb_lora_attn_r8_L1_3_5_7_9_11_13_15_17_19_21_23 | load+apply adapter file | 6.2 | - | 1.51 MiB file |
| swinb_lora_mlp_r8 | library | 2.9 | 2.7 | 32.0 |
| swinb_lora_mlp_r8 | reload full state dict | 127.1 | - | 336 MiB file |
| swinb_lora_mlp_r8 | load+apply adapter file | 7.6 | - | 4.09 MiB file |

Test accuracy (trained adapter):

| experiment | impl | unmerged % | merged % | after cycles % | pred agreement merged/unmerged |
|---|---|---|---|---|---|
| swinb_lora_attn_mlp_r8 | library | 91.84 | 91.84 | 91.84 | 1.0000 |
| swinb_lora_attn_r8 | library | 91.67 | 91.67 | 91.67 | 1.0000 |
| swinb_lora_attn_r8_L0-11 | library | 91.29 | 91.29 | 91.29 | 1.0000 |
| swinb_lora_attn_r8_L0_2_4_6_8_10_12_14_16_18_20_22 | library | 91.44 | 91.44 | 91.44 | 1.0000 |
| swinb_lora_attn_r8_L12-23 | library | 90.42 | 90.42 | 90.42 | 1.0000 |
| swinb_lora_attn_r8_L1_3_5_7_9_11_13_15_17_19_21_23 | library | 91.49 | 91.49 | 91.49 | 1.0000 |
| swinb_lora_mlp_r8 | library | 91.74 | 91.74 | 91.74 | 1.0000 |


## Excluded from this report (known invalid in the sweep JSONs)

- `inference_*` memory fields (probe measured with training state resident) — replaced by the profiler tables above.
- `batch1_forward_sec_*` — rough, unmerged adapters, 20 iterations.
- `best_checkpoint_size_mb` — a full state dict for scratch LoRA; artifact size above is analytic/exported.


Skipped files:

- results/legacy/model_seed42.json (experiment 'legacy' not in registry)
