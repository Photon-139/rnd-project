# Aggregated results

Source: per-run result JSONs (single source of truth). Memory in GiB (1024³ B), sizes in MiB. Peak memory = peak **allocated** by PyTorch during train+val, fp32, batch 64 (excludes CUDA context). `peak − static` = peak allocated minus (params + grads + AdamW state); it is not pure activation memory (for full FT it understates activations by about the gradient size). Per-batch timings use epochs ≥ 2. Every metric column is mean ± std over seeds (sample std, n−1); medians/min/max are in stats.json.

## ⚠ Assertion failures

- baseline_lr1e-5/library/seed1024: verification gate 'init_checks_passed' = None
- baseline_lr1e-5/library/seed1024: verification gate 'trainable_sets_match' = None
- baseline_lr1e-5/library/seed1024: verification gate 'forward_ok' = None
- baseline_lr1e-5/library/seed123: verification gate 'init_checks_passed' = None
- baseline_lr1e-5/library/seed123: verification gate 'trainable_sets_match' = None
- baseline_lr1e-5/library/seed123: verification gate 'forward_ok' = None
- baseline_lr1e-5/library/seed42: verification gate 'init_checks_passed' = None
- baseline_lr1e-5/library/seed42: verification gate 'trainable_sets_match' = None
- baseline_lr1e-5/library/seed42: verification gate 'forward_ok' = None
- baseline_lr1e-5/library/seed456: verification gate 'init_checks_passed' = None
- baseline_lr1e-5/library/seed456: verification gate 'trainable_sets_match' = None
- baseline_lr1e-5/library/seed456: verification gate 'forward_ok' = None
- baseline_lr1e-5/library/seed789: verification gate 'init_checks_passed' = None
- baseline_lr1e-5/library/seed789: verification gate 'trainable_sets_match' = None
- baseline_lr1e-5/library/seed789: verification gate 'forward_ok' = None
- baseline_lr3e-6/library/seed123: verification gate 'init_checks_passed' = None
- baseline_lr3e-6/library/seed123: verification gate 'trainable_sets_match' = None
- baseline_lr3e-6/library/seed123: verification gate 'forward_ok' = None
- baseline_lr3e-6/library/seed42: verification gate 'init_checks_passed' = None
- baseline_lr3e-6/library/seed42: verification gate 'trainable_sets_match' = None
- baseline_lr3e-6/library/seed42: verification gate 'forward_ok' = None
- baseline_lr3e-6/library/seed456: verification gate 'init_checks_passed' = None
- baseline_lr3e-6/library/seed456: verification gate 'trainable_sets_match' = None
- baseline_lr3e-6/library/seed456: verification gate 'forward_ok' = None
- lora_attn_r8_lr1e-3/library/seed123: verification gate 'init_checks_passed' = None
- lora_attn_r8_lr1e-3/library/seed123: verification gate 'trainable_sets_match' = None
- lora_attn_r8_lr1e-3/library/seed123: verification gate 'forward_ok' = None
- lora_attn_r8_lr1e-3/library/seed123: verification gate 'lora_forward_ok' = None
- lora_attn_r8_lr1e-3/library/seed42: verification gate 'init_checks_passed' = None
- lora_attn_r8_lr1e-3/library/seed42: verification gate 'trainable_sets_match' = None
- lora_attn_r8_lr1e-3/library/seed42: verification gate 'forward_ok' = None
- lora_attn_r8_lr1e-3/library/seed42: verification gate 'lora_forward_ok' = None
- lora_attn_r8_lr1e-3/library/seed456: verification gate 'init_checks_passed' = None
- lora_attn_r8_lr1e-3/library/seed456: verification gate 'trainable_sets_match' = None
- lora_attn_r8_lr1e-3/library/seed456: verification gate 'forward_ok' = None
- lora_attn_r8_lr1e-3/library/seed456: verification gate 'lora_forward_ok' = None
- lora_attn_r8_lr3e-4/library/seed123: verification gate 'init_checks_passed' = None
- lora_attn_r8_lr3e-4/library/seed123: verification gate 'trainable_sets_match' = None
- lora_attn_r8_lr3e-4/library/seed123: verification gate 'forward_ok' = None
- lora_attn_r8_lr3e-4/library/seed123: verification gate 'lora_forward_ok' = None
- lora_attn_r8_lr3e-4/library/seed42: verification gate 'init_checks_passed' = None
- lora_attn_r8_lr3e-4/library/seed42: verification gate 'trainable_sets_match' = None
- lora_attn_r8_lr3e-4/library/seed42: verification gate 'forward_ok' = None
- lora_attn_r8_lr3e-4/library/seed42: verification gate 'lora_forward_ok' = None
- lora_attn_r8_lr3e-4/library/seed456: verification gate 'init_checks_passed' = None
- lora_attn_r8_lr3e-4/library/seed456: verification gate 'trainable_sets_match' = None
- lora_attn_r8_lr3e-4/library/seed456: verification gate 'forward_ok' = None
- lora_attn_r8_lr3e-4/library/seed456: verification gate 'lora_forward_ok' = None
- lora_attn_r8_lr3e-5/library/seed123: verification gate 'init_checks_passed' = None
- lora_attn_r8_lr3e-5/library/seed123: verification gate 'trainable_sets_match' = None
- lora_attn_r8_lr3e-5/library/seed123: verification gate 'forward_ok' = None
- lora_attn_r8_lr3e-5/library/seed123: verification gate 'lora_forward_ok' = None
- lora_attn_r8_lr3e-5/library/seed42: verification gate 'init_checks_passed' = None
- lora_attn_r8_lr3e-5/library/seed42: verification gate 'trainable_sets_match' = None
- lora_attn_r8_lr3e-5/library/seed42: verification gate 'forward_ok' = None
- lora_attn_r8_lr3e-5/library/seed42: verification gate 'lora_forward_ok' = None
- lora_attn_r8_lr3e-5/library/seed456: verification gate 'init_checks_passed' = None
- lora_attn_r8_lr3e-5/library/seed456: verification gate 'trainable_sets_match' = None
- lora_attn_r8_lr3e-5/library/seed456: verification gate 'forward_ok' = None
- lora_attn_r8_lr3e-5/library/seed456: verification gate 'lora_forward_ok' = None
- runs come from 2 different source versions ['b4869e25285d', 'bad188ad14e4']; do not pool them without checking what changed

## Missing runs (registry seeds not found)

- baseline_lr1e-5/scratch/seed42
- baseline_lr1e-5/scratch/seed123
- baseline_lr1e-5/scratch/seed456
- baseline_lr1e-5/scratch/seed789
- baseline_lr1e-5/scratch/seed1024
- baseline_lr3e-6/scratch/seed42
- baseline_lr3e-6/scratch/seed123
- baseline_lr3e-6/scratch/seed456
- baseline_lr3e-6/scratch/seed789
- baseline_lr3e-6/scratch/seed1024
- baseline_lr3e-6/library/seed789
- baseline_lr3e-6/library/seed1024
- lora_attn_r8_lr1e-3/scratch/seed42
- lora_attn_r8_lr1e-3/scratch/seed123
- lora_attn_r8_lr1e-3/scratch/seed456
- lora_attn_r8_lr1e-3/scratch/seed789
- lora_attn_r8_lr1e-3/scratch/seed1024
- lora_attn_r8_lr1e-3/library/seed789
- lora_attn_r8_lr1e-3/library/seed1024
- lora_attn_r8_lr3e-4/scratch/seed42
- lora_attn_r8_lr3e-4/scratch/seed123
- lora_attn_r8_lr3e-4/scratch/seed456
- lora_attn_r8_lr3e-4/scratch/seed789
- lora_attn_r8_lr3e-4/scratch/seed1024
- lora_attn_r8_lr3e-4/library/seed789
- lora_attn_r8_lr3e-4/library/seed1024
- lora_attn_r8_lr3e-5/scratch/seed42
- lora_attn_r8_lr3e-5/scratch/seed123
- lora_attn_r8_lr3e-5/scratch/seed456
- lora_attn_r8_lr3e-5/scratch/seed789
- lora_attn_r8_lr3e-5/scratch/seed1024
- lora_attn_r8_lr3e-5/library/seed789
- lora_attn_r8_lr3e-5/library/seed1024
- baseline: no runs found
- baseline_lr1e-3: no runs found
- lora_attn_r16: no runs found
- lora_mlp_r16: no runs found
- lora_attn_mlp_r16: no runs found

## Table 1 — Full fine-tuning, learning-rate sweep

| experiment | lr | impl | n | test acc % | best ep | stop ep | epoch train s | fwd ms/batch | bwd ms/batch | opt ms/batch | peak alloc GiB | peak − static GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_lr1e-5 | 1e-05 | library | 5 | 90.71 ± 0.29 | 5.6 ± 0.5 | 10.6 ± 0.5 | 439.9 ± 0.5 | 208.96 ± 0.25 | 408.79 ± 0.43 | 5.72 ± 0.04 | 8.174 ± 0.000 | 6.895 ± 0.000 |
| baseline_lr3e-6 | 3e-06 | library | 3 | 90.11 ± 0.18 | 12.3 ± 0.6 | 17.3 ± 0.6 | 440.3 ± 0.1 | 209.15 ± 0.01 | 409.16 ± 0.05 | 5.76 ± 0.05 | 8.174 ± 0.000 | 6.895 ± 0.000 |

## Table 2 — LoRA configurations (lr 1e-4)

| experiment | lr | impl | n | test acc % | best ep | stop ep | epoch train s | fwd ms/batch | bwd ms/batch | opt ms/batch | peak alloc GiB | peak − static GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora_attn_r8 | 0.0001 | library | 5 | 91.10 ± 0.08 | 5.2 ± 0.8 | 10.2 ± 0.8 | 323.1 ± 0.4 | 219.77 ± 0.15 | 236.90 ± 0.24 | 1.06 ± 0.14 | 5.459 ± 0.000 | 5.131 ± 0.000 |
| lora_attn_r8 | 0.0001 | scratch | 5 | 91.09 ± 0.06 | 5.2 ± 0.8 | 10.2 ± 0.8 | 323.1 ± 0.4 | 219.67 ± 0.15 | 236.91 ± 0.24 | 1.13 ± 0.14 | 5.459 ± 0.000 | 5.132 ± 0.000 |
| lora_mlp_r8 | 0.0001 | library | 5 | 91.35 ± 0.15 | 3.4 ± 0.5 | 8.4 ± 0.5 | 325.4 ± 0.2 | 224.91 ± 0.03 | 235.07 ± 0.10 | 1.00 ± 0.12 | 7.011 ± 0.000 | 6.679 ± 0.000 |
| lora_mlp_r8 | 0.0001 | scratch | 5 | 91.30 ± 0.21 | 3.4 ± 0.5 | 8.4 ± 0.5 | 325.4 ± 0.1 | 224.81 ± 0.01 | 235.03 ± 0.02 | 0.99 ± 0.03 | 7.013 ± 0.000 | 6.681 ± 0.000 |
| lora_attn_mlp_r8 | 0.0001 | library | 5 | 91.43 ± 0.10 | 3.2 ± 0.4 | 8.2 ± 0.4 | 353.7 ± 0.2 | 235.52 ± 0.02 | 264.01 ± 0.08 | 1.63 ± 0.09 | 7.643 ± 0.000 | 7.305 ± 0.000 |
| lora_attn_mlp_r8 | 0.0001 | scratch | 5 | 91.44 ± 0.02 | 3.2 ± 0.4 | 8.2 ± 0.4 | 353.8 ± 0.1 | 235.43 ± 0.02 | 264.05 ± 0.09 | 1.70 ± 0.09 | 7.646 ± 0.000 | 7.308 ± 0.000 |
| lora_attn_r8_L0-5 | 0.0001 | library | 5 | 91.31 ± 0.11 | 8.0 ± 0.7 | 13.0 ± 0.7 | 311.6 ± 0.2 | 214.02 ± 0.03 | 226.60 ± 0.11 | 0.77 ± 0.14 | 5.233 ± 0.000 | 4.909 ± 0.000 |
| lora_attn_r8_L0-5 | 0.0001 | scratch | 5 | 91.30 ± 0.12 | 8.2 ± 0.8 | 13.2 ± 0.8 | 311.6 ± 0.2 | 213.96 ± 0.02 | 226.61 ± 0.09 | 0.81 ± 0.15 | 5.233 ± 0.000 | 4.909 ± 0.000 |
| lora_attn_r8_L6-11 | 0.0001 | library | 5 | 89.54 ± 0.13 | 6.8 ± 0.8 | 11.8 ± 0.8 | 234.0 ± 0.2 | 213.93 ± 0.03 | 116.47 ± 0.11 | 0.74 ± 0.12 | 3.061 ± 0.000 | 2.737 ± 0.000 |
| lora_attn_r8_L6-11 | 0.0001 | scratch | 5 | 89.54 ± 0.13 | 6.8 ± 0.8 | 11.8 ± 0.8 | 234.0 ± 0.2 | 213.86 ± 0.03 | 116.48 ± 0.11 | 0.76 ± 0.12 | 3.062 ± 0.000 | 2.738 ± 0.000 |
| lora_attn_r8_L0_2_4_6_8_10 | 0.0001 | library | 5 | 91.05 ± 0.08 | 7.6 ± 0.9 | 12.6 ± 0.9 | 311.2 ± 0.2 | 213.75 ± 0.04 | 226.31 ± 0.12 | 0.77 ± 0.14 | 5.233 ± 0.000 | 4.909 ± 0.000 |
| lora_attn_r8_L0_2_4_6_8_10 | 0.0001 | scratch | 5 | 91.08 ± 0.07 | 7.4 ± 0.5 | 12.4 ± 0.5 | 311.2 ± 0.2 | 213.71 ± 0.09 | 226.34 ± 0.10 | 0.77 ± 0.14 | 5.234 ± 0.000 | 4.910 ± 0.000 |
| lora_attn_r8_L1_3_5_7_9_11 | 0.0001 | library | 5 | 90.89 ± 0.22 | 6.6 ± 0.9 | 11.6 ± 0.9 | 298.5 ± 0.2 | 213.91 ± 0.03 | 208.12 ± 0.10 | 0.77 ± 0.13 | 4.871 ± 0.000 | 4.547 ± 0.000 |
| lora_attn_r8_L1_3_5_7_9_11 | 0.0001 | scratch | 5 | 90.96 ± 0.14 | 7.0 ± 1.0 | 12.0 ± 1.0 | 298.5 ± 0.2 | 213.84 ± 0.03 | 208.14 ± 0.11 | 0.80 ± 0.14 | 4.872 ± 0.000 | 4.548 ± 0.000 |
| lora_attn_r8_lr3e-5 | 3e-05 | library | 3 | 91.01 ± 0.18 | 12.3 ± 0.6 | 17.3 ± 0.6 | 323.9 ± 0.0 | 220.05 ± 0.00 | 237.44 ± 0.03 | 1.31 ± 0.04 | 5.459 ± 0.000 | 5.131 ± 0.000 |
| lora_attn_r8_lr3e-4 | 0.0003 | library | 3 | 91.39 ± 0.15 | 3.0 ± 0.0 | 8.0 ± 0.0 | 323.7 ± 0.0 | 220.03 ± 0.01 | 237.31 ± 0.04 | 1.18 ± 0.04 | 5.459 ± 0.000 | 5.131 ± 0.000 |
| lora_attn_r8_lr1e-3 | 0.001 | library | 3 | 90.36 ± 0.09 | 1.3 ± 0.6 | 6.3 ± 0.6 | 323.6 ± 0.1 | 220.02 ± 0.01 | 237.21 ± 0.04 | 1.09 ± 0.05 | 5.459 ± 0.000 | 5.131 ± 0.000 |

## Scratch − library, paired per seed

Each cell: mean difference ± std of the per-seed differences (95% t-CI half-width, n seeds). A CI that contains 0 means **no detectable difference at this sample size**; it is not evidence of equivalence. With n=3 the t critical value is 4.30, so intervals are wide.

| experiment | Δ test acc (pp) | Δ stop epoch | Δ epoch train (s) | Δ fwd (ms) | Δ bwd (ms) | Δ peak alloc (GiB) |
|---|---|---|---|---|---|---|
| lora_attn_r8 | -0.01 ± 0.05 (CI95 ±0.06, n=5) | +0.0 ± 0.0 (CI95 ±0.0, n=5) | -0.0 ± 0.2 (CI95 ±0.2, n=5) | -0.10 ± 0.06 (CI95 ±0.07, n=5) | +0.01 ± 0.12 (CI95 ±0.14, n=5) | +0.001 ± 0.000 (CI95 ±0.000, n=5) |
| lora_mlp_r8 | -0.05 ± 0.10 (CI95 ±0.13, n=5) | +0.0 ± 0.0 (CI95 ±0.0, n=5) | -0.1 ± 0.2 (CI95 ±0.2, n=5) | -0.10 ± 0.03 (CI95 ±0.03, n=5) | -0.03 ± 0.10 (CI95 ±0.12, n=5) | +0.002 ± 0.000 (CI95 ±0.000, n=5) |
| lora_attn_mlp_r8 | +0.01 ± 0.10 (CI95 ±0.12, n=5) | +0.0 ± 0.0 (CI95 ±0.0, n=5) | +0.1 ± 0.2 (CI95 ±0.3, n=5) | -0.08 ± 0.03 (CI95 ±0.03, n=5) | +0.04 ± 0.14 (CI95 ±0.17, n=5) | +0.003 ± 0.000 (CI95 ±0.000, n=5) |
| lora_attn_r8_L0-5 | -0.01 ± 0.09 (CI95 ±0.11, n=5) | +0.2 ± 0.4 (CI95 ±0.6, n=5) | -0.0 ± 0.1 (CI95 ±0.1, n=5) | -0.06 ± 0.04 (CI95 ±0.04, n=5) | +0.01 ± 0.04 (CI95 ±0.05, n=5) | +0.000 ± 0.000 (CI95 ±0.000, n=5) |
| lora_attn_r8_L6-11 | +0.00 ± 0.00 (CI95 ±0.00, n=5) | +0.0 ± 0.0 (CI95 ±0.0, n=5) | +0.0 ± 0.1 (CI95 ±0.1, n=5) | -0.07 ± 0.02 (CI95 ±0.03, n=5) | +0.01 ± 0.06 (CI95 ±0.07, n=5) | +0.001 ± 0.000 (CI95 ±0.000, n=5) |
| lora_attn_r8_L0_2_4_6_8_10 | +0.04 ± 0.07 (CI95 ±0.09, n=5) | -0.2 ± 0.4 (CI95 ±0.6, n=5) | +0.0 ± 0.1 (CI95 ±0.2, n=5) | -0.04 ± 0.10 (CI95 ±0.13, n=5) | +0.03 ± 0.09 (CI95 ±0.12, n=5) | +0.001 ± 0.000 (CI95 ±0.000, n=5) |
| lora_attn_r8_L1_3_5_7_9_11 | +0.08 ± 0.13 (CI95 ±0.16, n=5) | +0.4 ± 0.9 (CI95 ±1.1, n=5) | -0.0 ± 0.1 (CI95 ±0.1, n=5) | -0.08 ± 0.01 (CI95 ±0.02, n=5) | +0.02 ± 0.04 (CI95 ±0.05, n=5) | +0.001 ± 0.000 (CI95 ±0.000, n=5) |

### Per-seed accuracy differences

| experiment | seed | library acc % | scratch acc % | scratch − library (pp) |
|---|---|---|---|---|
| lora_attn_r8 | 42 | 91.17 | 91.07 | -0.10 |
| lora_attn_r8 | 123 | 91.12 | 91.14 | +0.02 |
| lora_attn_r8 | 456 | 90.97 | 90.99 | +0.02 |
| lora_attn_r8 | 789 | 91.12 | 91.11 | -0.01 |
| lora_attn_r8 | 1024 | 91.13 | 91.14 | +0.01 |
| lora_mlp_r8 | 42 | 91.30 | 91.41 | +0.11 |
| lora_mlp_r8 | 123 | 91.16 | 91.02 | -0.14 |
| lora_mlp_r8 | 456 | 91.33 | 91.20 | -0.13 |
| lora_mlp_r8 | 789 | 91.40 | 91.32 | -0.08 |
| lora_mlp_r8 | 1024 | 91.57 | 91.57 | +0.00 |
| lora_attn_mlp_r8 | 42 | 91.44 | 91.45 | +0.01 |
| lora_attn_mlp_r8 | 123 | 91.38 | 91.43 | +0.05 |
| lora_attn_mlp_r8 | 456 | 91.57 | 91.46 | -0.11 |
| lora_attn_mlp_r8 | 789 | 91.30 | 91.44 | +0.14 |
| lora_attn_mlp_r8 | 1024 | 91.47 | 91.41 | -0.06 |
| lora_attn_r8_L0-5 | 42 | 91.33 | 91.17 | -0.16 |
| lora_attn_r8_L0-5 | 123 | 91.39 | 91.41 | +0.02 |
| lora_attn_r8_L0-5 | 456 | 91.13 | 91.18 | +0.05 |
| lora_attn_r8_L0-5 | 789 | 91.40 | 91.37 | -0.03 |
| lora_attn_r8_L0-5 | 1024 | 91.32 | 91.38 | +0.06 |
| lora_attn_r8_L6-11 | 42 | 89.73 | 89.73 | +0.00 |
| lora_attn_r8_L6-11 | 123 | 89.51 | 89.51 | +0.00 |
| lora_attn_r8_L6-11 | 456 | 89.61 | 89.61 | +0.00 |
| lora_attn_r8_L6-11 | 789 | 89.42 | 89.42 | +0.00 |
| lora_attn_r8_L6-11 | 1024 | 89.45 | 89.45 | +0.00 |
| lora_attn_r8_L0_2_4_6_8_10 | 42 | 91.08 | 91.04 | -0.04 |
| lora_attn_r8_L0_2_4_6_8_10 | 123 | 90.95 | 91.10 | +0.15 |
| lora_attn_r8_L0_2_4_6_8_10 | 456 | 91.01 | 91.02 | +0.01 |
| lora_attn_r8_L0_2_4_6_8_10 | 789 | 91.15 | 91.20 | +0.05 |
| lora_attn_r8_L0_2_4_6_8_10 | 1024 | 91.04 | 91.06 | +0.02 |
| lora_attn_r8_L1_3_5_7_9_11 | 42 | 90.62 | 90.88 | +0.26 |
| lora_attn_r8_L1_3_5_7_9_11 | 123 | 91.21 | 91.14 | -0.07 |
| lora_attn_r8_L1_3_5_7_9_11 | 456 | 90.82 | 90.82 | +0.00 |
| lora_attn_r8_L1_3_5_7_9_11 | 789 | 90.94 | 91.08 | +0.14 |
| lora_attn_r8_L1_3_5_7_9_11 | 1024 | 90.84 | 90.90 | +0.06 |

## Trainable parameters and per-task artifact size

Artifact = head + adapters (LoRA) or the full weights (baseline), fp32, the same for both implementations.

| experiment | trainable elements | artifact MiB (analytic, fp32) | measured adapter export MiB |
|---|---|---|---|
| baseline_lr1e-5 | 85,875,556 | 327.59 | - |
| baseline_lr3e-6 | 85,875,556 | 327.59 | - |
| lora_attn_mlp_r8 | 1,256,548 | 4.79 | 4.83 |
| lora_attn_r8 | 519,268 | 1.98 | 2.00 |
| lora_attn_r8_L0-5 | 298,084 | 1.14 | 1.15 |
| lora_attn_r8_L0_2_4_6_8_10 | 298,084 | 1.14 | 1.15 |
| lora_attn_r8_L1_3_5_7_9_11 | 298,084 | 1.14 | 1.15 |
| lora_attn_r8_L6-11 | 298,084 | 1.14 | 1.15 |
| lora_attn_r8_lr1e-3 | 519,268 | 1.98 | 2.00 |
| lora_attn_r8_lr3e-4 | 519,268 | 1.98 | 2.00 |
| lora_attn_r8_lr3e-5 | 519,268 | 1.98 | 2.00 |
| lora_mlp_r8 | 814,180 | 3.11 | 3.12 |

### Profiler acceptance checks (147 checks: 147 pass, 0 warn, 0 fail)

The profiler tables below count as evidence only if these hold. They compare profile_memory.py against the independent in-loop measurements and against analytic byte counts.

| scope | check | status | detail |
|---|---|---|---|
| lora_attn_mlp_r8/library | grads (profiler) = grads (static breakdown) | PASS | 4.82 vs 4.79 MiB |
| lora_attn_mlp_r8/library | analytic resident (live tensors) = params + AdamW state + input batch | PASS | 0.3698 vs 0.3698 GiB |
| lora_attn_mlp_r8/library | runtime overhead (allocator/cuBLAS workspace) is small | PASS | 17.9 MiB |
| lora_attn_mlp_r8/library | batch-64 peak (profiler) vs in-loop training peak | PASS | 7.643 vs 7.643 GiB (0.0% apart) |
| lora_attn_mlp_r8/scratch | grads (profiler) = grads (static breakdown) | PASS | 4.82 vs 4.79 MiB |
| lora_attn_mlp_r8/scratch | analytic resident (live tensors) = params + AdamW state + input batch | PASS | 0.3696 vs 0.3696 GiB |

(Only the first rows are shown because every check passed; all are in stats.json.)


### Inference memory and latency (fresh process, no training; measured by profile_memory.py)

| experiment | impl | variant | params | resident weights GiB | b1 peak alloc GiB | b1 p50 ms | b64 peak alloc GiB | b64 p50 ms |
|---|---|---|---|---|---|---|---|---|
| lora_attn_mlp_r8 | library | fp32_unmerged | 87,132,104 | 0.3341 | 0.3437 | 8.41 | 0.948 | 235.7 |
| lora_attn_mlp_r8 | library | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.26 | 0.799 | 208.9 |
| lora_attn_mlp_r8 | scratch | fp32_unmerged | 87,055,204 | 0.3338 | 0.3457 | 8.37 | 1.092 | 235.6 |
| lora_attn_mlp_r8 | scratch | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.24 | 0.799 | 208.9 |
| lora_attn_r8 | library | fp32_unmerged | 86,394,824 | 0.3314 | 0.3387 | 7.47 | 0.801 | 219.8 |
| lora_attn_r8 | library | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.27 | 0.799 | 208.9 |
| lora_attn_r8 | scratch | fp32_unmerged | 86,317,924 | 0.3311 | 0.3384 | 7.42 | 0.800 | 219.7 |
| lora_attn_r8 | scratch | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.23 | 0.799 | 208.8 |
| lora_attn_r8_L0-5 | library | fp32_unmerged | 86,173,640 | 0.3305 | 0.3379 | 6.85 | 0.800 | 214.2 |
| lora_attn_r8_L0-5 | library | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.28 | 0.799 | 208.7 |
| lora_attn_r8_L0-5 | scratch | fp32_unmerged | 86,096,740 | 0.3303 | 0.3376 | 6.80 | 0.799 | 214.2 |
| lora_attn_r8_L0-5 | scratch | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.24 | 0.799 | 208.7 |
| lora_attn_r8_L0_2_4_6_8_10 | library | fp32_unmerged | 86,173,640 | 0.3305 | 0.3379 | 6.89 | 0.800 | 214.2 |
| lora_attn_r8_L0_2_4_6_8_10 | library | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.27 | 0.799 | 208.7 |
| lora_attn_r8_L0_2_4_6_8_10 | scratch | fp32_unmerged | 86,096,740 | 0.3303 | 0.3376 | 6.84 | 0.799 | 214.1 |
| lora_attn_r8_L0_2_4_6_8_10 | scratch | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.24 | 0.799 | 208.6 |
| lora_attn_r8_L1_3_5_7_9_11 | library | fp32_unmerged | 86,173,640 | 0.3305 | 0.3379 | 6.85 | 0.800 | 214.1 |
| lora_attn_r8_L1_3_5_7_9_11 | library | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.26 | 0.799 | 208.6 |
| lora_attn_r8_L1_3_5_7_9_11 | scratch | fp32_unmerged | 86,096,740 | 0.3303 | 0.3376 | 6.81 | 0.799 | 214.0 |
| lora_attn_r8_L1_3_5_7_9_11 | scratch | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.23 | 0.799 | 208.5 |
| lora_attn_r8_L6-11 | library | fp32_unmerged | 86,173,640 | 0.3305 | 0.3379 | 6.82 | 0.800 | 214.1 |
| lora_attn_r8_L6-11 | library | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.26 | 0.799 | 208.7 |
| lora_attn_r8_L6-11 | scratch | fp32_unmerged | 86,096,740 | 0.3303 | 0.3376 | 6.77 | 0.799 | 214.1 |
| lora_attn_r8_L6-11 | scratch | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.22 | 0.799 | 208.6 |
| lora_mlp_r8 | library | fp32_unmerged | 86,689,736 | 0.3325 | 0.3420 | 7.30 | 0.946 | 225.0 |
| lora_mlp_r8 | library | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.28 | 0.799 | 209.2 |
| lora_mlp_r8 | scratch | fp32_unmerged | 86,612,836 | 0.3322 | 0.3440 | 7.25 | 1.090 | 225.0 |
| lora_mlp_r8 | scratch | fp32_merged | 85,875,556 | 0.3294 | 0.3368 | 6.24 | 0.799 | 209.1 |

### Training-step decomposition at batch 64 (measured by profile_memory.py)

| experiment | impl | resident before fwd GiB | activation GiB | grads GiB | optimizer transient GiB | peak GiB |
|---|---|---|---|---|---|---|
| lora_attn_mlp_r8 | library | 0.387 | 7.003 | 0.0047 | 0.005 | 7.643 |
| lora_attn_mlp_r8 | scratch | 0.387 | 7.006 | 0.0047 | 0.005 | 7.645 |
| lora_attn_r8 | library | 0.379 | 4.827 | 0.0020 | 0.002 | 5.459 |
| lora_attn_r8 | scratch | 0.379 | 4.826 | 0.0020 | 0.002 | 5.459 |
| lora_attn_r8_L0-5 | library | 0.377 | 4.603 | 0.0011 | 0.001 | 5.233 |
| lora_attn_r8_L0-5 | scratch | 0.376 | 4.603 | 0.0011 | 0.001 | 5.233 |
| lora_attn_r8_L0_2_4_6_8_10 | library | 0.377 | 4.603 | 0.0011 | 0.001 | 5.233 |
| lora_attn_r8_L0_2_4_6_8_10 | scratch | 0.376 | 4.603 | 0.0011 | 0.001 | 5.233 |
| lora_attn_r8_L1_3_5_7_9_11 | library | 0.377 | 4.242 | 0.0011 | 0.001 | 4.871 |
| lora_attn_r8_L1_3_5_7_9_11 | scratch | 0.376 | 4.241 | 0.0011 | 0.001 | 4.871 |
| lora_attn_r8_L6-11 | library | 0.377 | 2.432 | 0.0011 | 0.001 | 3.061 |
| lora_attn_r8_L6-11 | scratch | 0.376 | 2.431 | 0.0011 | 0.001 | 3.061 |
| lora_mlp_r8 | library | 0.382 | 6.375 | 0.0031 | 0.003 | 7.011 |
| lora_mlp_r8 | scratch | 0.382 | 6.377 | 0.0031 | 0.003 | 7.012 |


### Merge / unmerge (measured by merge_study.py)

Correctness, fp32 (merged vs unmerged logits):

| experiment | adapter | impl | max abs Δlogit | top-1 agreement | library vs scratch (merged) max abs Δ |
|---|---|---|---|---|---|
| lora_attn_mlp_r8 | trained | library | 3.34e-06 | 1.000 | 0.00e+00 |
| lora_attn_mlp_r8 | trained | scratch | 3.34e-06 | 1.000 | 0.00e+00 |
| lora_attn_r8 | trained | library | 3.81e-06 | 1.000 | 0.00e+00 |
| lora_attn_r8 | trained | scratch | 3.81e-06 | 1.000 | 0.00e+00 |
| lora_attn_r8_L0-5 | trained | library | 2.98e-06 | 1.000 | 0.00e+00 |
| lora_attn_r8_L0-5 | trained | scratch | 2.98e-06 | 1.000 | 0.00e+00 |
| lora_attn_r8_L0_2_4_6_8_10 | trained | library | 2.15e-06 | 1.000 | 0.00e+00 |
| lora_attn_r8_L0_2_4_6_8_10 | trained | scratch | 2.15e-06 | 1.000 | 0.00e+00 |
| lora_attn_r8_L1_3_5_7_9_11 | trained | library | 3.81e-06 | 1.000 | 0.00e+00 |
| lora_attn_r8_L1_3_5_7_9_11 | trained | scratch | 3.81e-06 | 1.000 | 0.00e+00 |
| lora_attn_r8_L6-11 | trained | library | 1.43e-06 | 1.000 | 0.00e+00 |
| lora_attn_r8_L6-11 | trained | scratch | 1.43e-06 | 1.000 | 0.00e+00 |
| lora_mlp_r8 | trained | library | 5.01e-06 | 1.000 | 0.00e+00 |
| lora_mlp_r8 | trained | scratch | 5.01e-06 | 1.000 | 0.00e+00 |

Round-trip drift (merge → unmerge):

| experiment | impl | dtype | cycles | max abs ΔW vs pristine | max abs Δlogit vs original |
|---|---|---|---|---|---|
| lora_attn_mlp_r8 | library | fp32 | 100 | 5.96e-08 | 2.50e-06 |
| lora_attn_mlp_r8 | scratch | fp32 | 100 | 5.96e-08 | 2.50e-06 |
| lora_attn_r8 | library | fp32 | 100 | 1.19e-07 | 5.14e-06 |
| lora_attn_r8 | scratch | fp32 | 100 | 1.19e-07 | 5.14e-06 |
| lora_attn_r8_L0-5 | library | fp32 | 100 | 1.19e-07 | 1.73e-06 |
| lora_attn_r8_L0-5 | scratch | fp32 | 100 | 1.19e-07 | 1.73e-06 |
| lora_attn_r8_L0_2_4_6_8_10 | library | fp32 | 100 | 5.96e-08 | 1.67e-06 |
| lora_attn_r8_L0_2_4_6_8_10 | scratch | fp32 | 100 | 5.96e-08 | 1.67e-06 |
| lora_attn_r8_L1_3_5_7_9_11 | library | fp32 | 100 | 5.96e-08 | 4.29e-06 |
| lora_attn_r8_L1_3_5_7_9_11 | scratch | fp32 | 100 | 5.96e-08 | 4.29e-06 |
| lora_attn_r8_L6-11 | library | fp32 | 100 | 2.98e-08 | 1.43e-06 |
| lora_attn_r8_L6-11 | scratch | fp32 | 100 | 2.98e-08 | 1.43e-06 |
| lora_mlp_r8 | library | fp32 | 100 | 5.96e-08 | 1.67e-06 |
| lora_mlp_r8 | scratch | fp32 | 100 | 5.96e-08 | 1.67e-06 |

Cost:

| experiment | impl / item | merge-all ms | unmerge-all ms | transient / size |
|---|---|---|---|---|
| lora_attn_mlp_r8 | library | 3.2 | 3.0 | 18.0 |
| lora_attn_mlp_r8 | scratch | 2.0 | 2.0 | 18.0 |
| lora_attn_mlp_r8 | reload full state dict | 96.0 | - | 332 MiB file |
| lora_attn_mlp_r8 | load+apply adapter file | 7.9 | - | 4.83 MiB file |
| lora_attn_r8 | library | 2.1 | 1.8 | 4.5 |
| lora_attn_r8 | scratch | 1.0 | 1.0 | 4.5 |
| lora_attn_r8 | reload full state dict | 99.9 | - | 329 MiB file |
| lora_attn_r8 | load+apply adapter file | 4.8 | - | 2.00 MiB file |
| lora_attn_r8_L0-5 | library | 1.2 | 1.0 | 4.5 |
| lora_attn_r8_L0-5 | scratch | 0.5 | 0.5 | 4.5 |
| lora_attn_r8_L0-5 | reload full state dict | 91.1 | - | 329 MiB file |
| lora_attn_r8_L0-5 | load+apply adapter file | 2.6 | - | 1.15 MiB file |
| lora_attn_r8_L0_2_4_6_8_10 | library | 1.2 | 1.0 | 4.5 |
| lora_attn_r8_L0_2_4_6_8_10 | scratch | 0.5 | 0.5 | 4.5 |
| lora_attn_r8_L0_2_4_6_8_10 | reload full state dict | 94.0 | - | 329 MiB file |
| lora_attn_r8_L0_2_4_6_8_10 | load+apply adapter file | 2.7 | - | 1.15 MiB file |
| lora_attn_r8_L1_3_5_7_9_11 | library | 1.2 | 1.0 | 4.5 |
| lora_attn_r8_L1_3_5_7_9_11 | scratch | 0.5 | 0.5 | 4.5 |
| lora_attn_r8_L1_3_5_7_9_11 | reload full state dict | 92.5 | - | 329 MiB file |
| lora_attn_r8_L1_3_5_7_9_11 | load+apply adapter file | 2.7 | - | 1.15 MiB file |
| lora_attn_r8_L6-11 | library | 1.2 | 1.0 | 4.5 |
| lora_attn_r8_L6-11 | scratch | 0.5 | 0.5 | 4.5 |
| lora_attn_r8_L6-11 | reload full state dict | 95.8 | - | 329 MiB file |
| lora_attn_r8_L6-11 | load+apply adapter file | 2.7 | - | 1.15 MiB file |
| lora_mlp_r8 | library | 1.5 | 1.3 | 18.0 |
| lora_mlp_r8 | scratch | 1.2 | 1.2 | 18.0 |
| lora_mlp_r8 | reload full state dict | 90.9 | - | 330 MiB file |
| lora_mlp_r8 | load+apply adapter file | 3.8 | - | 3.12 MiB file |

Test accuracy (trained adapter):

| experiment | impl | unmerged % | merged % | after cycles % | pred agreement merged/unmerged |
|---|---|---|---|---|---|
| lora_attn_mlp_r8 | library | 91.44 | 91.44 | 91.44 | 1.0000 |
| lora_attn_mlp_r8 | scratch | 91.44 | 91.44 | 91.44 | 1.0000 |
| lora_attn_r8 | library | 91.17 | 91.17 | 91.17 | 1.0000 |
| lora_attn_r8 | scratch | 91.17 | 91.17 | 91.17 | 1.0000 |
| lora_attn_r8_L0-5 | library | 91.33 | 91.33 | 91.33 | 1.0000 |
| lora_attn_r8_L0-5 | scratch | 91.33 | 91.33 | 91.33 | 1.0000 |
| lora_attn_r8_L0_2_4_6_8_10 | library | 91.08 | 91.08 | 91.08 | 1.0000 |
| lora_attn_r8_L0_2_4_6_8_10 | scratch | 91.08 | 91.08 | 91.08 | 1.0000 |
| lora_attn_r8_L1_3_5_7_9_11 | library | 90.62 | 90.62 | 90.62 | 1.0000 |
| lora_attn_r8_L1_3_5_7_9_11 | scratch | 90.62 | 90.62 | 90.62 | 1.0000 |
| lora_attn_r8_L6-11 | library | 89.73 | 89.73 | 89.73 | 1.0000 |
| lora_attn_r8_L6-11 | scratch | 89.73 | 89.73 | 89.73 | 1.0000 |
| lora_mlp_r8 | library | 91.30 | 91.30 | 91.30 | 1.0000 |
| lora_mlp_r8 | scratch | 91.30 | 91.30 | 91.30 | 1.0000 |


## Excluded from this report (known invalid in the sweep JSONs)

- `inference_*` memory fields (probe measured with training state resident) — replaced by the profiler tables above.
- `batch1_forward_sec_*` — rough, unmerged adapters, 20 iterations.
- `best_checkpoint_size_mb` — a full state dict for scratch LoRA; artifact size above is analytic/exported.
