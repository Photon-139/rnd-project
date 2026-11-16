# Pre-registered predictions: DeiT-B and ViT-B suffix runs (written 2026-11-10, night)

Written **before** any result of the job-2 rerun (`job_vitb_lr_2.sh`, EXPS `deitb_lora_attn_r8
deitb_lora_attn_r8_L6-11 deitb_lora_attn_r8_L3-11 lora_attn_r8_L3-11`, seeds 42 123 456, library) was copied back
or seen.

**Already known when this was written:**
- DeiT-B all-blocks attention, seeds 42 and 123: 86.70% and 86.34% test accuracy; peak training memory 5.4587 GiB;
  about 322 s per training epoch (from the first, interrupted run of job 13062).
- The memory sweep (job 13063), which profiled every configuration below. Its peak memory is not a prediction any
  more; it is listed as the expected training value.

Nothing else about these four configurations was known.

Everything below is a **paired difference per seed** against all-blocks attention of the same model, at the same
seed (the seed also fixes the train/validation split). The source numbers are library runs in `results_v2/`
(ViT-B) and `results/swinb_*` (Swin-B).

## What the predictions are built from

| source (5 seeds unless stated) | first adapted block k / depth L | test acc − all blocks (pp) |
|---|---|---|
| ViT-B second half (blocks 6-11) | 6/12 = 0.5 | −1.56 ± 0.15 (per seed −1.44, −1.61, −1.36, −1.70, −1.68) |
| Swin-B second half (blocks 12-23) | 12/24 = 0.5 | −1.18 ± 0.19 (per seed −1.25, −1.23, −1.10, −1.41, −0.90) |
| ViT-B odd blocks | 1/12 | −0.22 ± 0.23 (no detectable difference; adapts only 6 blocks) |
| Swin-B odd blocks | 1/24 | +0.08 ± 0.27 (no detectable difference) |

**Accuracy model under test.** Δacc ≈ −β·(k/L), linear in the fraction of depth before the first adapter.
- β = 2.4–3.1 pp per unit of k/L, from the two second-half points (mean 2.74).
- The fill pattern after k is ignored.
- At k/L = 0.25 (start at block 3 of 12) it predicts −0.69 pp (range −0.6 to −0.8).

**The budget question this answers.** On DeiT-B with a budget of 4.5 GiB at batch 64, the memory model allows at
most "start at block 3" (4.26 GiB measured; starting at block 2 needs 4.66 GiB). The optimizer therefore chooses
attention LoRA on blocks 3-11, and the prediction below for that run is the test of that choice.

## Predictions

| # | quantity (mean over seeds 42, 123, 456) | prediction | holds if |
|---|---|---|---|
| P1 | DeiT-B: blocks 6-11 − all blocks | **−1.4 pp**, range [−2.5, −0.7] | mean inside the range **and** negative in all 3 seeds |
| P2 | DeiT-B: blocks 3-11 − all blocks | **−0.7 pp**, range [−1.4, −0.2] | mean inside the range |
| P3 | ViT-B: blocks 3-11 − all blocks (`results_v2/lora_attn_r8`, seeds 42/123/456) | **−0.8 pp**, range [−1.4, −0.2] | mean inside the range |
| P4 | ordering on DeiT-B | all blocks > blocks 3-11 > blocks 6-11 (means) | both inequalities hold |
| P5 | DeiT-B all-blocks seed 456, absolute | 86.5 ± 0.5 % | inside the range |

**How P2 and P3 split the possible outcomes:**
- **P3 holds, P2 fails:** the accuracy curve has the predicted shape on ViT-B but does not transfer to DeiT-B
  (different pretraining).
- **Both fail in the same direction:** the linear shape is wrong. Two alternatives to report:
  - **"step":** most of the loss comes from skipping the first few blocks, so blocks 3-11 ≈ blocks 6-11, with a
    difference of about −1.1 pp or worse;
  - **"flat":** blocks 0-2 hardly matter, so the difference is about −0.2 pp or better.
- **Both hold:** the rule "the loss is about β·k/L" transfers to an unseen model at an unmeasured point. Given 3
  seeds, call it consistent with the data, not established.

**Caveats fixed in advance:**
- **Seed count.** 3 seeds (the source studies have 5), so "detectable" (mean larger than its std, same sign in at
  least 4 of 5 seeds) cannot be applied. P1 uses "same sign in all 3 seeds" instead.
- **The ViT-B reference** (`results_v2/lora_attn_r8`) was trained by an earlier code version. The training code is
  unchanged; only registry entries were added. Seeded ViT-B GPU runs differ by 4–15 test images between code
  versions (0.04–0.15 pp), below the ranges above.
- **The ranges are judgement**, about ±0.6 pp around the central value. They are not statistical intervals.

## Cost predictions (the same runs; easier, but they check the training runs agree with the profiler)

| # | quantity | prediction | basis |
|---|---|---|---|
| C1 | peak training memory, batch 64: DeiT-B blocks 3-11, ViT-B blocks 3-11 | 4.260 ± 0.005 GiB | memory sweep (profiler and training agree within 0.004 GiB so far) |
| C2 | peak training memory, batch 64: DeiT-B blocks 6-11 | 3.061 ± 0.005 GiB | memory sweep; ViT-B blocks 6-11 trained at 3.0610 |
| C3 | average training epoch time (epochs 2+): blocks 3-11, DeiT-B and ViT-B | 279 ± 10 s | linear in reach between ViT-B all blocks (323 s) and blocks 6-11 (234 s) |
| C4 | average backward time per batch: blocks 3-11 | 177 ± 8 ms | the same interpolation (ViT-B 236.9 and 116.5 ms) |
| C5 | average training epoch time: DeiT-B blocks 6-11 | 234 ± 5 s | ViT-B blocks 6-11 |
