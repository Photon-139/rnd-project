"""Score the job-2 runs (DeiT-B + ViT-B suffixes) against memory_model/accuracy_predictions_2026-11-10.md.
The prediction ranges are copied here verbatim; the predictions file itself is never edited.
Usage:  python memory_model/score_predictions.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [42, 123, 456]


def run(exp, seed, d="results_v2"):
    return json.loads((ROOT / d / exp / f"library_seed{seed}.json").read_text())


def acc(exp):
    return np.array([run(exp, s)["metrics"]["test_acc"] for s in SEEDS])


def ms(x):
    return f"{np.mean(x):+.2f} ± {np.std(x, ddof=1):.2f}" if len(x) > 1 else f"{x[0]:.2f}"


def verdict(ok):
    return "HOLDS" if ok else "FAILS"


def main():
    d_all, d36, d311 = acc("deitb_lora_attn_r8"), acc("deitb_lora_attn_r8_L6-11"), acc("deitb_lora_attn_r8_L3-11")
    v_all, v311 = acc("lora_attn_r8"), acc("lora_attn_r8_L3-11")
    v611 = acc("lora_attn_r8_L6-11")

    print("Test accuracy per seed (42, 123, 456):")
    for name, a in [("DeiT-B all blocks", d_all), ("DeiT-B blocks 3-11", d311), ("DeiT-B blocks 6-11", d36),
                    ("ViT-B all blocks", v_all), ("ViT-B blocks 3-11", v311), ("ViT-B blocks 6-11", v611)]:
        print(f"  {name:22} {np.round(a, 2)}  mean {np.mean(a):.2f} ± {np.std(a, ddof=1):.2f}")

    p1, p2, p3 = d36 - d_all, d311 - d_all, v311 - v_all
    print("\nPredictions (paired per seed, vs all blocks of the same model):")
    print(f"P1 DeiT-B 6-11 − all: {ms(p1)}  per seed {np.round(p1, 2)}  predicted −1.4 [−2.5, −0.7], all negative"
          f" -> {verdict(-2.5 <= p1.mean() <= -0.7 and (p1 < 0).all())}")
    print(f"P2 DeiT-B 3-11 − all: {ms(p2)}  per seed {np.round(p2, 2)}  predicted −0.7 [−1.4, −0.2]"
          f" -> {verdict(-1.4 <= p2.mean() <= -0.2)}")
    print(f"P3 ViT-B  3-11 − all: {ms(p3)}  per seed {np.round(p3, 2)}  predicted −0.8 [−1.4, −0.2]"
          f" -> {verdict(-1.4 <= p3.mean() <= -0.2)}")
    print(f"P4 DeiT-B order all > 3-11 > 6-11: {d_all.mean():.2f} > {d311.mean():.2f} > {d36.mean():.2f}"
          f" -> {verdict(d_all.mean() > d311.mean() > d36.mean())}")
    s456 = d_all[2]
    print(f"P5 DeiT-B all, seed 456: {s456:.2f}  predicted 86.5 ± 0.5 -> {verdict(86.0 <= s456 <= 87.0)}")

    # reference: ViT-B 6-11 on the same three seeds, and the fraction of the second-half loss reached at k = 3
    r611 = v611 - v_all
    print(f"\nreference ViT-B 6-11 − all (same 3 seeds): {ms(r611)}")
    print(f"share of the k=6 loss already lost at k=3: DeiT-B {p2.mean() / p1.mean():.2f}, "
          f"ViT-B {p3.mean() / r611.mean():.2f}  (linear model: 0.50)")

    print("\nCosts (mean over seeds; epoch times exclude epoch 1):")

    def cost(exp):
        rs = [run(exp, s) for s in SEEDS]
        peak = [r["metrics"]["peak_memory_gb_train"] for r in rs]
        ep = [np.mean([p["train_time_sec"] for p in r["per_epoch"][1:]]) for r in rs]
        bwd = [r["metrics"]["avg_batch_backward_sec"] * 1e3 for r in rs]
        return np.mean(peak), np.ptp(peak), np.mean(ep), np.mean(bwd)

    for c, exp, pk, ept, bw in [("C1", "deitb_lora_attn_r8_L3-11", 4.260, 279, 177),
                                ("C1", "lora_attn_r8_L3-11", 4.260, 279, 177),
                                ("C2/C5", "deitb_lora_attn_r8_L6-11", 3.061, 234, None)]:
        p, spread, e, b = cost(exp)
        line = (f"{c:5} {exp:26} peak {p:.4f} GiB (seed spread {spread:.4f}; pred {pk} ± 0.005 -> "
                f"{verdict(abs(p - pk) <= 0.005)}), epoch {e:.0f} s (pred {ept} -> "
                f"{verdict(abs(e - ept) <= (10 if ept == 279 else 5))})")
        if bw:
            line += f", backward {b:.1f} ms (pred {bw} ± 8 -> {verdict(abs(b - bw) <= 8)})"
        print(line)

    print("\nOther run facts:")
    for exp in ["deitb_lora_attn_r8", "deitb_lora_attn_r8_L3-11", "deitb_lora_attn_r8_L6-11", "lora_attn_r8_L3-11"]:
        rs = [run(exp, s) for s in SEEDS]
        be = [r["metrics"]["best_epoch"] for r in rs]
        vl = [r["per_epoch"][r["metrics"]["best_epoch"] - 1]["val_loss"] for r in rs]
        code = {r.get("provenance", {}).get("source_sha256", "?")[:12] for r in rs}
        print(f"  {exp:26} best epochs {be}, val loss at kept epoch {np.round(vl, 4)}, code {code}")


if __name__ == "__main__":
    main()
