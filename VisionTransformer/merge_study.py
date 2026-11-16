#!/usr/bin/env python
"""
merge_study.py — merge / unmerge correctness, drift and cost for BOTH
implementations of one LoRA experiment. NO TRAINING.

    python merge_study.py --experiment lora_attn_r8 --seed 42                      # perturbed-B math check
    python merge_study.py --experiment lora_attn_r8 --seed 42 \
        --adapter results_adapters/lora_attn_r8/library_seed42_adapter.pt --eval-test   # trained adapter

What it measures (fp32 on the chosen device):
  correctness  merged vs unmerged logits (max|d|, top-1 agreement), per
               implementation, and library vs scratch before/after merging.
  drift        merge -> unmerge round trips (1 and --cycles): max |W - W0| over
               every adapted base weight, and the logit change vs the original.
  cost         time to merge-all / unmerge-all and the transient GPU memory,
               against loading a full ViT-B state dict from disk (warm page
               cache, i.e. a LOWER bound for the full-model reload).
  accuracy     (--eval-test, needs a TRAINED adapter) CIFAR-100 test accuracy
               unmerged / merged / after --cycles round trips, per
               implementation, and the prediction agreement.

Perturbed B (no --adapter) tests the merge ARITHMETIC; it says nothing about
trained values. Output: <out>/<experiment>/seed<seed>.json
"""
import argparse
import copy
import gc
import json
import tempfile
import time
from pathlib import Path

import torch

GiB = 1024 ** 3
MiB = 1024 ** 2


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


@torch.no_grad()
def _logits(model, x):
    out = model(x)
    return out.logits if hasattr(out, "logits") else out


def _max_diff(a, b):
    return float((a.float() - b.float()).abs().max())


def _agree(a, b):
    return float((a.argmax(-1) == b.argmax(-1)).float().mean())


def _weights(model):
    from models import lora_base_weights
    return {k: v.detach().clone() for k, v in lora_base_weights(model).items()}


def _drift(model, x, cycles, ref_logits, w0):
    """Round-trip merge->unmerge `cycles` times; report weight and logit drift."""
    from models import merge_all, unmerge_all, lora_base_weights
    out = {}
    done = 0
    for target in sorted({1, cycles}):
        while done < target:
            merge_all(model)
            unmerge_all(model)
            done += 1
        cur = lora_base_weights(model)
        out[f"after_{target}_cycles"] = {
            "max_abs_weight_drift": max(float((cur[k].float() - w0[k].float()).abs().max()) for k in w0),
            "max_abs_logit_change": _max_diff(_logits(model, x), ref_logits),
        }
    return out


def _predictions(model, loader, device, dtype, max_batches):
    preds, labels_all = [], []
    for i, (images, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        out = _logits(model, images.to(device, dtype))
        preds.append(out.argmax(-1).cpu())
        labels_all.append(labels)
    return torch.cat(preds), torch.cat(labels_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapter", default=None, help="trained adapter file; default: perturbed B")
    ap.add_argument("--b-std", type=float, default=0.02)
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--eval-max-batches", type=int, default=0, help="0 = whole test set")
    ap.add_argument("--n-inputs", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results_merge")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_path = Path(args.out) / args.experiment / f"seed{args.seed}.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {out_path}")

    from experiments import EXPERIMENTS, has_scratch_impl
    from models import (build_paired_models, apply_adapter_reference, load_adapter_file,
                        perturbed_adapter_reference, merge_all, unmerge_all)
    from train_utils import code_provenance

    cfg = copy.deepcopy(EXPERIMENTS[args.experiment])
    if cfg.lora is None:
        raise SystemExit(f"{args.experiment} has no LoRA adapters; nothing to merge")
    cfg.implementations = ["library", "scratch"] if has_scratch_impl(cfg.model_local_path) else ["library"]
    models = build_paired_models(cfg, seed=args.seed)

    if args.adapter:
        ref = load_adapter_file(args.adapter)
        source = f"trained:{args.adapter}"
    else:
        ref = perturbed_adapter_reference(models["library"], seed=args.seed, b_std=args.b_std)
        source = f"perturbed_B(std={args.b_std}, seed={args.seed})"
        if args.eval_test:
            print("[merge] WARNING: --eval-test with perturbed B measures arithmetic, not a trained model.")
    for m in models.values():
        apply_adapter_reference(m, ref)
    # Pristine CPU copies, taken BEFORE any merge: drift and accuracy are
    # measured against these, otherwise a first round trip (already done by the
    # correctness section) would silently be absorbed into the baseline.
    pristine = {k: copy.deepcopy(m).cpu().eval() for k, m in models.items()}
    for m in models.values():
        m.eval().to(device)

    x = torch.randn(args.n_inputs, 3, 224, 224, generator=torch.Generator().manual_seed(args.seed)).to(device)
    report = {"experiment": args.experiment, "seed": args.seed, "adapter_source": source,
              "device": str(device), "cycles": args.cycles, "provenance": code_provenance()}

    # ── correctness (fp32) ──
    unmerged = {k: _logits(m, x) for k, m in models.items()}
    merged = {}
    correctness = {}
    for k, m in models.items():
        merge_all(m)
        merged[k] = _logits(m, x)
        unmerge_all(m)
        correctness[k] = {"max_abs_diff_merged_vs_unmerged": _max_diff(merged[k], unmerged[k]),
                          "top1_agreement": _agree(merged[k], unmerged[k]),
                          "max_abs_logit": float(unmerged[k].abs().max())}
    if "scratch" in models:
        correctness["library_vs_scratch_unmerged"] = {"max_abs_diff": _max_diff(unmerged["library"], unmerged["scratch"])}
        correctness["library_vs_scratch_merged"] = {"max_abs_diff": _max_diff(merged["library"], merged["scratch"])}
    report["correctness_fp32"] = correctness

    # ── drift (fp32) ──
    drift = {}
    dtypes = [("fp32", torch.float32)]
    for dname, dtype in dtypes:
        drift[dname] = {}
        for k, m in models.items():
            mc = copy.deepcopy(pristine[k]).to(device=device, dtype=dtype).eval()
            xin = x.to(dtype)
            ref_logits = _logits(mc, xin)
            w0 = _weights(mc)
            drift[dname][k] = _drift(mc, xin, args.cycles, ref_logits, w0)
            del mc
    report["drift"] = drift

    # ── cost (fp32) ──
    cost = {}
    for k, m in models.items():
        times_merge, times_unmerge, transient = [], [], []
        for _ in range(5):
            _sync()
            if device.type == "cuda":
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            merge_all(m)
            _sync()
            times_merge.append(time.perf_counter() - t0)
            if device.type == "cuda":
                transient.append(torch.cuda.max_memory_allocated() - base)
            t0 = time.perf_counter()
            unmerge_all(m)
            _sync()
            times_unmerge.append(time.perf_counter() - t0)
        cost[k] = {"merge_all_ms_median": 1e3 * _median(times_merge),
                   "unmerge_all_ms_median": 1e3 * _median(times_unmerge),
                   "merge_transient_mib_max": (max(transient) / MiB) if transient else None}
    # Artifact timings use the scratch model (a plain state dict); a library-only family uses the PEFT model.
    art = models["scratch"] if "scratch" in models else models["library"]
    with tempfile.TemporaryDirectory() as d:
        full = Path(d) / "full_state.pt"
        torch.save(art.state_dict(), full)
        t_full = []
        for _ in range(3):
            t0 = time.perf_counter()
            sd = torch.load(full, map_location=device, weights_only=True)
            _sync()
            t_full.append(time.perf_counter() - t0)
            del sd
        full_mib = full.stat().st_size / MiB
        ad = Path(d) / "adapter.pt"
        from models import export_adapter
        export_adapter(art, ad)
        t_ad = []
        for _ in range(3):
            t0 = time.perf_counter()
            apply_adapter_reference(art, load_adapter_file(ad))
            _sync()
            t_ad.append(time.perf_counter() - t0)
        cost["reload_full_state_dict"] = {"size_mib": full_mib, "load_ms_median": 1e3 * _median(t_full),
                                          "note": "warm page cache: lower bound for a cold reload"}
        cost["swap_adapter_from_file"] = {"size_mib": ad.stat().st_size / MiB, "load_apply_ms_median": 1e3 * _median(t_ad)}
    report["cost_fp32"] = cost

    # ── accuracy on the test set (trained adapters) ──
    if args.eval_test:
        from train_utils import get_loaders
        from experiments import checkpoint_normalization
        mean, std, _ = checkpoint_normalization(cfg.model_local_path, cfg.arch)
        # same normalization the model was trained with (0.5 for ViT-B/L: the legacy loader call)
        norm_kwargs = {} if (mean, std) == ((0.5,) * 3, (0.5,) * 3) else {"norm": (mean, std)}
        test_loader = get_loaders(batch_size=64, seed=args.seed, **norm_kwargs)[2]
        acc = {}
        for k in models:
            m = copy.deepcopy(pristine[k]).to(device).eval()
            p_u, y = _predictions(m, test_loader, device, torch.float32, args.eval_max_batches)
            merge_all(m)
            p_m, _ = _predictions(m, test_loader, device, torch.float32, args.eval_max_batches)
            unmerge_all(m)
            for _ in range(args.cycles):
                merge_all(m)
                unmerge_all(m)
            p_c, _ = _predictions(m, test_loader, device, torch.float32, args.eval_max_batches)
            acc[k] = {"n_test": int(y.numel()),
                      "acc_unmerged": 100 * float((p_u == y).float().mean()),
                      "acc_merged": 100 * float((p_m == y).float().mean()),
                      f"acc_after_{args.cycles}_cycles": 100 * float((p_c == y).float().mean()),
                      "pred_agreement_merged_vs_unmerged": float((p_u == p_m).float().mean()),
                      "pred_agreement_cycled_vs_unmerged": float((p_u == p_c).float().mean())}
        report["test_accuracy"] = acc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    gc.collect()


if __name__ == "__main__":
    main()
