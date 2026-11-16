#!/usr/bin/env python
"""
profile_memory.py — standalone memory + latency profiler. NO TRAINING.

Memory footprints and kernel timings depend on tensor SHAPES, not weight
VALUES, so freshly built paired models are as valid as trained ones. Run ONE
(experiment, impl, mode) per process so the CUDA context and allocator start
clean, ideally on an otherwise idle slice:

    python profile_memory.py --experiment lora_attn_r8 --impl library --mode train
    python profile_memory.py --experiment lora_attn_r8 --impl library --mode inference

Modes
    train      exact per-phase memory of one AdamW training step (fp32,
               cross-entropy) at several batch sizes: resident_before_fwd,
               activation, grads, opt_transient, peak, plus a linear fit and
               OOM boundaries.
    inference  resident weights + peak memory + latency at batch 1 and 64, fp32,
               for {unmerged, merged} (merged = adapters folded in AND removed;
               dense baselines have only 'unmerged'). fp32 only, by design.

LoRA adapters carry a perturbed non-zero B (seeded) unless --adapter points at
a trained adapter file, so merged/unmerged variants do real work. Perturbed
values do not change shapes, hence not memory or latency.

Output: <out>/<experiment>/<impl>_seed<seed>_<mode>.json
"""
import argparse
import copy
import gc
import json
import os
import socket
import time
from pathlib import Path

import torch

GiB = 1024 ** 3
MiB = 1024 ** 2


def _cuda() -> bool:
    return torch.cuda.is_available()


def _sync():
    if _cuda():
        torch.cuda.synchronize()


def _clean():
    """Collect garbage, release cached blocks, reset peak counters."""
    gc.collect()
    if _cuda():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _mem(fn_name: str) -> int:
    return getattr(torch.cuda, fn_name)() if _cuda() else 0


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def _is_oom(e: BaseException) -> bool:
    """CUDA out-of-memory, including cuBLAS/cuDNN allocation failures that surface as a plain
    RuntimeError instead of torch.cuda.OutOfMemoryError."""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    msg = str(e)
    return isinstance(e, RuntimeError) and any(s in msg for s in (
        "out of memory", "ALLOC_FAILED", "CUBLAS_STATUS_ALLOC", "CUDNN_STATUS_ALLOC"))


# --------------------------------------------------------------------------- training
def profile_train(model, use_logits, batch_size, device, lr=1e-4, warmup=3, steps=5, num_classes=100):
    """Exact per-phase memory of one training step.

    resident_before_fwd  params + AdamW state + input batch + runtime overhead
                         (grads freed by zero_grad); resident_analytic_gb is the
                         first three counted from live tensors, runtime_overhead_gb the rest
    activation           memory the forward+loss leaves alive for backward
    grads                gradient memory right after backward (activations freed)
    opt_transient        extra memory peaked inside optimizer.step()
    peak                 max over forward / backward / step
    """
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(batch_size, 3, 224, 224, generator=g).to(device)
    y = torch.randint(0, num_classes, (batch_size,), generator=g).to(device)
    rows = []
    try:
        for i in range(warmup + steps):
            opt.zero_grad(set_to_none=True)
            _sync()
            base = _mem("memory_allocated")
            # What SHOULD be resident now, from live tensors: weights + AdamW
            # state (CUDA tensors only) + the input batch and labels. Anything
            # measured above this is runtime overhead (e.g. the cuBLAS
            # workspace the caching allocator holds once the first matmul ran).
            analytic = (sum(p.numel() * p.element_size() for p in model.parameters())
                        + sum(b.numel() * b.element_size() for b in model.buffers())
                        + sum(t.numel() * t.element_size() for st in opt.state.values()
                              for t in st.values() if torch.is_tensor(t) and t.is_cuda)
                        + x.numel() * x.element_size() + y.numel() * y.element_size())
            if _cuda():
                torch.cuda.reset_peak_memory_stats()

            out = model(x)
            logits = out.logits if use_logits else out
            loss = crit(logits, y)
            _sync()
            fwd_peak = _mem("max_memory_allocated")
            after_fwd = _mem("memory_allocated")

            if _cuda():
                torch.cuda.reset_peak_memory_stats()
            loss.backward()
            _sync()
            bwd_peak = _mem("max_memory_allocated")
            after_bwd = _mem("memory_allocated")
            out = logits = loss = None

            if _cuda():
                torch.cuda.reset_peak_memory_stats()
            opt.step()
            _sync()
            step_peak = _mem("max_memory_allocated")

            if i >= warmup:
                rows.append((base, after_fwd, fwd_peak, after_bwd, bwd_peak, step_peak, analytic))
    except RuntimeError as e:                      # torch.cuda.OutOfMemoryError is a RuntimeError
        if not _is_oom(e):
            raise
        out = logits = loss = None
        return {"batch_size": batch_size, "oom": True, "error": str(e).splitlines()[0][:200]}
    finally:
        opt.zero_grad(set_to_none=True)
        opt.state.clear()
        del opt, x, y
    base, after_fwd, fwd_peak, after_bwd, bwd_peak, step_peak, analytic = (
        _median([r[k] for r in rows]) for k in range(7))
    peak = max(fwd_peak, bwd_peak, step_peak)
    return {
        "batch_size": batch_size,
        "oom": False,
        "resident_before_fwd_gb": base / GiB,
        "resident_analytic_gb": analytic / GiB,
        "runtime_overhead_gb": ((base - analytic) / GiB) if _cuda() else 0.0,
        "activation_gb": (after_fwd - base) / GiB,
        "grads_gb": (after_bwd - base) / GiB,
        "opt_transient_gb": (step_peak - after_bwd) / GiB,
        "peak_gb": peak / GiB,
        "peak_above_resident_gb": (peak - base) / GiB,
    }


def fit_line(points):
    """Least squares: peak_gb = intercept + slope * batch_size over non-OOM points."""
    pts = [(p["batch_size"], p["peak_gb"]) for p in points if not p["oom"]]
    n = len(pts)
    if n < 2:
        return None
    sx = sum(a for a, _ in pts)
    sy = sum(b for _, b in pts)
    sxx = sum(a * a for a, _ in pts)
    sxy = sum(a * b for a, b in pts)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return {"intercept_gb": (sy - slope * sx) / n, "slope_gb_per_sample": slope}


# ------------------------------------------------------------------------- inference
@torch.inference_mode()
def _bench(model, x, warmup, iters):
    for _ in range(warmup):
        model(x)
    _sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        _sync()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"iters": iters, "mean_ms": 1e3 * sum(ts) / len(ts), "p50_ms": 1e3 * ts[len(ts) // 2],
            "p95_ms": 1e3 * ts[int(0.95 * (len(ts) - 1))], "min_ms": 1e3 * ts[0]}


def profile_inference(model_cpu, has_lora, device, dtypes, batches=((1, 100), (64, 20)), warmup=10):
    from models import merged_copy
    results = {}
    for dtype_name, dtype in dtypes:
        for merged in ((False, True) if has_lora else (False,)):
            key = f"{dtype_name}_{'merged' if merged else 'unmerged'}"
            _clean()
            m = (merged_copy(model_cpu) if merged else copy.deepcopy(model_cpu)).eval()
            m = m.to(device=device, dtype=dtype)
            # One tiny warm forward BEFORE measuring, for EVERY variant: the
            # caching allocator keeps a persistent cuBLAS workspace (counted in
            # memory_allocated) once the first matmul has run, so measuring the
            # first variant before any forward and later variants after one would
            # make identical models look different. Warming each variant makes
            # the runtime overhead uniform; it is reported, not hidden.
            with torch.inference_mode():
                m(torch.randn(1, 3, 224, 224).to(device, dtype))
            _sync()
            weights = _mem("memory_allocated")
            analytic = (sum(p.numel() * p.element_size() for p in m.parameters())
                        + sum(b.numel() * b.element_size() for b in m.buffers()))
            entry = {
                "resident_weights_gb": weights / GiB,
                "param_buffer_bytes_gb": analytic / GiB,
                "runtime_overhead_gb": ((weights - analytic) / GiB) if _cuda() else 0.0,
                "num_params": sum(p.numel() for p in m.parameters()),
                "by_batch": {},
            }
            for bs, iters in batches:
                x = torch.randn(bs, 3, 224, 224, generator=torch.Generator().manual_seed(0)).to(device, dtype)
                # One untimed forward BEFORE the peak window so lazily allocated
                # library workspaces (cuBLAS) are resident for every variant
                # alike, not only in the first variant's peak.
                with torch.inference_mode():
                    m(x)
                _sync()
                if _cuda():
                    torch.cuda.reset_peak_memory_stats()
                lat = _bench(m, x, warmup if bs == 1 else 3, iters)
                row = {
                    "peak_allocated_gb": _mem("max_memory_allocated") / GiB,
                    "peak_reserved_gb": _mem("max_memory_reserved") / GiB,
                    "latency": lat,
                }
                if _cuda():
                    free, total = torch.cuda.mem_get_info()
                    # Includes CUDA context + cached blocks; meaningful only on an idle slice.
                    row["device_used_gb"] = (total - free) / GiB
                entry["by_batch"][str(bs)] = row
                del x
            results[key] = entry
            del m
    _clean()
    return results


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--impl", choices=["library", "scratch"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["train", "inference"], required=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--adapter", default=None, help="trained adapter file (LoRA); default: perturbed B")
    ap.add_argument("--b-std", type=float, default=0.02)
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu (cpu = dry run)")
    ap.add_argument("--out", default="results_profile")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if _cuda() else "cpu"))
    if device.type == "cpu":
        print("[profile] CPU dry run: memory fields are 0 and train mode is skipped.")

    out_path = Path(args.out) / args.experiment / f"{args.impl}_seed{args.seed}_{args.mode}.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {out_path}")

    from experiments import EXPERIMENTS, has_scratch_impl
    from models import (build_paired_models, adapter_reference, apply_adapter_reference,
                        load_adapter_file, perturbed_adapter_reference)
    from train_utils import code_provenance

    cfg = copy.deepcopy(EXPERIMENTS[args.experiment])
    # scratch+LoRA needs the library reference; a family without a scratch implementation (Swin) is library-only
    cfg.implementations = ["library", "scratch"] if has_scratch_impl(cfg.model_local_path) else ["library"]
    if args.impl not in cfg.implementations:
        raise SystemExit(f"{args.experiment} has no {args.impl!r} implementation (available: {cfg.implementations})")
    models = build_paired_models(cfg, seed=args.seed)
    model = models.pop(args.impl)
    lib_for_ref = models.get("library", model)
    has_lora = cfg.lora is not None

    adapter_source = "none (full fine-tuning)"
    if has_lora:
        if args.adapter:
            ref = load_adapter_file(args.adapter)
            adapter_source = f"trained:{args.adapter}"
        else:
            ref = perturbed_adapter_reference(lib_for_ref, seed=args.seed, b_std=args.b_std)
            adapter_source = f"perturbed_B(std={args.b_std}, seed={args.seed})"
        apply_adapter_reference(model, ref)
    models.clear()
    gc.collect()

    trainable_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    report = {
        "experiment": args.experiment,
        "implementation": args.impl,
        "mode": args.mode,
        "seed": args.seed,
        "adapter_source": adapter_source,
        "env": {
            "torch": torch.__version__, "device": str(device),
            "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
        "provenance": code_provenance(),
        # head + adapters (LoRA) or all weights (full FT), fp32 — same for both impls
        "trainable_artifact_mib": trainable_bytes / MiB,
    }
    if _cuda():
        props = torch.cuda.get_device_properties(0)
        report["env"].update({"gpu": props.name, "total_memory_gb": props.total_memory / GiB,
                              "sm_count": props.multi_processor_count})

    if args.mode == "train":
        if device.type != "cuda":
            raise SystemExit("train mode needs a GPU (memory counters); use --mode inference for a dry run")
        use_logits = args.impl == "library"
        sweep = []
        oom_at = None
        for bs in sorted(args.batch_sizes):
            if oom_at is not None:
                # Memory only grows with the batch: once a size runs out of memory, larger ones would
                # too. Recorded as OOM without running them, so the sweep does not hit the limit again.
                sweep.append({"batch_size": bs, "oom": True, "not_run": f"larger than the OOM batch {oom_at}"})
                continue
            _clean()
            point = profile_train(model, use_logits, bs, device)
            sweep.append(point)
            if point["oom"]:
                oom_at = bs
                print(f"[profile] batch {bs}: out of memory ({point.get('error', '')}); larger batches not run")
            for p in model.parameters():
                p.grad = None
            _clean()
        report["train"] = {"sweep": sweep, "fit": fit_line(sweep)}
    else:
        dtypes = [("fp32", torch.float32)]
        batches = ((1, 100), (64, 20)) if device.type == "cuda" else ((1, 5), (4, 2))
        report["inference"] = profile_inference(model.cpu(), has_lora, device, dtypes, batches=batches)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
