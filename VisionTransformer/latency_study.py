#!/usr/bin/env python
"""
latency_study.py — prediction latency on real CIFAR-100 test images. NO TRAINING.

profile_memory.py times the forward pass alone, on one random-tensor batch that
already sits on the GPU. This script measures what a deployed model would do,
and repeats profile_memory's model-only timing on the same images for reference:

    end_to_end   batch 1 over the test images: take the stored test image, preprocess
                 it (resize to 224, normalize), copy it to the device, forward pass,
                 predicted class back to the host. Each stage is timed separately
                 (synchronized), so p50/p90/p99 are reported per stage and in total,
                 together with the accuracy of those predictions.
    sweep        batch sizes 1..128 over preprocessed test images: per-batch latency
                 (copy + forward + argmax) and throughput in images/s, plus the
                 model-only time (input already on the device) and peak memory.
    cuda_graph   model-only latency at batch 1 and 64 with the forward pass captured in
                 a CUDA graph (one launch per pass instead of one per GPU operation).
                 If the unmerged-LoRA penalty is kernel-launch overhead, it shrinks here.

Variants: unmerged and merged for LoRA (merged = adapters folded in and removed),
dense for full fine-tuning. --order flips which LoRA variant runs first, so repeats
can alternate it. Run ONE (experiment, impl, rep) per process; the job script runs
several reps to measure the variation between processes.

Full fine-tuning has no stored checkpoint (deleted after testing), so it is the
pretrained backbone with a new head: same shapes, so same time, but its accuracy
here is meaningless and is flagged as such.

Output: <out>/<experiment>/<impl>_seed<seed>_<device>_rep<rep>.json
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


def _cuda_dev(device) -> bool:
    return device.type == "cuda"


def _sync(device):
    if _cuda_dev(device):
        torch.cuda.synchronize()


def _logits(model, x):
    out = model(x)
    return out.logits if hasattr(out, "logits") else out


def _stats(ts_s):
    """Summary of a list of durations in seconds, reported in ms."""
    ts = sorted(ts_s)
    n = len(ts)
    if n == 0:
        return None
    q = lambda f: 1e3 * ts[min(n - 1, int(f * (n - 1)))]
    return {"n": n, "mean_ms": 1e3 * sum(ts) / n, "min_ms": 1e3 * ts[0], "p50_ms": 1e3 * ts[n // 2],
            "p90_ms": q(0.90), "p99_ms": q(0.99), "max_ms": 1e3 * ts[-1]}


def test_images(n, data_dir=None):
    """The first n CIFAR-100 test images as stored (32x32 PIL images) and their labels.
    torchvision keeps the whole test set in memory, so 'loading' an image here is an
    in-memory lookup, not a disk read."""
    import torchvision.datasets as datasets
    from experiments import DATA_DIR
    ds = datasets.CIFAR100(root=data_dir or DATA_DIR, train=False, download=False, transform=None)
    n = len(ds) if n is None else min(n, len(ds))
    return [ds[i] for i in range(n)]


# ------------------------------------------------------------------------------ measurements
@torch.inference_mode()
def end_to_end(model, items, tf, device, warmup):
    """Batch 1, one stored image at a time. The first `warmup` images are run but not timed."""
    stages = {"preprocess": [], "to_device": [], "forward": [], "to_host": [], "total": []}
    correct = 0
    for i, (img, label) in enumerate(items):
        t0 = time.perf_counter()
        x = tf(img).unsqueeze(0)
        t1 = time.perf_counter()
        x = x.to(device)
        _sync(device)
        t2 = time.perf_counter()
        logits = _logits(model, x)
        _sync(device)
        t3 = time.perf_counter()
        pred = int(logits.argmax(dim=-1).item())
        t4 = time.perf_counter()
        correct += int(pred == label)
        if i >= warmup:
            for k, a, b in (("preprocess", t0, t1), ("to_device", t1, t2), ("forward", t2, t3),
                            ("to_host", t3, t4), ("total", t0, t4)):
                stages[k].append(b - a)
    return {"warmup_images": warmup, "n_images": len(items),
            "accuracy_pct": 100.0 * correct / len(items),
            "stages": {k: _stats(v) for k, v in stages.items()}}


@torch.inference_mode()
def sweep(model, xs, ys, device, batch_sizes, warmup_batches=3):
    """Per batch size: every full batch of the preprocessed images, timed one by one."""
    out = {}
    pin = _cuda_dev(device)
    for bs in batch_sizes:
        nb = len(xs) // bs
        if nb <= warmup_batches:
            out[str(bs)] = {"skipped": f"only {nb} batches of {bs}"}
            continue
        try:
            gc.collect()
            if _cuda_dev(device):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            batches = [xs[i * bs:(i + 1) * bs] for i in range(nb)]
            if pin:
                batches = [b.pin_memory() for b in batches]
            e2e, fwd, correct, seen = [], [], 0, 0
            for j, b in enumerate(batches):
                t0 = time.perf_counter()
                xb = b.to(device, non_blocking=pin)
                pred = _logits(model, xb).argmax(dim=-1).cpu()   # .cpu() waits for the GPU
                t1 = time.perf_counter()
                # model-only: same batch, already on the device
                _sync(device)
                t2 = time.perf_counter()
                _logits(model, xb)
                _sync(device)
                t3 = time.perf_counter()
                if j >= warmup_batches:
                    e2e.append(t1 - t0)
                    fwd.append(t3 - t2)
                    correct += int((pred == ys[j * bs:(j + 1) * bs]).sum())
                    seen += bs
            row = {"batches_timed": len(e2e), "warmup_batches": warmup_batches,
                   "copy_forward_argmax": _stats(e2e), "forward_only": _stats(fwd),
                   "throughput_img_per_s": seen / sum(e2e),
                   "accuracy_pct": 100.0 * correct / seen}
            if _cuda_dev(device):
                row["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / GiB
            out[str(bs)] = row
        except torch.cuda.OutOfMemoryError as exc:
            out[str(bs)] = {"oom": True, "error": str(exc)[:200]}
            break
        finally:
            batches = None
    return out


def cuda_graph(model, xs, device, batch_sizes=(1, 64), iters=(100, 20), warmup=10):
    """Model-only latency with the forward pass captured in a CUDA graph. The graph is
    checked against the eager output on the same input before it is timed."""
    out = {}
    if not _cuda_dev(device):
        return {"skipped": "CUDA graphs need a GPU"}
    for bs, n in zip(batch_sizes, iters):
        try:
            static_x = xs[:bs].to(device).clone()
            with torch.inference_mode():
                ref = _logits(model, static_x).clone()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        _logits(model, static_x)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    static_out = _logits(model, static_x)
            g.replay()
            torch.cuda.synchronize()
            max_diff = float((static_out - ref).abs().max())
            for _ in range(warmup):
                g.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(n):
                t0 = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            out[str(bs)] = {"latency": _stats(ts), "max_abs_diff_vs_eager": max_diff, "warmup": warmup}
            del g, static_x, static_out, ref
        except Exception as exc:  # noqa: BLE001 — capture can fail for some models; record why
            out[str(bs)] = {"failed": f"{type(exc).__name__}: {str(exc)[:300]}"}
        gc.collect()
        torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--impl", choices=["library", "scratch"], default="library")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapter", default=None, help="trained seed-42 adapter file (LoRA; required)")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--order", choices=["unmerged-first", "merged-first"], default="unmerged-first")
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (torch.set_num_threads)")
    ap.add_argument("--n-e2e", type=int, default=None, help="images for end_to_end (default: all 10,000)")
    ap.add_argument("--e2e-warmup", type=int, default=50)
    ap.add_argument("--n-sweep", type=int, default=2048, help="images for the batch-size sweep")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="results_latency")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.threads:
        torch.set_num_threads(args.threads)
    out_path = Path(args.out) / args.experiment / f"{args.impl}_seed{args.seed}_{device.type}_rep{args.rep}.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {out_path}")

    from experiments import EXPERIMENTS, has_scratch_impl, checkpoint_normalization
    from models import build_paired_models, apply_adapter_reference, load_adapter_file, merged_copy
    from train_utils import code_provenance, make_transform

    cfg = copy.deepcopy(EXPERIMENTS[args.experiment])
    cfg.implementations = ["library", "scratch"] if has_scratch_impl(cfg.model_local_path) else ["library"]
    if args.impl not in cfg.implementations:
        raise SystemExit(f"{args.experiment} has no {args.impl!r} implementation")
    has_lora = cfg.lora is not None
    if has_lora and not args.adapter:
        # Never time an untrained adapter here: the accuracy would silently describe nothing.
        raise SystemExit("LoRA experiments need --adapter (the trained seed-42 adapter file)")

    models = build_paired_models(cfg, seed=args.seed)
    model = models.pop(args.impl)
    models.clear()
    gc.collect()
    if has_lora:
        apply_adapter_reference(model, load_adapter_file(args.adapter))
    model = model.cpu().eval()

    mean, std, norm_source = checkpoint_normalization(cfg.model_local_path, cfg.arch)
    tf = make_transform(mean, std)
    items = test_images(args.n_e2e, args.data_dir)
    sweep_items = items[:args.n_sweep] if len(items) >= args.n_sweep else test_images(args.n_sweep, args.data_dir)
    xs = torch.stack([tf(img) for img, _ in sweep_items])
    ys = torch.tensor([y for _, y in sweep_items])

    report = {
        "experiment": args.experiment, "implementation": args.impl, "seed": args.seed, "rep": args.rep,
        "order": args.order if has_lora else None,
        "adapter_source": f"trained:{args.adapter}" if has_lora else "none (full fine-tuning)",
        "weights_note": (None if has_lora else
                         "pretrained backbone + untrained head (no stored checkpoint): latency valid, "
                         "accuracy meaningless"),
        "inputs": "CIFAR-100 test images (in memory), resized to 224 and normalized on the CPU",
        "normalization": {"mean": list(mean), "std": list(std), "source": norm_source},
        "env": {"torch": torch.__version__, "device": str(device), "host": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "torch_threads": torch.get_num_threads(),
                "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32},
        "provenance": code_provenance(),
        "variants": {},
    }
    if _cuda_dev(device):
        report["env"]["gpu"] = torch.cuda.get_device_name(0)

    variants = ["dense"] if not has_lora else (["unmerged", "merged"] if args.order == "unmerged-first"
                                               else ["merged", "unmerged"])
    for v in variants:
        m = (merged_copy(model) if v == "merged" else copy.deepcopy(model)).eval().to(device)
        print(f"[latency] {args.experiment} {v}: end_to_end on {len(items)} images")
        entry = {"num_params": sum(p.numel() for p in m.parameters()),
                 "end_to_end": end_to_end(m, items, tf, device, args.e2e_warmup)}
        print(f"[latency] {args.experiment} {v}: sweep {args.batch_sizes}")
        entry["sweep"] = sweep(m, xs, ys, device, args.batch_sizes)
        if not args.no_graph:
            entry["cuda_graph"] = cuda_graph(m, xs, device)
        report["variants"][v] = entry
        del m
        gc.collect()
        if _cuda_dev(device):
            torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
