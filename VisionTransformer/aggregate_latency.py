#!/usr/bin/env python
"""
aggregate_latency.py — tables from latency_study.py outputs (results_latency/<exp>/*.json).

GPU values: mean of the per-process medians (p50) over the reps, with the range over reps
(min-max), so the variation between processes is visible. CPU values: the single CPU process.

    python aggregate_latency.py [--results results_latency] [--out report_latency]
"""
import argparse
import glob
import json
from pathlib import Path


def _rng(xs, fmt="{:.2f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "--"
    m = sum(xs) / len(xs)
    return (fmt + " ({}–{})").format(m, fmt.format(min(xs)), fmt.format(max(xs))) if len(xs) > 1 else fmt.format(m)


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_latency")
    ap.add_argument("--out", default="report_latency")
    args = ap.parse_args()

    rows, stats = [], {}
    for d in sorted(glob.glob(f"{args.results}/*/")):
        exp = Path(d).name
        reps = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/library_seed42_cuda_rep*.json"))]
        cpu_f = glob.glob(f"{d}/library_seed42_cpu_rep1.json")
        cpu = json.load(open(cpu_f[0])) if cpu_f else None
        if not reps:
            continue
        for v in reps[0]["variants"]:
            R = [r["variants"][v] for r in reps if v in r["variants"]]
            s = {
                "n_reps": len(R),
                "orders": [r.get("order") for r in reps],
                "accuracy_pct": [_get(x, "end_to_end", "accuracy_pct") for x in R],
                "n_images": [_get(x, "end_to_end", "n_images") for x in R],
                "e2e_total_p50": [_get(x, "end_to_end", "stages", "total", "p50_ms") for x in R],
                "e2e_total_p99": [_get(x, "end_to_end", "stages", "total", "p99_ms") for x in R],
                "e2e_forward_p50": [_get(x, "end_to_end", "stages", "forward", "p50_ms") for x in R],
                "e2e_preprocess_p50": [_get(x, "end_to_end", "stages", "preprocess", "p50_ms") for x in R],
                "e2e_copies_p50": [(_get(x, "end_to_end", "stages", "to_device", "p50_ms") or 0)
                                   + (_get(x, "end_to_end", "stages", "to_host", "p50_ms") or 0) for x in R],
                "graph_b1_p50": [_get(x, "cuda_graph", "1", "latency", "p50_ms") for x in R],
                "graph_b64_p50": [_get(x, "cuda_graph", "64", "latency", "p50_ms") for x in R],
                "graph_max_abs_diff": max((_get(x, "cuda_graph", b, "max_abs_diff_vs_eager") or 0)
                                          for x in R for b in ("1", "64")),
                "graph_failed": [_get(x, "cuda_graph", b, "failed") for x in R for b in ("1", "64")
                                 if _get(x, "cuda_graph", b, "failed")],
                "sweep": {},
                "cpu_forward_p50": _get(cpu, "variants", v, "end_to_end", "stages", "forward", "p50_ms") if cpu else None,
                "cpu_total_p50": _get(cpu, "variants", v, "end_to_end", "stages", "total", "p50_ms") if cpu else None,
                "cpu_threads": _get(cpu, "env", "torch_threads") if cpu else None,
                "lora": reps[0].get("adapter_source", "").startswith("trained"),
            }
            for bs in reps[0]["variants"][v]["sweep"]:
                s["sweep"][bs] = {
                    "forward_only_p50": [_get(x, "sweep", bs, "forward_only", "p50_ms") for x in R],
                    "copy_forward_argmax_p50": [_get(x, "sweep", bs, "copy_forward_argmax", "p50_ms") for x in R],
                    "throughput_img_per_s": [_get(x, "sweep", bs, "throughput_img_per_s") for x in R],
                    "peak_allocated_gb": [_get(x, "sweep", bs, "peak_allocated_gb") for x in R],
                }
            stats[f"{exp}/{v}"] = s
            rows.append((exp, v, s))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    L = ["# Prediction latency on CIFAR-100 test images (latency_study.py)", "",
         "GPU: mean of the per-process medians, (min–max over processes). Batch 1 = one stored test image at a time, "
         "end to end = preprocess + copy to GPU + forward + predicted class back. Accuracy is that of the "
         "end-to-end predictions (full fine-tuning has an untrained head, so its accuracy is meaningless).", "",
         "## Batch 1", "",
         "| experiment | variant | procs | acc % | end to end p50 ms | p99 ms (max) | forward p50 ms | preprocess ms | copies ms | CUDA graph forward ms | CPU forward ms |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for exp, v, s in rows:
        acc = s["accuracy_pct"][0]
        cpu_s = "--" if s["cpu_forward_p50"] is None else f"{s['cpu_forward_p50']:.1f}"
        p99 = max(x for x in s["e2e_total_p99"] if x is not None)
        L.append(f"| {exp} | {v} | {s['n_reps']} | {acc:.2f}{'' if s['lora'] else ' (untrained head)'} | "
                 f"{_rng(s['e2e_total_p50'])} | {p99:.2f} | "
                 f"{_rng(s['e2e_forward_p50'])} | {_rng(s['e2e_preprocess_p50'])} | {_rng(s['e2e_copies_p50'], '{:.3f}')} | "
                 f"{_rng(s['graph_b1_p50'])} | {cpu_s} |")
    L += ["", "## Batch-size sweep (2,048 test images; per batch: copy + forward + argmax; forward only)", "",
          "| experiment | variant | batch | copy+forward+argmax p50 ms | forward only p50 ms | images/s | peak GiB |",
          "|---|---|---|---|---|---|---|"]
    for exp, v, s in rows:
        for bs, r in s["sweep"].items():
            if r["throughput_img_per_s"][0] is None:
                continue
            L.append(f"| {exp} | {v} | {bs} | {_rng(r['copy_forward_argmax_p50'])} | {_rng(r['forward_only_p50'])} | "
                     f"{_rng(r['throughput_img_per_s'], '{:.0f}')} | {_rng(r['peak_allocated_gb'], '{:.3f}')} |")
    fails = [(e, v, s["graph_failed"]) for e, v, s in rows if s["graph_failed"]]
    L += ["", "## Checks", "",
          f"- CUDA-graph outputs vs eager: max |Δlogit| over all configs = "
          f"{max(s['graph_max_abs_diff'] for _, _, s in rows):.2e}",
          f"- CUDA-graph capture failures: {fails if fails else 'none'}"]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print(f"{len(rows)} (experiment, variant) rows; wrote {out}/report.md, stats.json")


if __name__ == "__main__":
    main()
