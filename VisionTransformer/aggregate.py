#!/usr/bin/env python
"""
aggregate.py — offline aggregator: per-run result JSONs -> report tables.

    python aggregate.py                                  # ./results, ./results_profile, ./results_merge
    python aggregate.py --results results --out report_out

The per-run JSONs are the ONLY source of truth (a SLURM job that dies leaves no
in-process summary). Output (in --out): report.md, per_run.csv, stats.json.

Field triage (see AUDIT_memory_metrics_and_pipeline.md §2). NEVER emitted from
sweep JSONs, because they are known-invalid for the runs in this repo:
  * inference_*  (probe ran with grads/AdamW/temporaries resident, and, for
                  library LoRA, a leaked base model)  -> use profile_memory.py
  * batch1_forward_sec_*  (rough, unmerged adapters, 20 iterations)
  * best_checkpoint_size_mb for scratch LoRA (full state dict, not the adapter)
Relabelled: activation_peak_gb_train is reported as peak_minus_static.
Per-batch phase timings exclude epoch 1 (warm-up) and are weighted by batch count.

Every run is asserted against the analytic expectations (trainable element and
tensor counts, verification gates, LR vs registry); failures are listed at the
top of the report, not hidden.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments import EXPERIMENTS, ARCH_BASELINE, arch_dims, dense_param_counts, trainable_counts
from run import calculate_stats, paired_diff_stats

GiB = 1024 ** 3
MiB = 1024 ** 2

# The original ViT-B registry. A missing run of one of these is always reported;
# newer entries are only reported when partially run (or with --require-all).
ORIGINAL_VITB = ("baseline", "baseline_lr1e-3", "baseline_lr1e-5", "lora_attn_r8", "lora_attn_r16",
                 "lora_mlp_r8", "lora_mlp_r16", "lora_attn_mlp_r8", "lora_attn_mlp_r16")


def expected_trainable(config: dict, exp_name: str, model_dims: Optional[dict] = None) -> tuple:
    """(elements, tensors) of the trainable set for an experiment config dict.
    Geometry comes from the registry entry's checkpoint config.json (ViT-B: 768/3072/12
    blocks, 100-class head -> 85,875,556 elements and 200 tensors for full FT)."""
    reg = EXPERIMENTS[exp_name]
    # Geometry precedence: the dims recorded in the run's own JSON, then the checkpoint's
    # config.json. Only ViT-B may fall back to built-in dims; any other architecture with
    # no geometry available is an error, never a silent ViT-B assumption.
    dims = model_dims or arch_dims(reg.model_local_path, strict=(reg.arch != "vitb"))
    return trainable_counts(dims, config.get("num_labels", reg.num_labels), config.get("lora"))


# ──────────────────────────── loading + per-run derivation ────────────────────────────

def load_runs(results_dir: Path):
    runs, skipped, problems = [], [], []
    for f in sorted(results_dir.glob("*/*_seed*.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            problems.append(f"{f}: unreadable JSON")
            continue
        m = r.get("metrics", {})
        if "static_memory_breakdown_gb" not in m:
            skipped.append(f"{f} (old schema: no static/activation decomposition)")
            continue
        if r.get("experiment") not in EXPERIMENTS:
            skipped.append(f"{f} (experiment '{r.get('experiment')}' not in registry)")
            continue
        r["_file"] = str(f)
        runs.append(r)
    return runs, skipped, problems


def steady_phase_ms(r: dict) -> Dict[str, float]:
    """Per-batch fwd/bwd/opt in ms from per_epoch, excluding epoch 1 when possible."""
    pe = r["per_epoch"][1:] or r["per_epoch"]
    nb = sum(e["num_train_batches"] for e in pe)
    return {k: 1e3 * sum(e[f"{k}_time_sec"] for e in pe) / nb
            for k in ("forward", "backward", "optim_step")} if nb else {}


def derive(r: dict, problems: List[str]) -> dict:
    m, cfg = r["metrics"], r["config"]
    b = m["static_memory_breakdown_gb"]
    tag = f"{r['experiment']}/{r['implementation']}/seed{r['seed']}"
    elems = round(b["grads"] * GiB / 4)
    try:
        exp_elems, exp_tensors = expected_trainable(cfg, r["experiment"], r.get("model_dims"))
    except (ValueError, FileNotFoundError) as e:
        problems.append(f"{tag}: cannot compute the analytic expectation ({e})")
        exp_elems = elems                                   # the failure is already listed as a problem
        exp_tensors = (r.get("verification") or {}).get("trainable_tensors_library")
    if elems != exp_elems:
        problems.append(f"{tag}: trainable elements {elems:,} != expected {exp_elems:,}")
    v = r.get("verification") or {}
    if v.get("trainable_tensors_library") not in (None, exp_tensors) or \
            v.get("trainable_tensors_scratch") not in (None, exp_tensors):
        problems.append(f"{tag}: trainable tensor count {v.get('trainable_tensors_library')}/"
                        f"{v.get('trainable_tensors_scratch')} != expected {exp_tensors}")
    if "library_only_ok" in v:
        # Library-only family (Swin): no pairing gates exist; the gate is models.library_only_checks.
        if v["library_only_ok"] is not True:
            problems.append(f"{tag}: library-only verification gate = {v['library_only_ok']}")
        if v.get("trainable_tensors") not in (None, exp_tensors):
            problems.append(f"{tag}: trainable tensor count {v.get('trainable_tensors')} != expected {exp_tensors}")
    else:
        for gate in ("init_checks_passed", "trainable_sets_match", "forward_ok"):
            if v.get(gate) is not True:
                problems.append(f"{tag}: verification gate '{gate}' = {v.get(gate)}")
        if cfg.get("lora") and v.get("lora_forward_ok") is not True:
            problems.append(f"{tag}: verification gate 'lora_forward_ok' = {v.get('lora_forward_ok')}")
    reg_lr = EXPERIMENTS[r["experiment"]].lr
    if cfg.get("lr") != reg_lr:
        problems.append(f"{tag}: lr {cfg.get('lr')} differs from registry {reg_lr} (CLI override?)")
    set_epochs = (r.get("training_setup") or {}).get("max_epochs")
    if set_epochs is not None and set_epochs != EXPERIMENTS[r["experiment"]].max_epochs:
        problems.append(f"{tag}: max_epochs {set_epochs} differs from registry "
                        f"{EXPERIMENTS[r['experiment']].max_epochs} (truncated run, --max-epochs?)")
    if m["peak_memory_gb_run"] != m["peak_memory_gb_train"]:
        problems.append(f"{tag}: run peak != train peak (reload/test exceeded training?)")

    ph = steady_phase_ms(r)
    return {
        "experiment": r["experiment"], "impl": r["implementation"], "seed": r["seed"],
        "lr": cfg.get("lr"), "file": r["_file"],
        "test_acc": m["test_acc"], "best_epoch": m["best_epoch"], "stopping_epoch": m["stopping_epoch"],
        "avg_epoch_train_time_sec": m["avg_epoch_train_time_sec"],
        "avg_epoch_val_time_sec": m["avg_epoch_val_time_sec"],
        "fwd_ms": ph.get("forward"), "bwd_ms": ph.get("backward"), "opt_ms": ph.get("optim_step"),
        "peak_alloc_gib": m["peak_memory_gb_train"], "peak_reserved_gib": m["peak_memory_reserved_gb_train"],
        "static_gib": m["static_memory_gb"], "params_gib": b["params"],
        "peak_minus_static_gib": m["activation_peak_gb_train"],
        "trainable_elems": elems,
        # head+adapters (LoRA) or full weights (baseline), fp32: identical for both impls
        "artifact_mib_analytic": 4 * exp_elems / MiB,
        "adapter_export_mib_measured": m.get("adapter_export_size_mb"),
        "git_commit": (r.get("provenance") or {}).get("git_commit"),
        "source_sha256": ((r.get("provenance") or {}).get("source_sha256") or "")[:12],
    }


# ──────────────────────────── statistics ────────────────────────────

METRICS = ["test_acc", "best_epoch", "stopping_epoch", "avg_epoch_train_time_sec", "fwd_ms", "bwd_ms",
           "opt_ms", "peak_alloc_gib", "peak_reserved_gib", "peak_minus_static_gib"]
PAIRED_METRICS = ["test_acc", "stopping_epoch", "avg_epoch_train_time_sec", "fwd_ms", "bwd_ms", "opt_ms",
                  "peak_alloc_gib"]


def group(rows: List[dict]) -> Dict[str, Dict[str, Dict[int, dict]]]:
    out: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for row in rows:
        out.setdefault(row["experiment"], {}).setdefault(row["impl"], {})[row["seed"]] = row
    return out


def per_impl_stats(grouped) -> dict:
    return {exp: {impl: {mt: calculate_stats([r[mt] for r in seeds.values() if r[mt] is not None])
                         for mt in METRICS}
                  for impl, seeds in impls.items()} for exp, impls in grouped.items()}


def paired(a: Dict[int, dict], b: Dict[int, dict], metric: str) -> Optional[dict]:
    """Per-seed a - b over the seeds both have."""
    seeds = sorted(set(a) & set(b))
    pairs = [(a[s][metric], b[s][metric], s) for s in seeds
             if a[s][metric] is not None and b[s][metric] is not None]
    if len(pairs) < 2:
        return None
    st = paired_diff_stats([p[0] for p in pairs], [p[1] for p in pairs])
    st["seeds"] = [p[2] for p in pairs]
    st["per_seed_diffs"] = [p[0] - p[1] for p in pairs]
    return st


def scratch_vs_library(grouped) -> dict:
    out = {}
    for exp, impls in grouped.items():
        if "scratch" in impls and "library" in impls:
            out[exp] = {mt: paired(impls["scratch"], impls["library"], mt) for mt in PAIRED_METRICS}
    return out


def vs_baseline(grouped, baseline_name: str = "baseline") -> dict:
    out = {}
    base = grouped.get(baseline_name, {})
    for exp, impls in grouped.items():
        if exp == baseline_name or not EXPERIMENTS[exp].lora:
            continue
        for impl, seeds in impls.items():
            if impl in base:
                out.setdefault(exp, {})[impl] = {mt: paired(seeds, base[impl], mt) for mt in PAIRED_METRICS}
    return out


# ──────────────────────────── markdown ────────────────────────────

def ms(st: dict, nd=2) -> str:
    return f"{st['mean']:.{nd}f} ± {st['std']:.{nd}f}" if st["n"] > 1 else f"{st['mean']:.{nd}f}"


def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def pair_cell(p: Optional[dict], nd=2) -> str:
    if p is None:
        return "-"
    return (f"{p['mean']:+.{nd}f} ± {p['std']:.{nd}f} "
            f"(CI95 ±{p['ci95_half_width']:.{nd}f}, n={p['n']})")


def impl_table(grouped, stats, exps: List[str]) -> str:
    """Every metric is shown the same way: mean ± std over seeds (n in its own
    column). Medians, min and max for every metric are in stats.json."""
    rows = []
    for exp in exps:
        for impl in ("library", "scratch"):
            if exp in grouped and impl in grouped[exp]:
                s = stats[exp][impl]
                rows.append([exp, f"{EXPERIMENTS[exp].lr:g}", impl, str(s["test_acc"]["n"]),
                             ms(s["test_acc"]), ms(s["best_epoch"], 1), ms(s["stopping_epoch"], 1),
                             ms(s["avg_epoch_train_time_sec"], 1),
                             ms(s["fwd_ms"], 2), ms(s["bwd_ms"], 2), ms(s["opt_ms"], 2),
                             ms(s["peak_alloc_gib"], 3), ms(s["peak_minus_static_gib"], 3)])
    return md_table(["experiment", "lr", "impl", "n", "test acc %", "best ep", "stop ep",
                     "epoch train s", "fwd ms/batch", "bwd ms/batch", "opt ms/batch",
                     "peak alloc GiB", "peak − static GiB"], rows)


def paired_section(svl, exps) -> str:
    rows = []
    for exp in exps:
        if exp not in svl:
            continue
        p = svl[exp]
        rows.append([exp, pair_cell(p["test_acc"]), pair_cell(p["stopping_epoch"], 1),
                     pair_cell(p["avg_epoch_train_time_sec"], 1), pair_cell(p["fwd_ms"]),
                     pair_cell(p["bwd_ms"]), pair_cell(p["peak_alloc_gib"], 3)])
    return md_table(["experiment", "Δ test acc (pp)", "Δ stop epoch", "Δ epoch train (s)",
                     "Δ fwd (ms)", "Δ bwd (ms)", "Δ peak alloc (GiB)"], rows)


def per_seed_acc_section(grouped, exps) -> str:
    rows = []
    for exp in exps:
        if exp in grouped and {"scratch", "library"} <= set(grouped[exp]):
            sc, li = grouped[exp]["scratch"], grouped[exp]["library"]
            for s in sorted(set(sc) & set(li)):
                rows.append([exp, str(s), f"{li[s]['test_acc']:.2f}", f"{sc[s]['test_acc']:.2f}",
                             f"{sc[s]['test_acc'] - li[s]['test_acc']:+.2f}"])
    return md_table(["experiment", "seed", "library acc %", "scratch acc %", "scratch − library (pp)"], rows)


def vs_base_section(vsb) -> str:
    rows = []
    for exp, impls in vsb.items():
        for impl, p in impls.items():
            rows.append([exp, impl, pair_cell(p["test_acc"]), pair_cell(p["peak_alloc_gib"], 2),
                         pair_cell(p["avg_epoch_train_time_sec"], 1), pair_cell(p["fwd_ms"]),
                         pair_cell(p["bwd_ms"]), pair_cell(p["opt_ms"], 2)])
    return md_table(["experiment", "impl", "Δ test acc (pp)", "Δ peak alloc (GiB)", "Δ epoch train (s)",
                     "Δ fwd (ms)", "Δ bwd (ms)", "Δ opt (ms)"], rows)


def artifact_section(grouped) -> str:
    rows = []
    for exp, impls in grouped.items():
        any_row = next(iter(next(iter(impls.values())).values()))
        meas = [r["adapter_export_mib_measured"] for i in impls.values() for r in i.values()
                if r["adapter_export_mib_measured"] is not None]
        rows.append([exp, f"{any_row['trainable_elems']:,}", f"{any_row['artifact_mib_analytic']:.2f}",
                     f"{sum(meas) / len(meas):.2f}" if meas else "-"])
    return md_table(["experiment", "trainable elements", "artifact MiB (analytic, fp32)",
                     "measured adapter export MiB"], rows)


_ARCH_LABEL = {"vitb": "ViT-B", "vitl": "ViT-L", "deitb": "DeiT-B", "swinb": "Swin-B"}


def _in_arch(exp_name: Optional[str], arch: str) -> bool:
    """True if the registry entry `exp_name` belongs to architecture `arch`
    (unknown names are excluded, as the sweep loader does)."""
    return exp_name in EXPERIMENTS and EXPERIMENTS[exp_name].arch == arch


def profile_checks(runs: List[dict], profile_dir: Path, arch: str = "vitb"):
    """Acceptance checks that cross-validate profile_memory.py against the
    independent in-loop measurements in the sweep JSONs. A profiler table is
    only trustworthy if these pass. Returns (rows, failures)."""
    rows, failures = [], []
    if not profile_dir.exists():
        return rows, failures

    sweep: Dict[tuple, dict] = {}
    for r in sorted(runs, key=lambda x: (x["seed"] != 42, x["seed"])):
        sweep.setdefault((r["experiment"], r["implementation"]), r["metrics"])

    def check(name, tag, ok, detail, hard=True):
        status = "PASS" if ok else ("FAIL" if hard else "WARN")
        rows.append([tag, name, status, detail])
        if not ok and hard:
            failures.append(f"profiler check failed: {tag}: {name}: {detail}")

    batch_gib = 64 * 3 * 224 * 224 * 4 / GiB   # the batch-64 input tensor
    train, infer = {}, {}
    for f in sorted(profile_dir.glob("*/*_train.json")):
        r = json.loads(f.read_text())
        if not _in_arch(r.get("experiment"), arch):
            continue
        pt = next((p for p in r["train"]["sweep"] if p.get("batch_size") == 64 and not p.get("oom")), None)
        if pt:
            train[(r["experiment"], r["implementation"])] = pt
    for f in sorted(profile_dir.glob("*/*_inference.json")):
        r = json.loads(f.read_text())
        if not _in_arch(r.get("experiment"), arch):
            continue
        infer[(r["experiment"], r["implementation"])] = r["inference"]

    for (exp, impl), pt in train.items():
        tag, m = f"{exp}/{impl}", sweep.get((exp, impl))
        if m is None:
            continue
        b = m["static_memory_breakdown_gb"]
        check("grads (profiler) = grads (static breakdown)", tag,
              abs(pt["grads_gb"] - b["grads"]) <= max(0.25 / 1024, 0.01 * b["grads"]),
              f"{pt['grads_gb'] * 1024:.2f} vs {b['grads'] * 1024:.2f} MiB")
        expect = b["params"] + b["adamw_states"] + batch_gib
        check("analytic resident (live tensors) = params + AdamW state + input batch", tag,
              abs(pt["resident_analytic_gb"] - expect) <= 0.005,
              f"{pt['resident_analytic_gb']:.4f} vs {expect:.4f} GiB")
        check("runtime overhead (allocator/cuBLAS workspace) is small", tag,
              -0.0005 <= pt["runtime_overhead_gb"] <= 0.04, f"{pt['runtime_overhead_gb'] * 1024:.1f} MiB")
        dev = abs(pt["peak_gb"] - m["peak_memory_gb_train"]) / m["peak_memory_gb_train"]
        check("batch-64 peak (profiler) vs in-loop training peak", tag, dev <= 0.03,
              f"{pt['peak_gb']:.3f} vs {m['peak_memory_gb_train']:.3f} GiB ({100 * dev:.1f}% apart)", hard=dev > 0.05)
    for exp in sorted({e for e, _ in train}):
        if (exp, "library") in train and (exp, "scratch") in train:
            a, c = train[(exp, "library")]["activation_gb"], train[(exp, "scratch")]["activation_gb"]
            check("library vs scratch activation within 1%", exp, abs(a - c) <= 0.01 * max(a, c),
                  f"{a:.3f} vs {c:.3f} GiB")

    base_name = ARCH_BASELINE[arch]
    for (exp, impl), inf in infer.items():
        tag = f"{exp}/{impl}"
        base_params = dense_param_counts(arch_dims(EXPERIMENTS[exp].model_local_path, strict=(arch != "vitb")),
                                         EXPERIMENTS[exp].num_labels)[0]
        if exp == base_name and sweep.get((exp, impl)) is not None and "fp32_unmerged" in inf:
            probe = sweep[(exp, impl)].get("inference_resident_overhead_gb")
            if probe is not None:
                mine = inf["fp32_unmerged"]["runtime_overhead_gb"]
                check("profiler runtime overhead agrees with the in-loop probe's (same GPU runtime state)", tag,
                      abs(mine - probe) <= 0.005, f"{mine * 1024:.1f} vs {probe * 1024:.1f} MiB", hard=False)
        for var, e in inf.items():
            check(f"{var}: runtime overhead (resident - parameter bytes) is small", tag,
                  -0.0005 <= e["runtime_overhead_gb"] <= 0.04, f"{e['runtime_overhead_gb'] * 1024:.1f} MiB")
            if var == "fp32_unmerged":
                st_m = sweep.get((exp, impl))
                if st_m is not None:
                    check("parameter bytes = static params of the sweep run", tag,
                          abs(e["param_buffer_bytes_gb"] - st_m["static_memory_breakdown_gb"]["params"]) <= 0.0005,
                          f"{e['param_buffer_bytes_gb']:.4f} vs {st_m['static_memory_breakdown_gb']['params']:.4f} GiB")
            b1 = e["by_batch"].get("1")
            if b1:
                check(f"{var}: batch-1 peak >= weights", tag,
                      b1["peak_allocated_gb"] + 1e-6 >= e["resident_weights_gb"],
                      f"{b1['peak_allocated_gb']:.4f} vs {e['resident_weights_gb']:.4f} GiB")
        merged = inf.get("fp32_merged")
        if merged:
            check(f"merged model has the dense {_ARCH_LABEL[arch]} parameter count", tag,
                  merged["num_params"] == base_params, f"{merged['num_params']:,} vs {base_params:,}")
            ref = infer.get((base_name, impl), {}).get("fp32_unmerged")
            if ref:
                check("merged resident (incl. runtime overhead) = dense baseline", tag,
                      abs(merged["resident_weights_gb"] - ref["resident_weights_gb"]) <= 0.0005,
                      f"{merged['resident_weights_gb']:.4f} vs {ref['resident_weights_gb']:.4f} GiB")
    return rows, failures


def profile_check_section(rows) -> str:
    if not rows:
        return ""
    n_fail = sum(1 for r in rows if r[2] == "FAIL")
    n_warn = sum(1 for r in rows if r[2] == "WARN")
    out = (f"### Profiler acceptance checks ({len(rows)} checks: {len(rows) - n_fail - n_warn} pass, "
           f"{n_warn} warn, {n_fail} fail)\n\n"
           "The profiler tables below count as evidence only if these hold. They compare profile_memory.py "
           "against the independent in-loop measurements and against analytic byte counts.\n\n")
    shown = [r for r in rows if r[2] != "PASS"] or rows[:6]
    out += md_table(["scope", "check", "status", "detail"], shown)
    if n_fail == 0 and n_warn == 0:
        out += "\n(Only the first rows are shown because every check passed; all are in stats.json.)\n"
    return out + "\n"


def profile_sections(profile_dir: Path, arch: str = "vitb") -> str:
    files = sorted(profile_dir.glob("*/*_inference.json")) if profile_dir.exists() else []
    tr = sorted(profile_dir.glob("*/*_train.json")) if profile_dir.exists() else []
    files = [f for f in files if _in_arch(f.parent.name, arch)]
    tr = [f for f in tr if _in_arch(f.parent.name, arch)]
    out = ""
    if files:
        rows = []
        for f in files:
            r = json.loads(f.read_text())
            for var, e in r["inference"].items():
                b1, b64 = e["by_batch"].get("1"), e["by_batch"].get("64")
                rows.append([r["experiment"], r["implementation"], var, f"{e['num_params']:,}",
                             f"{e['resident_weights_gb']:.4f}",
                             f"{b1['peak_allocated_gb']:.4f}" if b1 else "-",
                             f"{b1['latency']['p50_ms']:.2f}" if b1 else "-",
                             f"{b64['peak_allocated_gb']:.3f}" if b64 else "-",
                             f"{b64['latency']['p50_ms']:.1f}" if b64 else "-"])
        out += ("### Inference memory and latency (fresh process, no training; measured by profile_memory.py)\n\n"
                + md_table(["experiment", "impl", "variant", "params", "resident weights GiB",
                            "b1 peak alloc GiB", "b1 p50 ms", "b64 peak alloc GiB", "b64 p50 ms"], rows) + "\n")
    if tr:
        rows = []
        for f in tr:
            r = json.loads(f.read_text())
            for p in r["train"]["sweep"]:
                if p.get("batch_size") == 64 and not p.get("oom"):
                    rows.append([r["experiment"], r["implementation"], f"{p['resident_before_fwd_gb']:.3f}",
                                 f"{p['activation_gb']:.3f}", f"{p['grads_gb']:.4f}",
                                 f"{p['opt_transient_gb']:.3f}", f"{p['peak_gb']:.3f}"])
        out += ("### Training-step decomposition at batch 64 (measured by profile_memory.py)\n\n"
                + md_table(["experiment", "impl", "resident before fwd GiB", "activation GiB", "grads GiB",
                            "optimizer transient GiB", "peak GiB"], rows) + "\n")
    return out


def merge_section(merge_dirs: List[Path], arch: str = "vitb") -> str:
    files = sorted(f for d in merge_dirs if d.exists() for f in d.glob("*/seed*.json")
                   if _in_arch(f.parent.name, arch))
    if not files:
        return ""
    rows, drows, crows, arows = [], [], [], []
    for f in files:
        r = json.loads(f.read_text())
        src = "trained" if r["adapter_source"].startswith("trained") else "perturbed B"
        c = r["correctness_fp32"]
        for impl in ("library", "scratch"):
            if impl not in c:                       # library-only family (Swin): no scratch model was merged
                continue
            vs = c.get("library_vs_scratch_merged")
            rows.append([r["experiment"], src, impl, f"{c[impl]['max_abs_diff_merged_vs_unmerged']:.2e}",
                         f"{c[impl]['top1_agreement']:.3f}",
                         f"{vs['max_abs_diff']:.2e}" if vs else "-"])
            for dt, dd in r["drift"].items():
                last = list(dd[impl].values())[-1]
                drows.append([r["experiment"], impl, dt, str(r["cycles"]),
                              f"{last['max_abs_weight_drift']:.2e}", f"{last['max_abs_logit_change']:.2e}"])
            k = r["cost_fp32"][impl]
            crows.append([r["experiment"], impl, f"{k['merge_all_ms_median']:.1f}", f"{k['unmerge_all_ms_median']:.1f}",
                          f"{k['merge_transient_mib_max']:.1f}" if k["merge_transient_mib_max"] is not None else "-"])
            if "test_accuracy" in r:
                a = r["test_accuracy"][impl]
                arows.append([r["experiment"], impl, f"{a['acc_unmerged']:.2f}", f"{a['acc_merged']:.2f}",
                              f"{a[f'acc_after_{r['cycles']}_cycles']:.2f}",
                              f"{a['pred_agreement_merged_vs_unmerged']:.4f}"])
        rf, sw = r["cost_fp32"]["reload_full_state_dict"], r["cost_fp32"]["swap_adapter_from_file"]
        crows.append([r["experiment"], "reload full state dict", f"{rf['load_ms_median']:.1f}", "-",
                      f"{rf['size_mib']:.0f} MiB file"])
        crows.append([r["experiment"], "load+apply adapter file", f"{sw['load_apply_ms_median']:.1f}", "-",
                      f"{sw['size_mib']:.2f} MiB file"])
    out = "### Merge / unmerge (measured by merge_study.py)\n\n"
    out += "Correctness, fp32 (merged vs unmerged logits):\n\n" + md_table(
        ["experiment", "adapter", "impl", "max abs Δlogit", "top-1 agreement", "library vs scratch (merged) max abs Δ"], rows)
    out += "\nRound-trip drift (merge → unmerge):\n\n" + md_table(
        ["experiment", "impl", "dtype", "cycles", "max abs ΔW vs pristine", "max abs Δlogit vs original"], drows)
    out += "\nCost:\n\n" + md_table(["experiment", "impl / item", "merge-all ms", "unmerge-all ms", "transient / size"], crows)
    if arows:
        out += "\nTest accuracy (trained adapter):\n\n" + md_table(
            ["experiment", "impl", "unmerged %", "merged %", "after cycles %", "pred agreement merged/unmerged"], arows)
    return out + "\n"


def build_report(grouped, stats, svl, vsb, problems, skipped, missing, profile_dir, merge_dirs, check_rows=(),
                 arch: str = "vitb") -> str:
    sweep_exps = [e for e, c in EXPERIMENTS.items() if c.arch == arch and not c.lora]
    lora_exps = [e for e, c in EXPERIMENTS.items() if c.arch == arch and c.lora]
    out = ["# Aggregated results\n" if arch == "vitb" else f"# Aggregated results — {arch}\n",
           "Source: per-run result JSONs (single source of truth). Memory in GiB (1024³ B), sizes in MiB. "
           "Peak memory = peak **allocated** by PyTorch during train+val, fp32, batch 64 (excludes CUDA context). "
           "`peak − static` = peak allocated minus (params + grads + AdamW state); it is not pure activation memory "
           "(for full FT it understates activations by about the gradient size). "
           "Per-batch timings use epochs ≥ 2. Every metric column is mean ± std over seeds "
           "(sample std, n−1); medians/min/max are in stats.json.\n"]
    if problems:
        out.append("## ⚠ Assertion failures\n\n" + "\n".join(f"- {p}" for p in problems) + "\n")
    if missing:
        out.append("## Missing runs (registry seeds not found)\n\n" + "\n".join(f"- {m}" for m in missing) + "\n")
    out.append("## Table 1 — Full fine-tuning, learning-rate sweep\n\n" + impl_table(grouped, stats, sweep_exps))
    out.append("## Table 2 — LoRA configurations (lr 1e-4)\n\n" + impl_table(grouped, stats, lora_exps))
    out.append("## Scratch − library, paired per seed\n\n"
               "Each cell: mean difference ± std of the per-seed differences (95% t-CI half-width, n seeds). "
               "A CI that contains 0 means **no detectable difference at this sample size**; it is not evidence of "
               "equivalence. With n=3 the t critical value is 4.30, so intervals are wide.\n\n"
               + paired_section(svl, sweep_exps + lora_exps))
    out.append("### Per-seed accuracy differences\n\n" + per_seed_acc_section(grouped, sweep_exps + lora_exps))
    if vsb:
        out.append("## LoRA − full fine-tuning (lr 1e-4 baseline), paired per seed, same implementation\n\n"
                   + vs_base_section(vsb))
    out.append("## Trainable parameters and per-task artifact size\n\n"
               "Artifact = head + adapters (LoRA) or the full weights (baseline), fp32, the same for both "
               "implementations.\n\n" + artifact_section(grouped))
    out.append(profile_check_section(list(check_rows)))
    out.append(profile_sections(profile_dir, arch))
    out.append(merge_section(merge_dirs, arch))
    out.append("## Excluded from this report (known invalid in the sweep JSONs)\n\n"
               "- `inference_*` memory fields (probe measured with training state resident) — replaced by the profiler tables above.\n"
               "- `batch1_forward_sec_*` — rough, unmerged adapters, 20 iterations.\n"
               "- `best_checkpoint_size_mb` — a full state dict for scratch LoRA; artifact size above is analytic/exported.\n")
    if skipped:
        out.append("\nSkipped files:\n\n" + "\n".join(f"- {s}" for s in skipped) + "\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--profile", default="results_profile")
    ap.add_argument("--merge", nargs="+", default=["results_merge", "results_merge_trained"])
    ap.add_argument("--out", default="report_out")
    ap.add_argument("--arch", default="vitb", choices=sorted(ARCH_BASELINE),
                    help="Which architecture's experiments to aggregate (each gets its own tables).")
    ap.add_argument("--require-all", action="store_true",
                    help="Also list registry experiments of this architecture that have no runs at all "
                         "(by default only partially completed experiments are reported as missing).")
    args = ap.parse_args()

    runs, skipped, problems = load_runs(Path(args.results))
    runs = [r for r in runs if EXPERIMENTS[r["experiment"]].arch == args.arch]
    rows = [derive(r, problems) for r in runs]
    grouped = group(rows)
    stats = per_impl_stats(grouped)
    svl, vsb = scratch_vs_library(grouped), vs_baseline(grouped, ARCH_BASELINE[args.arch])

    missing = []
    for exp, impls in grouped.items():
        for impl in EXPERIMENTS[exp].implementations:
            for seed in EXPERIMENTS[exp].seeds:
                if seed not in impls.get(impl, {}):
                    missing.append(f"{exp}/{impl}/seed{seed}")
    for exp, c in EXPERIMENTS.items():
        if c.arch == args.arch and exp not in grouped and (
                args.require_all or (args.arch == "vitb" and exp in ORIGINAL_VITB)):
            missing.append(f"{exp}: no runs found")

    # Pooling check: runs produced by different code must not be mixed silently.
    hashes = sorted({r["source_sha256"] for r in rows if r["source_sha256"]})
    no_prov = sum(1 for r in rows if not r["source_sha256"])
    if len(hashes) > 1:
        problems.append(f"runs come from {len(hashes)} different source versions {hashes}; "
                        f"do not pool them without checking what changed")
    if no_prov:
        skipped.append(f"{no_prov} of {len(rows)} aggregated runs have no provenance block "
                       f"(produced before schema v2); their code version is unrecorded")

    check_rows, check_failures = profile_checks(runs, Path(args.profile), args.arch)
    problems.extend(check_failures)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(build_report(grouped, stats, svl, vsb, problems, skipped, missing,
                                                Path(args.profile), [Path(d) for d in args.merge], check_rows, args.arch))
    if rows:
        with open(out / "per_run.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out / "stats.json").write_text(json.dumps(
        {"per_impl": stats, "scratch_minus_library": svl, "config_minus_baseline": vsb,
         "assertion_failures": problems, "missing": missing, "profiler_checks": check_rows}, indent=2))
    if not rows:
        print(f"WARNING: no runs found for --arch {args.arch} under {args.results}; the report is empty")
    print(f"{len(rows)} runs aggregated; {len(problems)} assertion failures; {len(missing)} missing cells")
    print(f"wrote {out}/report.md, per_run.csv, stats.json")


if __name__ == "__main__":
    main()
