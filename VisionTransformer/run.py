"""
Single CLI entry point for running ViT fine-tuning and LoRA experiments.

Examples:
    python run.py --experiment baseline                     # run all 5 default seeds paired
    python run.py --experiment lora_attn_r8 --seeds 42      # single seed test
    python run.py --experiment baseline lora_attn_r8        # multiple experiments
    python run.py --experiment all                          # run all registered experiments
    python run.py --impl scratch --seeds 42                 # run scratch only
    python run.py --list                                    # list all experiments in registry
    python run.py --experiment lora_attn_r8 --seeds 42 --verify-only   # prove init matching, no training
"""

import argparse
import os
import sys
import math
import copy
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from experiments import EXPERIMENTS, ExperimentConfig


def calculate_stats(values: List[float]) -> Dict[str, Any]:
    """Returns {n, mean, std, median, min, max} for a list of values.

    std is the sample standard deviation (n-1). Medians matter for timing
    metrics, whose means are dragged by occasional scheduler/GPU spikes.
    """
    stats: Dict[str, Any] = {"n": len(values), "mean": 0.0, "std": 0.0,
                             "median": 0.0, "min": 0.0, "max": 0.0}
    if not values:
        return stats
    ordered = sorted(values)
    stats["min"] = ordered[0]
    stats["max"] = ordered[-1]
    stats["mean"] = sum(values) / len(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        stats["median"] = ordered[mid]
    else:
        stats["median"] = (ordered[mid - 1] + ordered[mid]) / 2
    if len(values) > 1:
        variance = sum((x - stats["mean"]) ** 2 for x in values) / (len(values) - 1)
        stats["std"] = math.sqrt(variance)
    return stats


# Two-sided 95% t critical values for df = 1..30 (for paired-difference CIs with
# few seeds; avoids a scipy dependency). Beyond df=30 the normal approximation
# (1.96) is used, and that fact is recorded in the stats block.
_T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
              13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
              19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
              25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
_T_CRIT_95_LARGE_DF = 1.96  # normal approximation, used above df=30


def paired_diff_stats(values_a: List[float], values_b: List[float]) -> Optional[Dict[str, Any]]:
    """Per-seed paired difference (a - b) with a 95% confidence interval.

    For a given seed the two implementations share the data split and paired
    initialization, so the difference removes the between-seed variance that
    makes marginal mean ± std uninterpretable. Requires the lists to be aligned
    per seed and at least 2 pairs. Returns None otherwise.
    """
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    diffs = [a - b for a, b in zip(values_a, values_b)]
    stats = calculate_stats(diffs)
    df = stats["n"] - 1
    t_crit = _T_CRIT_95.get(df)
    if t_crit is None:
        # Never silently omit the headline statistic: fall back to the normal
        # approximation and record that the CI is approximate.
        t_crit = _T_CRIT_95_LARGE_DF
        stats["ci_method"] = "normal-approximation (df>30)"
    stats["ci95_half_width"] = t_crit * stats["std"] / math.sqrt(stats["n"])
    return stats


def resolve_implementations(base_config, impl_arg: str, explicit: bool = False) -> List[str]:
    """Intersect the CLI's implementation request with what the config supports.

    'paired' means "both", but an experiment may legitimately support only one
    implementation. The config always wins: forcing both would silently build a
    scratch model with no library reference (unpaired init presented as paired).

    Downgrade policy: when the user EXPLICITLY asked for 'paired' and the config
    cannot deliver both, that is a silent loss of the comparison the run exists
    to make — so it is an error. When 'paired' is merely the default, a note is
    printed and the run proceeds with what the config supports.
    """
    requested = ["library", "scratch"] if impl_arg == "paired" else [impl_arg]
    active = [impl for impl in requested if impl in base_config.implementations]
    if not active:
        sys.exit(
            f"Error: experiment '{base_config.name}' supports {base_config.implementations}, "
            f"but --impl {impl_arg} requests {requested}."
        )
    if len(active) < len(requested):
        message = (f"Experiment '{base_config.name}' supports only {active}, "
                   f"but {impl_arg} was requested ({requested}).")
        if explicit and impl_arg == "paired":
            sys.exit(f"Error: {message} Refusing to silently run a partial comparison — "
                     f"pass --impl {' or --impl '.join(active)} to proceed with one implementation.")
        print(f"[runner] {message} Proceeding with what the config supports.")
    return active


def run_pre_train_checks(models: Dict[str, Any], seed: int, device, forward_tol: float, config=None) -> dict:
    """All pre-training verification in one place:

    1. summarize_init                    — transferred tensors (head + LoRA A/B) identical
    2. trainable_signature               — both implementations optimize the same parameters
    3. check_forward_equivalence         — both implementations compute the same function
    4. check_forward_equivalence_with_lora — same, with every adapter ACTIVE
                                          (LoRA B inits to zeros, so the plain
                                          forward check is blind to the adapter path)

    Returns the combined report dict. Moves both models to `device` for the
    checks, then moves them BACK and releases the cache: whichever
    implementation trains first must not carry the other one's weights in its
    peak-memory figure (reset_peak_memory_stats resets to the CURRENT
    allocation, so leftover weights would inflate it). The release is complete:
    `.cpu()` moves every parameter off the GPU (freeing those allocations), and
    `empty_cache()` returns the freed blocks to the driver — so by the time the
    training loop's own reset runs, only the model being trained is resident.
    Empirically confirmed: paired runs report identical peak memory for both
    implementations (e.g. 5.459 vs 5.459 GB in the seed-42 smoke test).

    A family without a scratch implementation (Swin) has nothing to pair against; given its
    `config`, the single library model gets models.library_only_checks instead, reported
    under "library_only".
    """
    # Imported here (not at module level) so `--list` does not pay the
    # torch/transformers/peft import cost. These names MUST NOT live in
    # main()'s import block: function-local imports there are invisible to
    # this module-level function.
    from models import (summarize_init, trainable_signature, check_forward_equivalence,
                        check_forward_equivalence_with_lora, has_lora_params, library_only_checks)
    from experiments import has_scratch_impl

    report = summarize_init(models)
    if "library" not in models or "scratch" not in models:
        if config is None and "library" in models and getattr(getattr(models["library"], "config", None),
                                                              "depths", None):
            # A hierarchical (Swin) model has no scratch twin, so without the config it would train
            # with no gate at all. Refuse instead of returning a bare note.
            raise RuntimeError("run_pre_train_checks needs config= for a library-only (Swin) model; "
                               "without it no pre-training check would run")
        if config is not None and "library" in models and not has_scratch_impl(config.model_local_path):
            models["library"] = models["library"].to(device)
            report["library_only"] = library_only_checks(models["library"], config, seed, device, forward_tol)
            models["library"] = models["library"].cpu()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return report

    sig_lib = trainable_signature(models["library"])
    sig_scr = trainable_signature(models["scratch"])
    report["param_check"] = {
        "library": sig_lib,
        "scratch": sig_scr,
        "ok": sig_lib == sig_scr,
    }

    models["library"] = models["library"].to(device)
    models["scratch"] = models["scratch"].to(device)
    report["forward"] = check_forward_equivalence(
        models["library"], models["scratch"], seed=seed, device=device, tol=forward_tol
    )
    if has_lora_params(models["library"]) and has_lora_params(models["scratch"]):
        report["forward_lora"] = check_forward_equivalence_with_lora(
            models["library"], models["scratch"], seed=seed, device=device, tol=forward_tol
        )

    models["library"] = models["library"].cpu()
    models["scratch"] = models["scratch"].cpu()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def enforce_checks(report: dict, label: str) -> None:
    """Print the verification report; exit(1) if any check failed."""
    lo = report.get("library_only")
    if lo is not None:
        if not lo.get("ok"):
            print(f"[{label}] FAILED — refusing to train. Library-only checks failed:")
            print(json.dumps(lo, indent=2))
            sys.exit(1)
        line = (f"[{label}] OK (library-only family, no paired check): trainable "
                f"{lo['trainable']['elements']:,} elements / {lo['trainable']['tensors']} tensors as expected")
        if "merge" in lo:
            line += (f"; LoRA B=0 identity {lo['identity_max_abs_diff']:.1e}; merged-vs-unmerged "
                     f"max|Δlogits| {lo['merge']['merged_vs_unmerged_max_abs_diff']:.2e} "
                     f"({lo['merge']['adapters_perturbed']} adapters)")
        print(line)
        return
    if "note" in report:
        print(f"[{label}] {report['note']}")
        return

    failures = []
    if not report.get("ok"):
        failures.append("initialization tensors are not identical")
    if not report.get("param_check", {}).get("ok"):
        failures.append("trainable parameter sets differ between implementations")
    fwd = report.get("forward")
    if fwd is not None and not fwd.get("forward_ok"):
        failures.append("forward outputs differ beyond tolerance")
    fwd_lora = report.get("forward_lora")
    if fwd_lora is not None and "note" not in fwd_lora and not fwd_lora.get("forward_ok"):
        failures.append("forward outputs differ beyond tolerance with LoRA adapters active")

    if failures:
        print(f"[{label}] FAILED — refusing to train. Reasons: " + "; ".join(failures))
        print(json.dumps(report, indent=2))
        sys.exit(1)

    line = (f"[{label}] OK: {report['equal']}/{report['compared']} init tensors identical "
            f"(max|diff| = {report['max_abs_diff']:.2e}); "
            f"{report['param_check']['library']['trainable_count']} trainable tensors on both sides "
            f"(distinct from the element count printed as 'Trainable Parameters' below)")
    if fwd is not None:
        line += f"; forward max|Δlogits| = {fwd['forward_max_abs_diff']:.2e} (tol {fwd['tolerance']:.0e})"
    if fwd_lora is not None and "note" not in fwd_lora:
        line += (f"; LoRA-path max|Δlogits| = {fwd_lora['forward_max_abs_diff']:.2e} "
                 f"({fwd_lora['lora_adapters_exercised']} adapters)")
    print(line)


def _paired_metric_diff(impls: Dict[str, List[Dict[str, Any]]], metric: str) -> Optional[Dict[str, Any]]:
    """Paired per-seed difference (scratch − library) for one metrics key.

    Only seeds present in BOTH implementations are paired. Dropping a seed
    silently would shrink the evidence base invisibly, so any dropped seed is
    reported loudly.
    """
    lib = {r["seed"]: r["metrics"][metric] for r in impls.get("library", [])}
    scr = {r["seed"]: r["metrics"][metric] for r in impls.get("scratch", [])}
    common = sorted(set(lib) & set(scr))
    dropped_lib = sorted(set(lib) - set(scr))
    dropped_scr = sorted(set(scr) - set(lib))
    if dropped_lib or dropped_scr:
        print(f"[paired] WARNING: {metric} — seeds dropped for lack of a pair: "
              f"library-only {dropped_lib}, scratch-only {dropped_scr}. "
              f"Paired stats use {len(common)} of "
              f"{max(len(lib), len(scr))} seeds.")
    if len(common) < 2:
        return None
    pair = paired_diff_stats([scr[s] for s in common], [lib[s] for s in common])
    if pair is not None:
        pair["seeds"] = common
    return pair


def cross_config_pairs(summary_data: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
    """Paired per-seed difference of each experiment vs the baseline experiment.

    Requires the same seed for both runs. With the seed-driven data policy, the
    same seed yields the same split, shuffle order and head init, so the
    difference isolates the configuration change (e.g. LoRA vs full FT) — the
    comparison supervisors actually ask about. Marginal mean ± std across
    different configs cannot show this.
    """
    if "baseline" not in summary_data:
        return {}
    base_impls = summary_data["baseline"]
    pairs: Dict[str, Any] = {}
    for exp_name, impls in summary_data.items():
        if exp_name == "baseline":
            continue
        for impl_name, runs in impls.items():
            base_runs = {r["seed"]: r["metrics"]["test_acc"] for r in base_impls.get(impl_name, [])}
            runs_by_seed = {r["seed"]: r["metrics"]["test_acc"] for r in runs}
            common = sorted(set(base_runs) & set(runs_by_seed))
            if len(common) < 2:
                # Silent only when there is genuinely nothing to compare (the
                # implementation never ran for baseline). When both sides ran
                # but share too few seeds, say so — a silently missing
                # comparison is the failure mode this pipeline must not have.
                if base_runs and runs_by_seed:
                    print(f"[vs-baseline] WARNING: {exp_name}/{impl_name} shares only "
                          f"{len(common)} seed(s) with baseline ({common}); "
                          f"comparison skipped.")
                continue
            if len(common) < max(len(base_runs), len(runs_by_seed)):
                dropped = sorted((set(base_runs) | set(runs_by_seed)) - set(common))
                print(f"[vs-baseline] WARNING: {exp_name}/{impl_name} — seeds dropped for "
                      f"lack of a baseline pair: {dropped}. Comparison uses {len(common)} of "
                      f"{max(len(base_runs), len(runs_by_seed))} seeds.")
            pair = paired_diff_stats([runs_by_seed[s] for s in common], [base_runs[s] for s in common])
            if pair is not None:
                pair["seeds"] = common
                pairs.setdefault(exp_name, {})[impl_name] = pair
    return pairs


_EQUIVALENCE_NOTE = (
    "A paired CI that excludes 0 indicates a systematic difference between "
    "implementations. Absence of a significant difference is NOT evidence of "
    "equivalence: to claim equivalence, declare a margin up front (e.g. ±0.3pp) "
    "and require the paired CI to lie entirely inside it."
)


def print_summary_table(summary_data: Dict[str, Dict[str, List[Dict[str, Any]]]]):
    """Prints a structured comparison table: per implementation, mean ± std and
    median across seeds; the per-seed paired difference (scratch − library)
    wherever both implementations ran on the same seeds; and each experiment
    paired against baseline."""
    print("\n" + "=" * 100)
    print("EXPERIMENT EXECUTION SUMMARY (Mean ± Std across seeds)")
    print("=" * 100)
    print(f"{'Experiment':<22} | {'Impl':<8} | {'Seeds':<6} | {'Test Acc (%)':<24} | "
          f"{'Epoch Time (s)':<18} | {'Fwd med (ms)':<12} | {'Bwd med (ms)':<12} | {'Opt med (ms)':<11} | "
          f"{'Peak Mem alloc med (GB)':<22} | {'Peak Mem res med (GB)':<21}")
    print("-" * 168)

    approx_used = False  # any paired CI computed with the normal approximation

    for exp_name, impls in summary_data.items():
        for impl_name, runs in impls.items():
            if not runs:
                continue
            acc = calculate_stats([r["metrics"]["test_acc"] for r in runs])
            ep = calculate_stats([r["metrics"]["avg_epoch_train_time_sec"] for r in runs])
            fw = calculate_stats([r["metrics"]["avg_batch_forward_sec"] * 1000 for r in runs])
            bw = calculate_stats([r["metrics"]["avg_batch_backward_sec"] * 1000 for r in runs])
            op = calculate_stats([r["metrics"]["avg_batch_optim_sec"] * 1000 for r in runs])
            mem = calculate_stats([r["metrics"]["peak_memory_gb_train"] for r in runs])
            mem_res = calculate_stats([r["metrics"]["peak_memory_reserved_gb_train"] for r in runs])

            acc_str = f"{acc['mean']:.2f} ± {acc['std']:.2f} (med {acc['median']:.2f})"
            time_str = f"{ep['mean']:.1f} ± {ep['std']:.1f}"

            print(f"{exp_name:<22} | {impl_name:<8} | {acc['n']:<6} | {acc_str:<24} | "
                  f"{time_str:<18} | {fw['median']:<12.2f} | {bw['median']:<12.2f} | {op['median']:<11.2f} | "
                  f"{mem['median']:<22.2f} | {mem_res['median']:<21.2f}")

        # Paired view: same seed → same data split and paired init, so the
        # between-seed variance cancels and only the implementation difference
        # remains. Per-epoch time, not total: the two implementations can
        # early-stop after different epoch counts.
        pair_acc = _paired_metric_diff(impls, "test_acc")
        pair_ep = _paired_metric_diff(impls, "avg_epoch_train_time_sec")
        for pair in (pair_acc, pair_ep):
            if pair is not None and "ci_method" in pair:
                approx_used = True
        if pair_acc is not None or pair_ep is not None:
            n = (pair_acc or pair_ep)["n"]
            acc_part = (f"Δacc {pair_acc['mean']:+.2f} ± {pair_acc['ci95_half_width']:.2f}"
                        if pair_acc else "-")
            time_part = (f"Δep {pair_ep['mean']:+.1f} ± {pair_ep['ci95_half_width']:.1f}s"
                         if pair_ep else "-")
            print(f"{exp_name:<22} | {'PAIRED':<8} | {n:<6} | {acc_part:<24} | "
                  f"{time_part:<18} | {'':<12} | {'':<12} | {'':<11} | {'':<22} | "
                  f"{'scratch − library':<21}")

    # ── Each experiment vs baseline, paired per seed ──
    vs_base = cross_config_pairs(summary_data)
    if vs_base:
        print("-" * 100)
        print("EXPERIMENT vs BASELINE, paired per seed (Δ = this config − baseline, "
              "same implementation)")
        for exp_name, impls in vs_base.items():
            for impl_name, pair in impls.items():
                print(f"{exp_name:<22} | {'vs BASE':<8} | {pair['n']:<6} | "
                      f"Δacc {pair['mean']:+.2f} ± {pair['ci95_half_width']:.2f} (CI95) | "
                      f"seeds {pair['seeds']} | {impl_name}")

    print("=" * 100)
    if approx_used:
        print("Note: at least one paired CI above used the normal approximation "
              "(df>30) rather than a t critical value — see 'ci_method' in summary.json.")
    print(_EQUIVALENCE_NOTE)
    print()


def write_summary_json(summary_data: Dict[str, Dict[str, List[Dict[str, Any]]]], path: Path) -> None:
    """Machine-readable copy of the summary statistics: per implementation and
    per experiment, the paired differences wherever available, and the
    paired-vs-baseline view."""
    payload: Dict[str, Any] = {"interpretation": _EQUIVALENCE_NOTE, "experiments": {}}
    for exp_name, impls in summary_data.items():
        entry: Dict[str, Any] = {}
        for impl_name, runs in impls.items():
            if not runs:
                continue
            entry[impl_name] = {
                "n": len(runs),
                "test_acc": calculate_stats([r["metrics"]["test_acc"] for r in runs]),
                "avg_epoch_train_time_sec": calculate_stats([r["metrics"]["avg_epoch_train_time_sec"] for r in runs]),
                "total_train_time_sec": calculate_stats([r["metrics"]["total_train_time_sec"] for r in runs]),
                "avg_batch_forward_sec": calculate_stats([r["metrics"]["avg_batch_forward_sec"] for r in runs]),
                "avg_batch_backward_sec": calculate_stats([r["metrics"]["avg_batch_backward_sec"] for r in runs]),
                "avg_batch_optim_sec": calculate_stats([r["metrics"]["avg_batch_optim_sec"] for r in runs]),
                "peak_memory_gb_train": calculate_stats([r["metrics"]["peak_memory_gb_train"] for r in runs]),
                "peak_memory_gb_run": calculate_stats([r["metrics"]["peak_memory_gb_run"] for r in runs]),
                "peak_memory_reserved_gb_train": calculate_stats([r["metrics"]["peak_memory_reserved_gb_train"] for r in runs]),
                "peak_memory_reserved_gb_run": calculate_stats([r["metrics"]["peak_memory_reserved_gb_run"] for r in runs]),
                "static_memory_gb": calculate_stats([r["metrics"]["static_memory_gb"] for r in runs]),
                "activation_peak_gb_train": calculate_stats([r["metrics"]["activation_peak_gb_train"] for r in runs]),
                "batch1_forward_sec_median": calculate_stats([r["metrics"]["batch1_forward_sec_median"] for r in runs]),
                "inference_peak_memory_allocated_gb": calculate_stats([r["metrics"]["inference_peak_memory_allocated_gb"] for r in runs]),
                "inference_peak_memory_reserved_gb": calculate_stats([r["metrics"]["inference_peak_memory_reserved_gb"] for r in runs]),
            }
            # Optional fields: present when a checkpoint was produced. A missing
            # key must never abort the summary at the end of a long job.
            if all(r["metrics"].get("best_checkpoint_size_mb") is not None for r in runs):
                entry[impl_name]["best_checkpoint_size_mb"] = calculate_stats(
                    [r["metrics"]["best_checkpoint_size_mb"] for r in runs])
        pair_acc = _paired_metric_diff(impls, "test_acc")
        pair_ep = _paired_metric_diff(impls, "avg_epoch_train_time_sec")
        if pair_acc is not None:
            entry["paired_test_acc_diff_scratch_minus_library"] = pair_acc
        if pair_ep is not None:
            entry["paired_epoch_time_diff_scratch_minus_library"] = pair_ep
        if entry:
            payload["experiments"][exp_name] = entry

    vs_base = cross_config_pairs(summary_data)
    if vs_base:
        payload["experiments_vs_baseline"] = vs_base

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Summary JSON written to: {path}")


def summarize_verification(report: dict) -> dict:
    """Trimmed, JSON-friendly verification summary for the result record.

    The full report (with per-implementation shape multisets) is printed to the
    job log; this is the compact version stored per run so the data files
    themselves prove paired initialization was verified. Without it, that
    evidence exists only in the .out logs — unrecoverable once logs rotate.
    """
    if "library_only" in report:
        lo = report["library_only"]
        out = {"note": report["note"], "library_only_ok": lo["ok"],
               "trainable_elements": lo["trainable"]["elements"], "trainable_tensors": lo["trainable"]["tensors"]}
        if "merge" in lo:
            out.update(identity_max_abs_diff=lo["identity_max_abs_diff"],
                       merged_vs_unmerged_max_abs_diff=lo["merge"]["merged_vs_unmerged_max_abs_diff"],
                       lora_adapters_exercised=lo["merge"]["adapters_perturbed"])
        return out
    if "note" in report:
        return {"note": report["note"]}
    out: Dict[str, Any] = {
        "init_tensors_compared": report.get("compared"),
        "init_tensors_equal": report.get("equal"),
        "init_max_abs_diff": report.get("max_abs_diff"),
        "trainable_sets_match": report.get("param_check", {}).get("ok"),
        "trainable_tensors_library": report.get("param_check", {}).get("library", {}).get("trainable_count"),
        "trainable_tensors_scratch": report.get("param_check", {}).get("scratch", {}).get("trainable_count"),
        "init_checks_passed": bool(report.get("ok")),
    }
    fwd = report.get("forward")
    if fwd is not None:
        out["forward_max_abs_diff"] = fwd.get("forward_max_abs_diff")
        out["forward_ok"] = fwd.get("forward_ok")
    fwd_lora = report.get("forward_lora")
    if fwd_lora is not None and "note" not in fwd_lora:
        out["lora_forward_max_abs_diff"] = fwd_lora.get("forward_max_abs_diff")
        out["lora_forward_ok"] = fwd_lora.get("forward_ok")
        out["lora_adapters_exercised"] = fwd_lora.get("lora_adapters_exercised")
    return out


def expand_experiment_names(names):
    """'all' means every ViT-B registry entry. ViT-L entries must be named explicitly:
    their runs are long and memory-heavy and should never be queued by accident."""
    if "all" in names:
        return [n for n, c in EXPERIMENTS.items() if c.arch == "vitb"]
    return list(names)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ViT and LoRA experiments with matched initialization and structured logging."
    )
    parser.add_argument(
        "--experiment",
        "-e",
        nargs="+",
        help="One or more experiment names to run, or 'all' (every ViT-B entry; ViT-L/H entries "
             "must be named explicitly because their runs are long and memory-heavy).",
    )
    parser.add_argument(
        "--seeds",
        "-s",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds to run (overrides config defaults).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the learning rate from the experiment config (default: use the config's lr). "
             "The effective value is recorded in the result JSON.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override the epoch cap from the experiment config. Intended for smoke tests and "
             "regression checks; the effective value is recorded in training_setup and aggregate.py "
             "flags truncated runs. Combine with --results-dir so truncated runs do not land "
             "in ./results.",
    )
    parser.add_argument(
        "--impl",
        choices=["scratch", "library", "paired"],
        default=None,
        help="Which implementation(s) to execute: 'scratch', 'library', or 'paired' (default: paired). "
             "An explicit 'paired' is an error if the experiment config cannot supply both.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save JSON experiment results (default: ./results).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save best checkpoint weights (default: ./checkpoints).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing result JSONs. Without it, the run aborts BEFORE any "
             "training if any result file it would write already exists.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all registered experiment configurations and exit.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Build paired models for one seed, verify init matching, print the report, and exit without training.",
    )
    parser.add_argument(
        "--forward-tol",
        type=float,
        default=1e-3,
        help="Fail before training if max|Δlogits| between implementations exceeds this (default 0.001; "
             "fp32 differences between two implementations are normally ~1e-5, so this leaves an "
             "order of magnitude of headroom while still catching structural mismatches).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── List experiments ──
    if args.list:
        print("\nRegistered Experiments:")
        print(f"  {'Name':<24} | {'Type':<12} | {'Rank / Alpha':<14} | {'Impls':<18} | Targets")
        print("  " + "-" * 90)
        for name, cfg in EXPERIMENTS.items():
            if cfg.lora:
                exp_type = "LoRA"
                r_a = f"r={cfg.lora.rank}, a={cfg.lora.alpha}"
                targets = ", ".join(cfg.lora.target_attention + cfg.lora.target_mlp)
                if cfg.lora.layers is not None:
                    targets += f"  [layers {cfg.lora.layers}]"
            else:
                exp_type = "Full FT"
                r_a = "-"
                targets = "All (no LoRA)"
            impls = ",".join(cfg.implementations)
            print(f"  {name:<24} | {exp_type:<12} | {r_a:<14} | {impls:<18} | {targets}")
        print()
        return

    if not args.experiment:
        print("Error: Please specify at least one experiment using --experiment or view available with --list.")
        sys.exit(1)

    exp_names = expand_experiment_names(args.experiment)

    # Validate experiments
    for exp_name in exp_names:
        if exp_name not in EXPERIMENTS:
            print(f"Error: Unknown experiment '{exp_name}'. Use --list to see available options.")
            sys.exit(1)

    import torch
    from peft import PeftModel
    from models import build_paired_models
    from train_utils import train_and_evaluate, result_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing experiments on device: {device}")

    # Results collector for summary table
    summary_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        name: {"scratch": [], "library": []} for name in exp_names
    }

    # Resolve --impl once: 'paired' is the default, but an EXPLICIT 'paired'
    # request is treated strictly (see resolve_implementations).
    impl_arg = args.impl or "paired"
    impl_explicit = args.impl is not None

    # ── Verify-only mode: build paired models, prove init matching, exit ──
    # Verifies EVERY requested experiment, then reports — a single-impl
    # experiment does not stop the remaining checks. Exit 2 (after the loop)
    # means "at least one experiment verified nothing"; exit 1 means a check
    # failed (raised inside enforce_checks).
    if args.verify_only:
        single_impl_seen = False
        for exp_name in exp_names:
            base_config = EXPERIMENTS[exp_name]
            seeds = args.seeds or base_config.seeds
            verify_config = copy.deepcopy(base_config)
            verify_config.implementations = resolve_implementations(
                base_config, impl_arg, explicit=impl_explicit)
            print(f"\n{'=' * 70}")
            print(f"VERIFY-ONLY: {exp_name} | Seed: {seeds[0]}")
            print(f"{'=' * 70}")
            models = build_paired_models(verify_config, seed=seeds[0])
            report = run_pre_train_checks(models, seeds[0], device, args.forward_tol, config=verify_config)
            print(json.dumps(report, indent=2))
            enforce_checks(report, "Verify-Only")
            if "note" in report and "library_only" not in report:
                # verify-only exists to prove pairing; with one implementation it
                # can only prove the model BUILDS. Say so loudly.
                print(f"[Verify-Only] WARNING: {exp_name} built only one implementation, so "
                      f"paired-init verification did NOT run for it. This confirms the model "
                      f"builds, not that initialization matches.")
                single_impl_seen = True
        if single_impl_seen:
            sys.exit(2)
        return

    # ── Overwrite guard: refuse BEFORE any training if a target JSON exists ──
    # A multi-hour sweep must not silently replace earlier valid results.
    if not args.overwrite:
        existing = []
        for exp_name in exp_names:
            cfg = EXPERIMENTS[exp_name]
            impls = resolve_implementations(cfg, impl_arg, explicit=impl_explicit)
            for seed in (args.seeds or cfg.seeds):
                for impl in impls:
                    path = result_path(args.results_dir, exp_name, impl, seed)
                    if path.exists():
                        existing.append(str(path))
        if existing:
            print("Error: refusing to overwrite existing results (pass --overwrite to replace, "
                  "or use --results-dir for a separate location):")
            for path in existing:
                print(f"  {path}")
            sys.exit(1)

    for exp_name in exp_names:
        base_config = EXPERIMENTS[exp_name]
        seeds = args.seeds or base_config.seeds

        # Determine implementations to build (CLI request ∩ config support)
        active_config = copy.deepcopy(base_config)
        active_config.implementations = resolve_implementations(
            base_config, impl_arg, explicit=impl_explicit)

        for seed in seeds:
            print(f"\n{'#' * 70}")
            print(f"# Experiment: {exp_name} | Seed: {seed} | Impls: {active_config.implementations}")
            print(f"{'#' * 70}")

            # Build models with paired initialization
            models = build_paired_models(active_config, seed=seed)

            # Full verification gate: init tensors, trainable parameter set, forward outputs
            report = run_pre_train_checks(models, seed, device, args.forward_tol, config=active_config)
            enforce_checks(report, "Init Check")

            # Execution order: library first, then scratch (or whatever subset is configured)
            order = [impl for impl in ["library", "scratch"] if impl in models]

            for impl_name in order:
                # pop, not index: for library+LoRA runs, train_and_evaluate
                # reloads the best checkpoint by building a FRESH model and
                # rebinding its local reference. If the dict still held the old
                # one, that dead model's ~0.34GB would stay resident through the
                # batch-1 inference probe and inflate inference_peak_memory.
                model = models.pop(impl_name).to(device)
                use_logits = (impl_name == "library")

                # Custom save/load functions for PEFT
                save_fn = None
                load_fn = None
                if impl_name == "library" and active_config.lora:
                    def peft_save(m, p):
                        # PEFT writes a DIRECTORY (adapter weights only). Name the
                        # path like a directory instead of impersonating a .pt file,
                        # and report the path actually written so the training loop
                        # reloads from the right place.
                        peft_dir = p.parent / (p.stem + "_peft")
                        m.save_pretrained(peft_dir)
                        return peft_dir

                    def peft_load(m, p):
                        from models import hf_classifier_class
                        # Load the adapter INTO the existing model instead of
                        # rebuilding a base + wrapper. During LoRA training the
                        # base is frozen, so its weights are byte-identical
                        # before and after training; rebuilding it (a) leaves a
                        # second ~0.32GB model resident through the batch-1
                        # inference probe, inflating inference_peak_memory for
                        # library runs, and (b) spikes peak_memory_run during
                        # reload. load_adapter restores the adapter AND the
                        # modules_to_save head from the same checkpoint.
                        if isinstance(m, PeftModel):
                            m.load_adapter(str(p), adapter_name="default")
                            m.to(device)
                            return m
                        base = hf_classifier_class(active_config.model_local_path).from_pretrained(
                            active_config.model_local_path,
                            num_labels=active_config.num_labels,
                            ignore_mismatched_sizes=True,
                            local_files_only=True,
                        ).to(device)
                        return PeftModel.from_pretrained(base, p)

                    save_fn = peft_save
                    load_fn = peft_load

                # Run training and evaluation
                result = train_and_evaluate(
                    model=model,
                    run_config=active_config,
                    seed=seed,
                    results_dir=args.results_dir,
                    checkpoint_dir=args.checkpoint_dir,
                    use_logits=use_logits,
                    lr=args.lr,
                    max_epochs=args.max_epochs,
                    verification=summarize_verification(report),
                    save_fn=save_fn,
                    load_fn=load_fn,
                    impl_name=impl_name,
                )

                summary_results[exp_name][impl_name].append(result)

                # Clean up GPU memory for next model / seed
                model.cpu()
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ── Final multi-seed summary table ──
    print_summary_table(summary_results)

    # Machine-readable copy of the same statistics. Under SLURM the file is
    # tagged with the job id so concurrent jobs (e.g. a 2-3 way seed split)
    # never race on one summary.json; per-run JSONs and checkpoints are
    # already collision-free via per-experiment nesting. Locally (no
    # SLURM_JOB_ID) it stays plain summary.json.
    summary_dir = Path(args.results_dir) if args.results_dir else (Path(__file__).resolve().parent / "results")
    job_id = os.environ.get("SLURM_JOB_ID")
    summary_file = f"summary_job{job_id}.json" if job_id else "summary.json"
    write_summary_json(summary_results, summary_dir / summary_file)


if __name__ == "__main__":
    main()
