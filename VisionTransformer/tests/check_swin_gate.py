"""Pass/fail check for the Swin-B gate job (job_swin_gate_2.sh), and the certificate the main jobs require.

    python tests/check_swin_gate.py                       # after the gate's smoke runs: check, write certificate
    python tests/check_swin_gate.py --verify-certificate  # at the start of a main job: same code as the gate?
    python tests/check_swin_gate.py --check-existing E1 E2 # main job: outputs already on disk made by this code?

The gate rehearses every step of the main Swin jobs on 1-epoch runs (training, adapter export, profiler,
merge study, aggregation). This script reads what they wrote and fails (exit 1) on anything that would
make the main runs fail or give wrong numbers. Only when everything passes does it write
results_smoke/gate_certificate.json, holding a hash of every script the main jobs run. The main jobs
refuse to start without a certificate for the code currently on disk, so code synced after the gate
cannot slip into a 24 h job unchecked.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GiB = 1024 ** 3
MiB = 1024 ** 2
CODE = ["run.py", "train_utils.py", "models.py", "vision_transformer.py", "lora.py", "experiments.py",
        "profile_memory.py", "merge_study.py", "aggregate.py",
        "job_swin_gate_2.sh", "job_swin_ft_2.sh", "job_swin_lora_2.sh", "tests/check_swin_gate.py"]
OUT = Path(os.environ.get("GATE_DIR", ROOT))      # where the smoke outputs are (override only for testing)
CERT = OUT / "results_smoke" / "gate_certificate.json"
FT, LORA = "swinb_baseline_lr1e-4", "swinb_lora_attn_mlp_r8"
IMAGENET_MEAN = [0.485, 0.456, 0.406]

# Thresholds, fixed before the gate runs.
MIN_ACC_AFTER_1_EPOCH = 50.0     # pretrained Swin-B after one epoch is far above this; below it means a broken
                                 # backbone, wrong normalization or wrong labels (a random model scores ~1%)
MAX_RESERVED_FRACTION = 0.90     # of the slice's memory, at batch 64
PROFILER_PEAK_TOL_GIB = 0.05     # profiler batch-64 peak vs the training run's peak (ViT-B agreed within 0.01)
MERGE_TOL = 1e-3                 # max |Δlogit| merged vs unmerged
EPOCHS_PER_RUN_ESTIMATE = 12     # conservative: ViT-B runs stopped after 6-10 epochs
JOB_CAP_H = 24.0


def code_hash() -> str:
    h = hashlib.sha256()
    for name in CODE:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()


def verify_certificate() -> int:
    if not CERT.exists():
        print(f"REFUSING: {CERT} not found. Run job_swin_gate_2.sh first (it writes the certificate only if "
              f"every check passes).")
        return 1
    cert = json.loads(CERT.read_text())
    now = code_hash()
    if cert.get("code_sha256") != now:
        print(f"REFUSING: the code on disk ({now[:12]}) is not the code the gate passed "
              f"({str(cert.get('code_sha256'))[:12]}). Files were changed or synced after the gate; rerun it.")
        return 1
    print(f"gate certificate OK: code {now[:12]} passed the gate at {cert.get('created')}")
    return 0


def check_existing(exps) -> int:
    """The main jobs skip every output that already exists. Refuse if any of those was produced by
    different code (e.g. left over from an earlier attempt): skipping it would silently mix versions."""
    from train_utils import code_provenance
    now = code_provenance()["source_sha256"]
    bad, seen = [], 0
    for exp in exps:
        for pattern in (f"results/{exp}/library_seed*.json", f"results_profile/{exp}/library_seed*_*.json",
                        f"results_merge_trained/{exp}/seed*.json"):
            for f in sorted(OUT.glob(pattern)):
                seen += 1
                try:
                    sha = (json.loads(f.read_text()).get("provenance") or {}).get("source_sha256")
                except json.JSONDecodeError:
                    sha = "<unreadable JSON>"
                if sha != now:
                    bad.append(f"{f} (code {str(sha)[:12]})")
    if bad:
        print(f"REFUSING: {len(bad)} existing output(s) were not produced by the code on disk ({now[:12]}) and "
              f"would be skipped and mixed in. Move them away (archive) or delete them, then resubmit:")
        for b in bad:
            print("  " + b)
        return 1
    print(f"existing outputs OK: {seen} file(s) found, all produced by code {now[:12]}")
    return 0


class Checks:
    def __init__(self):
        self.failed, self.warned = [], []

    def need(self, ok, what, detail=""):
        print(f"  [{'ok' if ok else 'FAIL'}] {what}{(': ' + detail) if detail else ''}")
        if not ok:
            self.failed.append(what)

    def warn(self, ok, what, detail=""):
        print(f"  [{'ok' if ok else 'WARN'}] {what}{(': ' + detail) if detail else ''}")
        if not ok:
            self.warned.append(what)


def _load(path: Path, c: Checks):
    if not path.exists():
        c.need(False, f"{path} exists")
        return None
    return json.loads(path.read_text())


def check_run(exp: str, c: Checks) -> dict:
    from experiments import EXPERIMENTS, arch_dims, trainable_counts
    print(f"\n{exp} (1-epoch smoke run)")
    r = _load(OUT / "results_smoke" / exp / "library_seed42.json", c)
    if r is None:
        return {}
    m, cfg = r["metrics"], EXPERIMENTS[exp]
    v = r.get("verification") or {}
    c.need(v.get("library_only_ok") is True, "library-only pre-training check passed and recorded")
    norm = r["training_setup"]["input_normalization"]
    c.need(norm["mean"] == IMAGENET_MEAN and norm["source"] == "checkpoint",
           "ImageNet normalization read from the checkpoint", f"{norm['mean']} ({norm['source']})")
    c.need(r["model_dims"] and r["model_dims"].get("family") == "swin", "model geometry recorded as Swin")
    exp_elems = trainable_counts(arch_dims(cfg.model_local_path, True), cfg.num_labels,
                                 cfg.to_dict()["lora"])[0]
    got = round(m["static_memory_breakdown_gb"]["grads"] * GiB / 4)
    c.need(got == exp_elems, "trainable parameters = analytic count", f"{got:,} vs {exp_elems:,}")
    c.need(m["test_acc"] >= MIN_ACC_AFTER_1_EPOCH, f"test accuracy after 1 epoch >= {MIN_ACC_AFTER_1_EPOCH}%",
           f"{m['test_acc']:.2f}%")
    total = r["environment"].get("total_memory_gb") or 0
    res = m["peak_memory_reserved_gb_train"]
    c.need(total > 0 and res <= MAX_RESERVED_FRACTION * total, "batch 64 fits with headroom",
           f"peak reserved {res:.2f} of {total:.2f} GiB, peak allocated {m['peak_memory_gb_train']:.2f} GiB")
    c.warn(m.get("inference_memory_clean") is True, "in-run inference probe clean",
           "the profiler measures inference separately, so this only affects the in-run fields")
    c.warn(m["peak_memory_gb_run"] == m["peak_memory_gb_train"], "run peak = train peak")
    if cfg.lora:
        ad = OUT / "results_smoke" / exp / "library_seed42_adapter.pt"
        c.need(ad.exists(), "adapter exported")
        size = m.get("adapter_export_size_mb")
        c.need(size is not None and abs(size - 4 * exp_elems / MiB) < 0.2, "adapter size = 4 B x trainable",
               f"{size} MiB vs {4 * exp_elems / MiB:.2f}")
    epoch_s = m["avg_epoch_train_time_sec"] + m["avg_epoch_val_time_sec"]
    print(f"  epoch: {m['avg_epoch_train_time_sec']:.0f} s train + {m['avg_epoch_val_time_sec']:.0f} s val; "
          f"fwd/bwd/opt {m['avg_batch_forward_sec'] * 1e3:.1f}/{m['avg_batch_backward_sec'] * 1e3:.1f}/"
          f"{m['avg_batch_optim_sec'] * 1e3:.2f} ms per batch")
    return {"epoch_s": epoch_s, "peak": m["peak_memory_gb_train"]}


def check_profile(exp: str, run_peak, c: Checks):
    print(f"\n{exp} (profiler)")
    tr = _load(OUT / "results_smoke_profile" / exp / "library_seed42_train.json", c)
    if tr is not None:
        pts = {p["batch_size"]: p for p in tr["train"]["sweep"]}
        p64 = pts.get(64)
        c.need(p64 is not None and not p64["oom"], "profiler train step at batch 64 ran")
        if p64 is not None and not p64["oom"] and run_peak is not None:
            c.need(abs(p64["peak_gb"] - run_peak) <= PROFILER_PEAK_TOL_GIB,
                   "profiler batch-64 peak matches the training run", f"{p64['peak_gb']:.3f} vs {run_peak:.3f} GiB")
    inf = _load(OUT / "results_smoke_profile" / exp / "library_seed42_inference.json", c)
    if inf is not None:
        want = {"fp32_unmerged", "fp32_merged"} if "lora" in exp else {"fp32_unmerged"}
        c.need(want <= set(inf["inference"]), "inference variants measured", ", ".join(sorted(inf["inference"])))
        if "lora" in exp:
            c.need(inf["adapter_source"].startswith("trained:"), "profiler used the trained adapter",
                   inf["adapter_source"])


def check_merge(c: Checks):
    print(f"\n{LORA} (merge study)")
    r = _load(OUT / "results_smoke_merge" / LORA / "seed42.json", c)
    if r is None:
        return
    c.need(r["adapter_source"].startswith("trained:"), "merge study used the trained adapter")
    lib = r["correctness_fp32"]["library"]
    c.need(lib["max_abs_diff_merged_vs_unmerged"] <= MERGE_TOL, "merged = unmerged logits",
           f"{lib['max_abs_diff_merged_vs_unmerged']:.2e}")
    a = r["test_accuracy"]["library"]
    c.need(a["acc_merged"] == a["acc_unmerged"] and a["pred_agreement_merged_vs_unmerged"] == 1.0,
           "merged and unmerged predict the same classes", f"{a['acc_unmerged']:.2f}% on {a['n_test']} images")


def check_aggregate(c: Checks):
    print("\naggregate.py on the smoke results")
    s = _load(OUT / "results_smoke" / "aggregate" / "stats.json", c)
    if s is None:
        return
    expected = ("truncated run", "run peak != train peak")       # 1-epoch runs; the second is a WARN above
    unexpected = [p for p in s["assertion_failures"] if not any(e in p for e in expected)]
    c.need(not unexpected, "no unexpected aggregator assertion failures", "; ".join(unexpected)[:400])


def estimate(times: dict):
    print("\nEstimated main-job durations (conservative: "
          f"{EPOCHS_PER_RUN_ESTIMATE} epochs per run; real runs stop earlier when validation loss stops improving)")
    for job, exp, n_runs in (("job_swin_ft_2.sh  ", FT, 6), ("job_swin_lora_2.sh", LORA, 6)):
        if exp not in times or not times[exp]:
            continue
        h = n_runs * EPOCHS_PER_RUN_ESTIMATE * times[exp]["epoch_s"] / 3600
        note = "fits the 24 h cap" if h < JOB_CAP_H - 2 else \
            "MAY EXCEED the 24 h cap: the job is rerun-safe (resubmit to continue), or pass fewer SEEDS"
        print(f"  {job}: ~{h:.1f} h for {n_runs} runs (+ ~1 h profiler/merge for LoRA) -> {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-certificate", action="store_true")
    ap.add_argument("--check-existing", nargs="+", metavar="EXPERIMENT")
    args = ap.parse_args()
    if args.verify_certificate:
        sys.exit(verify_certificate())
    if args.check_existing:
        sys.exit(check_existing(args.check_existing))
    if CERT.exists():
        CERT.unlink()                     # a failed re-check must not leave an old certificate behind
    c = Checks()
    times = {exp: check_run(exp, c) for exp in (FT, LORA)}
    for exp in (FT, LORA):
        check_profile(exp, times.get(exp, {}).get("peak"), c)
    check_merge(c)
    check_aggregate(c)
    estimate(times)
    if c.warned:
        print(f"\n{len(c.warned)} warning(s): {c.warned}")
    if c.failed:
        print(f"\nGATE FAILED ({len(c.failed)}): {c.failed}\nNo certificate written; the main jobs will refuse to start.")
        sys.exit(1)
    import datetime
    CERT.write_text(json.dumps({"code_sha256": code_hash(), "files": CODE,
                                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                                "warnings": c.warned}, indent=2))
    print(f"\nGATE PASSED. Certificate written: {CERT} (code {code_hash()[:12]})")


if __name__ == "__main__":
    main()
