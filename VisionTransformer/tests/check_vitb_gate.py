"""Pass/fail check for the ViT-B placement gate job (job_vitb_gate_2.sh), and the certificate the main jobs require.

    python tests/check_vitb_gate.py                        # after the gate's smoke runs: check, write certificate
    python tests/check_vitb_gate.py --verify-certificate   # at the start of a main job: same code as the gate?
    python tests/check_vitb_gate.py --check-existing E1 E2 # main job: outputs already on disk made by this code?

Same idea as tests/check_swin_gate.py, for PAIRED ViT-B runs (library + scratch). The gate rehearses every step
of job_vitb_lora_2.sh on 1-epoch runs (paired training, adapter export, profiler for both implementations, merge
study, aggregation). This script reads what they wrote and fails (exit 1) on anything that would make the main
runs fail or give wrong numbers. On top of the Swin checks it requires:
  - the paired pre-training gate passed and was recorded (identical init, identical trainable set, forward
    equivalence plain and with LoRA active), in both result files;
  - library and scratch agree on static memory and peak memory (an attention-kernel mismatch once caused a
    ~1.3 GiB gap), and the merge study's library-vs-scratch merged logits agree;
  - regression against the archived ViT-B run (pre-refactor code) for lora_attn_r8 seed 42: identical static
    memory, epoch-1 peak memory and trainable tensors; epoch-1 losses within 0.02 (warning only, as in
    tests/compare_regression.py).
Only when everything passes does it write results_smoke_vitb/gate_certificate.json, holding a hash of every
script the main jobs run. Independent of the Swin certificate: neither gate's files include the other's job
scripts, so both certificates stay valid as long as the shared .py files do not change.
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
IMPLS = ("library", "scratch")
CODE = ["run.py", "train_utils.py", "models.py", "vision_transformer.py", "lora.py", "experiments.py",
        "profile_memory.py", "merge_study.py", "aggregate.py",
        "job_vitb_gate_2.sh", "job_vitb_lora_2.sh", "job_vitb_lr_2.sh", "tests/check_vitb_gate.py"]
OUT = Path(os.environ.get("GATE_DIR", ROOT))      # where the smoke outputs are (override only for testing)
SMOKE = "results_smoke_vitb"
CERT = OUT / SMOKE / "gate_certificate.json"
# Archived ViT-B results (pre-refactor code). On the cluster the code runs from VisionTransformer_2, nested in the
# original tree, so the archive is the parent's results/.
ARCHIVE = Path(os.environ.get("ARCHIVE_RESULTS", ROOT.parent / "results"))
LORAS = ("lora_attn_r8", "lora_attn_r8_L6-11")    # all blocks (the reference) and a block subset
LR_RUNS = ("baseline_lr3e-6", "lora_attn_r8_lr1e-3")  # job_vitb_lr_2.sh: lowest and highest new rate, library only
REGRESSION_EXP = "lora_attn_r8"
VIT_MEAN = [0.5, 0.5, 0.5]                        # google/vit-base-patch16-224's preprocessor_config.json
MAIN_DIRS = ("results_v2", "results_v2_profile", "results_v2_merge")

# Thresholds, fixed before the gate runs.
MIN_ACC_AFTER_1_EPOCH = 50.0     # pretrained ViT-B after one epoch is far above this (a random model scores ~1%)
MIN_ACC_LR_1_EPOCH = 20.0        # learning-rate runs: the extreme rates learn slower (full FT at 1e-3 was at 41% after
                                 # epoch 1), so this only catches a run that does not learn or diverges
MAX_RESERVED_FRACTION = 0.90     # of the slice's memory, at batch 64
PROFILER_PEAK_TOL_GIB = 0.05     # profiler batch-64 peak vs the training run's peak (ViT-B agreed within 0.01)
PAIR_PEAK_TOL_GIB = 0.05         # library vs scratch peak (archived ViT-B LoRA runs: 0.002)
MEM_TOL_GIB = 0.005              # static memory, and regression against the archive (as compare_regression.py)
PAIR_ACC_WARN_PP = 0.5           # library vs scratch accuracy after 1 epoch: warning only
LOSS_TOL = 0.02                  # regression epoch-1 losses: warning only (as compare_regression.py)
MERGE_TOL = 1e-3                 # max |Δlogit| merged vs unmerged, and library vs scratch merged
EPOCHS_PER_RUN_ESTIMATE = 12     # conservative: archived ViT-B LoRA runs stopped after 9-11 epochs
SEEDS_PER_JOB = 5
JOB_CAP_H = 24.0


def code_hash() -> str:
    h = hashlib.sha256()
    for name in CODE:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()


def verify_certificate() -> int:
    if not CERT.exists():
        print(f"REFUSING: {CERT} not found. Run job_vitb_gate_2.sh first (it writes the certificate only if "
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


def check_existing(exps, dirs) -> int:
    """The main jobs skip every output that already exists. Refuse if any of those was produced by
    different code (e.g. left over from an earlier attempt): skipping it would silently mix versions."""
    from train_utils import code_provenance
    now = code_provenance()["source_sha256"]
    res, prof, merge = dirs
    bad, seen = [], 0
    for exp in exps:
        for pattern in (f"{res}/{exp}/*_seed*.json", f"{prof}/{exp}/*_seed*_*.json", f"{merge}/{exp}/seed*.json"):
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


def check_run(exp: str, impl: str, c: Checks):
    from experiments import EXPERIMENTS, arch_dims, trainable_counts
    print(f"\n{exp} / {impl} (1-epoch smoke run)")
    r = _load(OUT / SMOKE / exp / f"{impl}_seed42.json", c)
    if r is None:
        return None
    m, cfg = r["metrics"], EXPERIMENTS[exp]
    v = r.get("verification") or {}
    c.need(all(v.get(k) is True for k in ("init_checks_passed", "trainable_sets_match", "forward_ok",
                                          "lora_forward_ok")),
           "paired pre-training gate passed and recorded",
           f"init {v.get('init_tensors_equal')}/{v.get('init_tensors_compared')} equal, "
           f"forward {v.get('forward_max_abs_diff')}, with LoRA {v.get('lora_forward_max_abs_diff')}")
    c.need(v.get("init_max_abs_diff") == 0.0, "identical initial tensors", f"max diff {v.get('init_max_abs_diff')}")
    norm = r["training_setup"]["input_normalization"]
    c.need(norm["mean"] == VIT_MEAN and norm["source"] == "checkpoint",
           "ViT normalization read from the checkpoint", f"{norm['mean']} ({norm['source']})")
    dims = r.get("model_dims") or {}            # ViT geometry carries no "family" key (only Swin's does)
    c.need(dims.get("family", "vit") == "vit" and dims.get("hidden_size") == 768
           and dims.get("num_hidden_layers") == 12, "model geometry recorded as ViT-B",
           f"hidden {dims.get('hidden_size')}, {dims.get('num_hidden_layers')} blocks")
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
    ad = OUT / SMOKE / exp / f"{impl}_seed42_adapter.pt"
    c.need(ad.exists(), "adapter exported")
    size = m.get("adapter_export_size_mb")
    c.need(size is not None and abs(size - 4 * exp_elems / MiB) < 0.2, "adapter size = 4 B x trainable",
           f"{size} MiB vs {4 * exp_elems / MiB:.2f}")
    print(f"  epoch: {m['avg_epoch_train_time_sec']:.0f} s train + {m['avg_epoch_val_time_sec']:.0f} s val; "
          f"fwd/bwd/opt {m['avg_batch_forward_sec'] * 1e3:.1f}/{m['avg_batch_backward_sec'] * 1e3:.1f}/"
          f"{m['avg_batch_optim_sec'] * 1e3:.2f} ms per batch")
    return r


def check_lr_run(exp: str, c: Checks):
    """A 1-epoch library-only run as job_vitb_lr_2.sh makes it: the rate actually used, the model, the count."""
    import math
    from experiments import EXPERIMENTS, arch_dims, trainable_counts
    print(f"\n{exp} / library (1-epoch smoke run, learning-rate check)")
    r = _load(OUT / SMOKE / exp / "library_seed42.json", c)
    if r is None:
        return None
    m, cfg = r["metrics"], EXPERIMENTS[exp]
    c.need(r["config"]["lr"] == cfg.lr, "learning rate used = registry", f"{r['config']['lr']} vs {cfg.lr}")
    norm = r["training_setup"]["input_normalization"]
    c.need(norm["mean"] == VIT_MEAN and norm["source"] == "checkpoint",
           "ViT normalization read from the checkpoint", f"{norm['mean']} ({norm['source']})")
    exp_elems = trainable_counts(arch_dims(cfg.model_local_path, True), cfg.num_labels,
                                 cfg.to_dict()["lora"])[0]
    got = round(m["static_memory_breakdown_gb"]["grads"] * GiB / 4)
    c.need(got == exp_elems, "trainable parameters = analytic count", f"{got:,} vs {exp_elems:,}")
    losses = [r["per_epoch"][0][k] for k in ("train_loss", "val_loss")]
    c.need(all(math.isfinite(x) for x in losses), "epoch-1 losses finite", f"{losses}")
    c.need(m["test_acc"] >= MIN_ACC_LR_1_EPOCH, f"test accuracy after 1 epoch >= {MIN_ACC_LR_1_EPOCH}%",
           f"{m['test_acc']:.2f}%")
    total = r["environment"].get("total_memory_gb") or 0
    res = m["peak_memory_reserved_gb_train"]
    c.need(total > 0 and res <= MAX_RESERVED_FRACTION * total, "batch 64 fits with headroom",
           f"peak reserved {res:.2f} of {total:.2f} GiB")
    if cfg.lora is not None:
        c.need((OUT / SMOKE / exp / "library_seed42_adapter.pt").exists(), "adapter exported")
    return r


def check_pair(exp: str, runs: dict, c: Checks):
    print(f"\n{exp} (library vs scratch)")
    lib, scr = runs.get("library"), runs.get("scratch")
    if lib is None or scr is None:
        c.need(False, "both implementations ran")
        return
    lm, sm = lib["metrics"], scr["metrics"]
    for k in lm["static_memory_breakdown_gb"]:
        a, b = lm["static_memory_breakdown_gb"][k], sm["static_memory_breakdown_gb"].get(k, float("nan"))
        c.need(abs(a - b) <= MEM_TOL_GIB, f"static {k} equal", f"{a:.4f} vs {b:.4f} GiB")
    c.need(abs(lm["peak_memory_gb_train"] - sm["peak_memory_gb_train"]) <= PAIR_PEAK_TOL_GIB,
           "peak allocated agrees", f"{lm['peak_memory_gb_train']:.3f} vs {sm['peak_memory_gb_train']:.3f} GiB")
    d = abs(lm["test_acc"] - sm["test_acc"])
    c.warn(d <= PAIR_ACC_WARN_PP, f"test accuracy within {PAIR_ACC_WARN_PP} pp after 1 epoch",
           f"{lm['test_acc']:.2f} vs {sm['test_acc']:.2f}")


def check_regression(runs: dict, c: Checks):
    """The refactored code against the archived pre-refactor run, on what one epoch can show."""
    print(f"\n{REGRESSION_EXP} seed 42 vs the archived run ({ARCHIVE})")
    for impl in IMPLS:
        new = runs.get(impl)
        old_f = ARCHIVE / REGRESSION_EXP / f"{impl}_seed42.json"
        if not old_f.exists():
            c.warn(False, f"{impl}: archived run found", f"{old_f} missing; regression not checked")
            continue
        if new is None:
            continue
        old = json.loads(old_f.read_text())
        nb, ob = new["metrics"]["static_memory_breakdown_gb"], old["metrics"]["static_memory_breakdown_gb"]
        for k in ob:
            c.need(abs(nb.get(k, float("nan")) - ob[k]) <= MEM_TOL_GIB, f"{impl}: static {k} = archive",
                   f"{nb.get(k, float('nan')):.4f} vs {ob[k]:.4f} GiB")
        for k in ("trainable_tensors_library", "trainable_tensors_scratch"):
            nv, ov = (new.get("verification") or {}).get(k), (old.get("verification") or {}).get(k)
            c.need(nv == ov, f"{impl}: verification.{k} = archive", f"{nv} vs {ov}")
        ne, oe = new["per_epoch"][0], old["per_epoch"][0]
        if "peak_memory_allocated_gb" in oe:
            c.need(abs(ne["peak_memory_allocated_gb"] - oe["peak_memory_allocated_gb"]) <= MEM_TOL_GIB,
                   f"{impl}: epoch-1 peak allocated = archive",
                   f"{ne['peak_memory_allocated_gb']:.4f} vs {oe['peak_memory_allocated_gb']:.4f} GiB")
        for k in ("train_loss", "val_loss"):
            d = abs(ne[k] - oe[k])
            c.warn(d <= LOSS_TOL, f"{impl}: epoch-1 {k} within {LOSS_TOL} of archive",
                   f"{ne[k]:.4f} vs {oe[k]:.4f} (diff {d:.4f})")


def check_profile(exp: str, impl: str, run_peak, c: Checks):
    print(f"\n{exp} / {impl} (profiler)")
    tr = _load(OUT / f"{SMOKE}_profile" / exp / f"{impl}_seed42_train.json", c)
    if tr is not None:
        pts = {p["batch_size"]: p for p in tr["train"]["sweep"]}
        p64 = pts.get(64)
        c.need(p64 is not None and not p64["oom"], "profiler train step at batch 64 ran")
        if p64 is not None and not p64["oom"] and run_peak is not None:
            c.need(abs(p64["peak_gb"] - run_peak) <= PROFILER_PEAK_TOL_GIB,
                   "profiler batch-64 peak matches the training run", f"{p64['peak_gb']:.3f} vs {run_peak:.3f} GiB")
        c.need(tr["adapter_source"].startswith("trained:"), "profiler (train) used the trained adapter",
               tr["adapter_source"])
    inf = _load(OUT / f"{SMOKE}_profile" / exp / f"{impl}_seed42_inference.json", c)
    if inf is not None:
        want = {"fp32_unmerged", "fp32_merged"}
        c.need(want <= set(inf["inference"]), "inference variants measured", ", ".join(sorted(inf["inference"])))
        c.need(inf["adapter_source"].startswith("trained:"), "profiler (inference) used the trained adapter",
               inf["adapter_source"])


def check_merge(exp: str, c: Checks):
    print(f"\n{exp} (merge study)")
    r = _load(OUT / f"{SMOKE}_merge" / exp / "seed42.json", c)
    if r is None:
        return
    c.need(r["adapter_source"].startswith("trained:"), "merge study used the trained adapter")
    corr = r["correctness_fp32"]
    for impl in IMPLS:
        if impl not in corr:
            c.need(False, f"{impl}: merge study ran")
            continue
        c.need(corr[impl]["max_abs_diff_merged_vs_unmerged"] <= MERGE_TOL, f"{impl}: merged = unmerged logits",
               f"{corr[impl]['max_abs_diff_merged_vs_unmerged']:.2e}")
        a = r["test_accuracy"][impl]
        c.need(a["acc_merged"] == a["acc_unmerged"] and a["pred_agreement_merged_vs_unmerged"] == 1.0,
               f"{impl}: merged and unmerged predict the same classes",
               f"{a['acc_unmerged']:.2f}% on {a['n_test']} images")
    lvs = (corr.get("library_vs_scratch_merged") or {}).get("max_abs_diff")
    c.need(lvs is not None and lvs <= MERGE_TOL, "library = scratch after merging the same adapter", f"{lvs}")


def check_aggregate(c: Checks):
    print("\naggregate.py on the smoke results")
    s = _load(OUT / SMOKE / "aggregate" / "stats.json", c)
    if s is None:
        return
    expected = ("truncated run", "run peak != train peak")       # 1-epoch runs; the second is a WARN above
    # Library-only ViT-B runs (learning-rate check) record no paired gate: job_vitb_lr_2.sh runs the paired
    # check once per experiment with run.py --verify-only before training, so only that flag is expected for them.
    lr_gate = tuple(f"{exp}/library/" for exp in LR_RUNS)
    unexpected = [p for p in s["assertion_failures"] if not any(e in p for e in expected)
                  and not (p.startswith(lr_gate) and "verification gate" in p and p.endswith("= None"))]
    c.need(not unexpected, "no unexpected aggregator assertion failures", "; ".join(unexpected)[:400])


def estimate(runs: dict, lr_runs: dict):
    print(f"\nEstimated duration of one job_vitb_lora_2.sh job = one experiment x {SEEDS_PER_JOB} seeds, paired "
          f"(conservative: {EPOCHS_PER_RUN_ESTIMATE} epochs per run; real runs stop earlier when validation loss "
          f"stops improving)")
    for exp in LORAS:
        rs = runs.get(exp, {})
        if not all(rs.get(i) for i in IMPLS):
            continue
        epoch_s = sum(rs[i]["metrics"]["avg_epoch_train_time_sec"] + rs[i]["metrics"]["avg_epoch_val_time_sec"]
                      for i in IMPLS)
        h = SEEDS_PER_JOB * EPOCHS_PER_RUN_ESTIMATE * epoch_s / 3600
        note = "fits the 24 h cap" if h + 1 < JOB_CAP_H - 2 else \
            "MAY EXCEED the 24 h cap: the job is rerun-safe (resubmit to continue), or pass fewer SEEDS"
        print(f"  {exp}: ~{h:.1f} h training + ~1 h profiler/merge study -> {note}")
    print("Estimated duration of job_vitb_lr_2.sh per experiment x 3 seeds, library only (same epoch estimate)")
    for exp, r in lr_runs.items():
        if r is None:
            continue
        epoch_s = r["metrics"]["avg_epoch_train_time_sec"] + r["metrics"]["avg_epoch_val_time_sec"]
        print(f"  {exp}: ~{3 * EPOCHS_PER_RUN_ESTIMATE * epoch_s / 3600:.1f} h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-certificate", action="store_true")
    ap.add_argument("--check-existing", nargs="+", metavar="EXPERIMENT")
    ap.add_argument("--dirs", nargs=3, default=list(MAIN_DIRS), metavar=("RESULTS", "PROFILE", "MERGE"),
                    help="--check-existing: the main job's output folders")
    args = ap.parse_args()
    if args.verify_certificate:
        sys.exit(verify_certificate())
    if args.check_existing:
        sys.exit(check_existing(args.check_existing, args.dirs))
    if CERT.exists():
        CERT.unlink()                     # a failed re-check must not leave an old certificate behind
    c = Checks()
    runs = {exp: {impl: check_run(exp, impl, c) for impl in IMPLS} for exp in LORAS}
    for exp in LORAS:
        check_pair(exp, runs[exp], c)
    check_regression(runs[REGRESSION_EXP], c)
    lr_runs = {exp: check_lr_run(exp, c) for exp in LR_RUNS}
    for exp in LORAS:
        for impl in IMPLS:
            r = runs[exp][impl]
            check_profile(exp, impl, r["metrics"]["peak_memory_gb_train"] if r else None, c)
        check_merge(exp, c)
    check_aggregate(c)
    estimate(runs, lr_runs)
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
