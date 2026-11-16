"""Compare regression reruns (refactored code) against the archived ViT-B result JSONs.

    python tests/compare_regression.py NEW_RESULTS_DIR [--old results] [--epochs N]

For every <exp>/<impl>_seed<N>.json in NEW_RESULTS_DIR, the archived file of the same name under --old is compared.
Tolerances are fixed here, BEFORE the runs:
  exact   trainable element/tensor counts, verification gates, best_epoch, stopping_epoch
  <=0.005 GiB  static memory and peak allocated memory (memory is deterministic; the slack is rounding / node differences)
  <=5 test images (0.05 pp)  test accuracy. The earlier adapter retrain on the OLD code matched its original within 1-4
          images, so this is the measured noise floor plus one image.
  per-epoch train/val loss (truncated full-FT check, epochs 1..N): WARN above 0.02 absolute. This tolerance is a
          pre-registered guess, not a measured noise floor: a WARN means "look", not "broken".
Exit status 1 on any FAIL.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("--old", default="results")
    ap.add_argument("--epochs", type=int, default=None, help="compare per-epoch losses for epochs 1..N")
    ap.add_argument("--acc-tol-pp", type=float, default=0.05)
    ap.add_argument("--mem-tol-gib", type=float, default=0.005)
    ap.add_argument("--loss-tol", type=float, default=0.02)
    a = ap.parse_args()

    rows, fails = [], 0

    def rec(tag, what, ok, detail, hard=True):
        nonlocal fails
        status = "PASS" if ok else ("FAIL" if hard else "WARN")
        fails += (not ok) and hard
        rows.append((tag, what, status, detail))

    files = sorted(Path(a.new).glob("*/*_seed*.json"))
    if not files:
        sys.exit(f"no result files under {a.new}")
    for f in files:
        old_f = Path(a.old) / f.parent.name / f.name
        tag = f"{f.parent.name}/{f.stem}"
        if not old_f.exists():
            rec(tag, "archived counterpart exists", False, str(old_f))
            continue
        n, o = json.loads(f.read_text()), json.loads(old_f.read_text())
        nm, om = n["metrics"], o["metrics"]
        nv, ov = n.get("verification") or {}, o.get("verification") or {}
        for k in ("trainable_tensors_library", "trainable_tensors_scratch", "init_checks_passed",
                  "trainable_sets_match", "forward_ok", "lora_forward_ok"):
            if k in ov or k in nv:
                rec(tag, f"verification.{k}", nv.get(k) == ov.get(k), f"{nv.get(k)} vs {ov.get(k)}")
        nb, ob = nm["static_memory_breakdown_gb"], om["static_memory_breakdown_gb"]
        for k in ob:
            rec(tag, f"static {k} GiB", abs(nb[k] - ob[k]) <= a.mem_tol_gib, f"{nb[k]:.4f} vs {ob[k]:.4f}")
        truncated = (n.get("training_setup") or {}).get("max_epochs") not in (None, o["config"].get("max_epochs"))
        if not truncated:
            rec(tag, "peak allocated GiB", abs(nm["peak_memory_gb_train"] - om["peak_memory_gb_train"]) <= a.mem_tol_gib,
                f"{nm['peak_memory_gb_train']:.4f} vs {om['peak_memory_gb_train']:.4f}")
            rec(tag, "best_epoch", nm["best_epoch"] == om["best_epoch"], f"{nm['best_epoch']} vs {om['best_epoch']}")
            rec(tag, "stopping_epoch", nm["stopping_epoch"] == om["stopping_epoch"],
                f"{nm['stopping_epoch']} vs {om['stopping_epoch']}")
            d = abs(nm["test_acc"] - om["test_acc"])
            rec(tag, f"test accuracy within {a.acc_tol_pp} pp", d <= a.acc_tol_pp + 1e-9,
                f"{nm['test_acc']:.2f} vs {om['test_acc']:.2f} (diff {d:.2f} pp = {round(d * 100)} images)")
        n_ep = a.epochs or (len(n["per_epoch"]) if truncated else 0)
        for e in range(n_ep):
            for k in ("train_loss", "val_loss"):
                d = abs(n["per_epoch"][e][k] - o["per_epoch"][e][k])
                rec(tag, f"epoch {e + 1} {k}", d <= a.loss_tol, f"{n['per_epoch'][e][k]:.4f} vs "
                    f"{o['per_epoch'][e][k]:.4f} (diff {d:.4f})", hard=False)
        # a refactored run must carry the NEW provenance, an archived run may not
        rec(tag, "new run has a source hash", bool((n.get("provenance") or {}).get("source_sha256")), "")
    w = max(len(r[1]) for r in rows)
    for tag, what, status, detail in rows:
        print(f"{status:4}  {tag:42} {what:{w}}  {detail}")
    print(f"\n{sum(r[2] == 'PASS' for r in rows)} pass, {sum(r[2] == 'WARN' for r in rows)} warn, {fails} fail")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
