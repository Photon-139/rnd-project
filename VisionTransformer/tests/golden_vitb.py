"""Golden reference for "the ViT-B pipeline did not change".

    python tests/golden_vitb.py capture OUT.json [--root DIR]   # record
    python tests/golden_vitb.py compare GOLDEN.json NEW.json    # diff

Capture it ONCE on the unmodified tree (pass --root pointing at a copy of the old
sources), then capture again on the refactored tree and compare. CPU only.

Recorded, for every registry experiment present in the golden file:
  * config dict (compare is restricted to the keys the golden file has, so purely
    additive new fields do not count as a change)
  * SHA-256 of every transferred init tensor (classifier head, LoRA A/B) and of the
    full state dict of BOTH implementations
  * trainable signature of both implementations
  * the full pre-training verification report (init equality, forward equivalence,
    forward equivalence with non-zero B)
  * logits of both implementations on a fixed input
  * data pipeline per seed: train/val split indices, labels of the first batches in
    the order train_and_evaluate would draw them, transform output on a fixed image
Exact equality is required for hashes and reports (same machine, same torch).
"""
import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

SEEDS = [42, 123]
BASELINE_ALIASES_SEEDS = [42]          # the three baselines share one model; one seed is enough


def _sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _sha_state(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def capture(root: Path, out: Path, data_dir: str, only=None):
    sys.path.insert(0, str(root))
    os.environ["DATA_DIR"] = data_dir
    import torch
    from experiments import EXPERIMENTS
    from models import build_paired_models, _extract_reference_tensors, trainable_signature
    import run as run_mod
    import train_utils

    device = torch.device("cpu")
    rec = {"torch": torch.__version__, "experiments": {}, "data": {}}

    for name, cfg in EXPERIMENTS.items():
        if only is not None and name not in only:
            continue
        entry = {"config": copy.deepcopy(cfg.to_dict()), "seeds": {}}
        seeds = BASELINE_ALIASES_SEEDS if name.startswith("baseline") and name != "baseline" else SEEDS
        for seed in seeds:
            print(f"[golden] {name} seed {seed}", flush=True)
            models = build_paired_models(copy.deepcopy(cfg), seed=seed)
            lib, scr = models["library"], models["scratch"]
            ref = _extract_reference_tensors(lib)
            ref_hash = hashlib.sha256()
            for k in sorted(ref, key=str):
                ref_hash.update(str(k).encode())
                ref_hash.update(ref[k].numpy().tobytes())
            g = torch.Generator().manual_seed(1234)
            x = torch.randn(2, 3, 224, 224, generator=g)
            logits = {}
            with torch.no_grad():
                for impl, m in (("library", lib), ("scratch", scr)):
                    m.eval()
                    out_t = m(x)
                    out_t = out_t.logits if hasattr(out_t, "logits") else out_t
                    logits[impl] = {"sha": _sha(out_t), "head": out_t.flatten()[:8].tolist()}
            report = run_mod.run_pre_train_checks(models, seed, device, 1e-3)
            entry["seeds"][str(seed)] = {
                "ref_sha": ref_hash.hexdigest(),
                "ref_keys": len(ref),
                "state_sha": {"library": _sha_state(lib), "scratch": _sha_state(scr)},
                "trainable": {"library": trainable_signature(lib), "scratch": trainable_signature(scr)},
                "report": report,
                "logits": logits,
            }
            del models, lib, scr
        rec["experiments"][name] = entry

    # data pipeline
    import torch
    from torchvision import datasets
    for seed in SEEDS:
        train_loader, val_loader, test_loader = train_utils.get_loaders(batch_size=64, seed=seed)
        tr, va = train_loader.dataset, val_loader.dataset
        gen = train_loader.sampler.generator
        gen.manual_seed(seed)                       # exactly what train_and_evaluate does
        labels = []
        first_pix = None
        for i, (xb, yb) in enumerate(train_loader):
            if i == 0:
                first_pix = _sha(xb)
            labels.append(yb.tolist())
            if i == 2:
                break
        rec["data"][str(seed)] = {
            "train_idx_sha": hashlib.sha256(str(list(tr.indices)).encode()).hexdigest(),
            "val_idx_sha": hashlib.sha256(str(list(va.indices)).encode()).hexdigest(),
            "first_batches_labels": labels,
            "first_batch_pixels_sha": first_pix,
            "n_train": len(tr), "n_val": len(va), "n_test": len(test_loader.dataset),
        }
    img = datasets.CIFAR100(root=data_dir, train=False, download=False,
                            transform=train_utils.transform)[0][0]
    rec["data"]["transform_first_test_image_sha"] = _sha(img)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"[golden] wrote {out}")


def _diff(a, b, path, bad, restrict=True):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:                                  # golden keys only: additive fields allowed
            if k == "model_local_path":              # where the checkpoint lives, not what the code does
                continue
            if k not in b:
                bad.append(f"{path}/{k}: missing in new")
            else:
                _diff(a[k], b[k], f"{path}/{k}", bad)
    elif a != b:
        bad.append(f"{path}: {str(a)[:80]} != {str(b)[:80]}")


def compare(golden: Path, new: Path):
    g, n = json.loads(golden.read_text()), json.loads(new.read_text())
    bad = []
    if g["torch"] != n["torch"]:
        print(f"WARNING: torch versions differ ({g['torch']} vs {n['torch']}); hashes may differ legitimately")
    _diff(g["experiments"], n["experiments"], "experiments", bad)
    _diff(g["data"], n["data"], "data", bad)
    if bad:
        print(f"GOLDEN COMPARE FAILED: {len(bad)} difference(s)")
        for line in bad[:40]:
            print("  " + line)
        sys.exit(1)
    n_cmp = sum(len(e["seeds"]) for e in g["experiments"].values())
    print(f"GOLDEN COMPARE OK: {len(g['experiments'])} experiments / {n_cmp} (experiment, seed) builds, "
          f"data pipeline for seeds {SEEDS}: identical")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("out")
    c.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    c.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    c.add_argument("--only-from", default=None,
                   help="golden JSON whose experiment names to restrict this capture to "
                        "(use when the registry has grown since the golden file was made)")
    d = sub.add_parser("compare")
    d.add_argument("golden")
    d.add_argument("new")
    a = ap.parse_args()
    if a.cmd == "capture":
        if not a.data_dir:
            raise SystemExit("--data-dir (or DATA_DIR) must point at the extracted cifar100 folder")
        only = set(json.loads(Path(a.only_from).read_text())["experiments"]) if a.only_from else None
        capture(Path(a.root), Path(a.out), a.data_dir, only)
    else:
        compare(Path(a.golden), Path(a.new))
