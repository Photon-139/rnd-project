"""Compare the memory sweep (results_mem_profile/, job 13063) with the predictions written before it ran
(memory_model/fit_and_predictions_2026-11-10.txt). The constants are refitted here from the same earlier profiles
only (fit_memory.load() does not read results_mem_profile/), so they are the pre-registered ones.
Usage:  python memory_model/compare_sweep.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_memory as F

ROOT = F.ROOT


def measured():
    out = {}
    for f in sorted(ROOT.glob("results_mem_profile/*/library_seed42_train.json")):
        d = json.loads(f.read_text())
        out[d["experiment"]] = {p["batch_size"]: (None if p["oom"] else p["peak_gb"]) for p in d["train"]["sweep"]}
    return out


def config_of(name):
    cfg = F.X.EXPERIMENTS[name]
    l = cfg.lora
    return F.arch_of(name), l.layers, bool(l.target_attention), bool(l.target_mlp)


def main():
    rows = F.load()
    v8 = [r for r in rows if r["arch"] == "vitb" and r["rank"] == 8]
    sw = [r for r in rows if r["arch"] == "swinb"]
    coefs = {"vitb_head": F.fit(v8), "swin_head": F.fit(sw), "joint": F.fit(v8 + sw, head=False)}
    meas = measured()

    for B in (64, 256):
        print(f"\n== batch {B}: measured vs pre-registered peak training memory (GiB); error in %")
        print(f"{'config':28} {'k':>2} {'meas':>7} | " + " | ".join(f"{n:>9} err%" for n in coefs))
        errs = {n: [] for n in coefs}
        for name, sweep in meas.items():
            m = sweep.get(B)
            arch, layers, attn, mlp = config_of(name)
            f = F.features(arch, layers, attn, mlp)
            cells = []
            for n, c in coefs.items():
                p = F.resident_gb(arch, layers, attn, mlp, B) + B * F.predict(c, f, len(c) == 4)
                if m is None:
                    cells.append(f"{p:9.2f}  OOM ")
                else:
                    e = 100 * (p - m) / m
                    errs[n].append((arch, abs(e)))
                    cells.append(f"{p:9.2f} {e:+5.1f}")
            print(f"{name:28} {f['k']:2d} {('OOM' if m is None else f'{m:7.3f}'):>7} | " + " | ".join(cells))
        for n, es in errs.items():
            by = {}
            for arch, e in es:
                by.setdefault(arch, []).append(e)
            print(f"  {n:9}: " + ", ".join(f"{a} max {max(v):.1f}% mean {np.mean(v):.1f}%" for a, v in sorted(by.items())))

    # ViT geometry: the per-block cost of starting later, measured directly on DeiT-B (k = 0..10)
    print("\n== DeiT-B suffixes: drop in peak (batch 64) per block that k moves later")
    ks = sorted((F.features(*config_of(n))["k"], meas[n][64]) for n in meas if n.startswith("deitb"))
    for (k0, m0), (k1, m1) in zip(ks, ks[1:]):
        print(f"  k {k0:2d} -> {k1:2d}: {(m0 - m1) / (k1 - k0):.4f} GiB per block")


if __name__ == "__main__":
    main()
