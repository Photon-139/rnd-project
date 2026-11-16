"""Exploratory memory model for LoRA placement (analysis only; reads profiler JSONs, writes nothing in the pipeline).

Model, per configuration, for the peak training memory at batch size B (fp32, GiB):

    peak(B) = resident(B) + e + B * a
    a       = c_reach * S_reach + c_attn * S_attn + c_mlp * S_mlp + c_head

    s_b      size of block b's input per image (tokens_b * width_b * 4 bytes), from config.json
    S_reach  sum of s_b over blocks b >= k  (k = first adapted block: backward must run through these)
    S_attn   sum of s_b over blocks with q/k/v adapters  (the adapters' A needs the LN output stored)
    S_mlp    sum of s_b over blocks with fc1/fc2 adapters
    resident parameters + grads/AdamW of trainable params + input batch (exact formula, from the JSON)

The c's are dimensionless (bytes stored per byte of block input), so a fit on one architecture can be
tried on another. Usage:  python memory_model/fit_memory.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import experiments as X  # read-only use of the registry (layer lists, targets)

GIB = 1024 ** 3
MODEL_DIR = {"vitb": "vit_base", "deitb": "deit_base", "vitl": "vit_large", "swinb": "swin_base"}


def block_geometry(arch):
    """List of (tokens, width, stage) per block, from the checkpoint's config.json."""
    c = json.loads((ROOT / "hf_vit" / MODEL_DIR[arch] / "config.json").read_text())
    if c["model_type"] == "vit":
        tokens = (c["image_size"] // c["patch_size"]) ** 2 + 1
        return [(tokens, c["hidden_size"], 0)] * c["num_hidden_layers"]
    side = c["image_size"] // c["patch_size"]
    out = []
    for stage, depth in enumerate(c["depths"]):
        out += [((side >> stage) ** 2, c["embed_dim"] << stage, stage)] * depth
    return out


def features(arch, layers, attn, mlp):
    geo = block_geometry(arch)
    s = np.array([t * w * 4 for t, w, _ in geo]) / GIB          # GiB per image
    blocks = list(range(len(geo))) if layers is None else sorted(layers)
    k = blocks[0]
    return dict(k=k, n=len(blocks), depth=len(geo),
                S_reach=s[k:].sum(), S_attn=s[blocks].sum() if attn else 0.0,
                S_mlp=s[blocks].sum() if mlp else 0.0)


def arch_of(name):
    return getattr(X.EXPERIMENTS[name], "arch", None) or "vitb"


def load():
    rows = []
    files = sorted(ROOT.glob("results_v2_profile/*/library_seed42_train.json")) + \
        sorted(ROOT.glob("results_profile/*/library_seed42_train.json"))
    for f in files:
        d = json.loads(f.read_text())
        name = d["experiment"]
        cfg = X.EXPERIMENTS[name]
        lora = getattr(cfg, "lora", None)
        if lora is None or "_lr" in name:                       # full FT and lr duplicates: skip
            continue
        arch = arch_of(name)
        code = "v2" if "results_v2_profile" in str(f) else "v1"
        if arch == "vitb" and code == "v1" and lora.rank == 8:  # same shapes rerun in v2; keep v2
            continue
        sw = [p for p in d["train"]["sweep"] if not p["oom"] and p["batch_size"] >= 8]
        B = np.array([p["batch_size"] for p in sw], float)
        above = np.array([p["peak_gb"] - p["resident_before_fwd_gb"] for p in sw])
        a, e = np.polyfit(B, above, 1)
        resid_lin = above - (a * B + e)
        r = dict(name=name, arch=arch, code=code, rank=lora.rank,
                 attn=bool(lora.target_attention), mlp=bool(lora.target_mlp),
                 a=a, e=e, lin_max_err=np.abs(resid_lin).max(),
                 sweep={int(p["batch_size"]): (p["peak_gb"], p["resident_analytic_gb"] + p["runtime_overhead_gb"],
                                               p["resident_before_fwd_gb"]) for p in sw})
        r.update(features(arch, lora.layers, r["attn"], r["mlp"]))
        rows.append(r)
    return rows


COLS = ["S_reach", "S_attn", "S_mlp"]


def design(rows, head=True):
    X_ = np.array([[r[c] for c in COLS] + ([1.0] if head else []) for r in rows])
    return X_, np.array([r["a"] for r in rows])


def fit(rows, head=True):
    A, y = design(rows, head)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def predict(coef, r, head=True):
    x = [r[c] for c in COLS] + ([1.0] if head else [])
    return float(np.dot(coef[:len(x)], x)) if head else float(np.dot(coef[:3], x))


def peak64(r, a):
    peak, res_analytic, res_meas = r["sweep"][64]
    return res_meas + r["e"] + 64 * a


def short(name):
    return (name.replace("swinb_", "S:").replace("lora_", "").replace("_r8", "").replace("_r16", "/r16")
            .replace("L0_2_4_6_8_10_12_14_16_18_20_22", "even").replace("L1_3_5_7_9_11_13_15_17_19_21_23", "odd")
            .replace("L0_2_4_6_8_10", "even").replace("L1_3_5_7_9_11", "odd"))


def show(title, rows, coef_by_arch, head=True):
    print(f"\n== {title}")
    print(f"{'config':24} {'k':>2} {'n':>2} {'meas b64':>9} {'pred b64':>9} {'err GiB':>8} {'err %':>6}")
    errs = []
    for r in rows:
        coef = coef_by_arch(r)
        a_hat = predict(coef, r, head)
        m, p = r["sweep"][64][0], peak64(r, a_hat)
        errs.append(abs(p - m) / m)
        print(f"{short(r['name']):24} {r['k']:2d} {r['n']:2d} {m:9.3f} {p:9.3f} {p - m:+8.3f} {100 * (p - m) / m:+6.1f}")
    print(f"max |err| {100 * max(errs):.1f}%, mean {100 * np.mean(errs):.1f}%")


def main():
    rows = load()
    v8 = [r for r in rows if r["arch"] == "vitb" and r["rank"] == 8]
    sw = [r for r in rows if r["arch"] == "swinb"]
    r16 = [r for r in rows if r["rank"] == 16]

    print("Per-image activation slope a (GiB/img) from the batch sweep (B = 8..256), and its linearity:")
    for r in v8 + sw + r16:
        print(f"  {short(r['name']):24} a = {r['a']:.5f}  e = {r['e']:+.3f}  max dev from line {r['lin_max_err']:.3f} GiB")

    for label, data in [("ViT-B", v8), ("Swin-B", sw)]:
        c = fit(data)
        print(f"\n{label} own fit: c_reach {c[0]:.2f}  c_attn {c[1]:.2f}  c_mlp {c[2]:.2f}  head {c[3] * 64:.3f} GiB @b64")
    cj = fit(v8 + sw, head=False)
    print(f"\nJoint fit, no head term: c_reach {cj[0]:.2f}  c_attn {cj[1]:.2f}  c_mlp {cj[2]:.2f}")

    # 1. leave one configuration out, within each model
    for label, data in [("ViT-B", v8), ("Swin-B", sw)]:
        show(f"{label}: leave-one-out (fit on the other configs of the same model)",
             data, lambda r, d=data: fit([q for q in d if q is not r]))

    # 2. transfer across architectures: constants from one model, no head term (it is not transferable)
    cv, cs = fit(v8, head=False), fit(sw, head=False)
    print(f"\nViT-B constants, no head term: {np.round(cv, 2)};  Swin-B: {np.round(cs, 2)}")
    show("Swin-B predicted from ViT-B constants (no Swin data used)", sw, lambda r: cv, head=False)
    show("ViT-B predicted from Swin-B constants (no ViT data used)", v8, lambda r: cs, head=False)

    # 3. rank 16 (old code, ViT-B) predicted by the rank-8 model: rank only enters through the static term
    cvh = fit(v8)
    show("ViT-B rank 16 (old code) predicted from the rank-8 fit", r16, lambda r: cvh)

    # 4. pre-registered predictions for configurations not yet profiled
    preregister({"vitb_head": cvh, "swin_head": fit(sw), "joint": cj})


def resident_gb(arch, layers, attn, mlp, B, rank=8):
    """Parameters (fp32) + grads and AdamW moments of trainable params + the input batch: the exact formula the
    profiler's resident_analytic_gb uses, plus its measured 0.0175 GiB runtime overhead."""
    dims = X.arch_dims(str(ROOT / "hf_vit" / MODEL_DIR[arch]), strict=True)
    dense, _ = X.dense_param_counts(dims, 100)
    lora = dict(rank=rank, target_attention=["q_proj", "k_proj", "v_proj"] if attn else [],
                target_mlp=["fc1", "fc2"] if mlp else [], layers=layers)
    trainable, _ = X.trainable_counts(dims, 100, lora)
    head = dense - X.dense_param_counts(dims, 0)[0]
    total = dense + (trainable - head)
    img = dims.get("image_size", 224) ** 2 * 3 * 4
    return (4 * total + 12 * trainable + B * img) / GIB + 0.0175


PLANNED = [  # (label, arch, layers, attn, mlp)
    ("DeiT-B attn all", "deitb", None, 1, 0),
    ("DeiT-B attn L6-11", "deitb", list(range(6, 12)), 1, 0),
] + [(f"DeiT-B attn L{k}-11", "deitb", list(range(k, 12)), 1, 0) for k in (1, 2, 3, 4, 5, 8, 10)] + [
    ("ViT-L attn all", "vitl", None, 1, 0),
    ("ViT-L attn L12-23", "vitl", list(range(12, 24)), 1, 0),
    ("ViT-L mlp all", "vitl", None, 0, 1),
    ("ViT-L attn+mlp all", "vitl", None, 1, 1),
] + [(f"Swin-B attn L{k}-23", "swinb", list(range(k, 24)), 1, 0) for k in (2, 4, 8, 16, 22)]


def preregister(coefs):
    print("\n== Pre-registered peak training memory (GiB), not yet measured. Three constant sets:")
    print("   vitb_head = ViT-B fit with head term; swin_head = Swin-B fit with head term;"
          " joint = both models, dimensionless only (no head term)")
    print(f"{'config':24} {'k':>2} {'n':>2} {'resident':>8} | " + " | ".join(f"{n:>9} b64 {n[:4]:>4} b256" for n in coefs))
    for lab, arch, layers, attn, mlp in PLANNED:
        f = features(arch, layers, attn, mlp)
        cells = []
        for name, c in coefs.items():
            head = len(c) == 4
            a_hat = predict(c, f, head)
            cells.append(f"{resident_gb(arch, layers, attn, mlp, 64) + 64 * a_hat:13.2f} "
                         f"{resident_gb(arch, layers, attn, mlp, 256) + 256 * a_hat:9.2f}")
        print(f"{lab:24} {f['k']:2d} {f['n']:2d} {resident_gb(arch, layers, attn, mlp, 64):8.3f} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
