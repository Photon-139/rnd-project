"""Recompute every number used in the final report and the slides from the per-run JSONs.
Run from VisionTransformer/:  python final_report/scripts/make_data.py   (writes final_report/data/summary.json)"""
import glob, json, os, statistics as st, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "final_report", "data")
SEEDS5, SEEDS3 = [42, 123, 456, 789, 1024], [42, 123, 456]

def run(folder, exp, seed, impl="library"):
    f = os.path.join(ROOT, folder, exp, f"{impl}_seed{seed}.json")
    if not os.path.exists(f):
        return None
    j = json.load(open(f)); m = j["metrics"]; pe = j["per_epoch"]
    return dict(acc=m["test_acc"], val=pe[m["best_epoch"] - 1]["val_loss"], best=m["best_epoch"],
                stop=m["stopping_epoch"], ep_train=m["avg_epoch_train_time_sec"], ep_val=m["avg_epoch_val_time_sec"],
                tot_train=m["total_train_time_sec"] / 3600, tot_val=m["total_val_time_sec"] / 3600,
                fwd=m["avg_batch_forward_sec"] * 1e3, bwd=m["avg_batch_backward_sec"] * 1e3,
                opt=m["avg_batch_optim_sec"] * 1e3, peak=m["peak_memory_gb_train"], static=m["static_memory_gb"],
                lr=j["config"]["lr"], val1=pe[0]["val_loss"])

def ms(xs):
    return [st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0]

def summarise(folder, exp, seeds, impl="library"):
    rs = [run(folder, exp, s, impl) for s in seeds]
    if any(r is None for r in rs):
        return None
    out = {k: ms([r[k] for r in rs]) for k in rs[0]}
    out["acc_seeds"] = [r["acc"] for r in rs]; out["val_seeds"] = [r["val"] for r in rs]
    out["best_seeds"] = [r["best"] for r in rs]; out["stop_seeds"] = [r["stop"] for r in rs]
    out["n"] = len(seeds)
    return out

def paired(a, b, key="acc_seeds"):
    d = [x - y for x, y in zip(a[key], b[key])]
    m, s = ms(d)
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    need = len(d) if len(d) == 3 else 4
    return dict(mean=m, std=s, pos=pos, neg=neg, n=len(d), seeds=d,
                detectable=bool(abs(m) > s and max(pos, neg) >= need))

def sweep(folder, exp):
    f = os.path.join(ROOT, folder, exp, "library_seed42_train.json")
    if not os.path.exists(f): return None
    return {str(r["batch_size"]): (None if r["oom"] else r["peak_gb"]) for r in json.load(open(f))["train"]["sweep"]}

def infer(folder, exp):
    f = os.path.join(ROOT, folder, exp, "library_seed42_inference.json")
    if not os.path.exists(f): return None
    inf = json.load(open(f))["inference"]
    return {v: {b: d["peak_allocated_gb"] for b, d in inf[v]["by_batch"].items()} for v in inf}

S = {}
# ViT-B reference: full FT at 3e-5, its best rate on validation loss (run with --lr; files named baseline_lr1e-5)
V = {"ft": ("results_v2_lr3e-5", "baseline_lr1e-5"), "attn": ("results_v2", "lora_attn_r8"), "mlp": ("results_v2", "lora_mlp_r8"),
     "both": ("results_v2", "lora_attn_mlp_r8"), "first": ("results_v2", "lora_attn_r8_L0-5"),
     "even": ("results_v2", "lora_attn_r8_L0_2_4_6_8_10"), "odd": ("results_v2", "lora_attn_r8_L1_3_5_7_9_11"),
     "second": ("results_v2", "lora_attn_r8_L6-11")}
W = {"ft": "swinb_baseline_lr1e-5", "attn": "swinb_lora_attn_r8", "mlp": "swinb_lora_mlp_r8", "both": "swinb_lora_attn_mlp_r8",
     "first": "swinb_lora_attn_r8_L0-11", "even": "swinb_lora_attn_r8_L0_2_4_6_8_10_12_14_16_18_20_22",
     "odd": "swinb_lora_attn_r8_L1_3_5_7_9_11_13_15_17_19_21_23", "second": "swinb_lora_attn_r8_L12-23"}
S["vitb"] = {k: summarise(f, e, SEEDS5) for k, (f, e) in V.items()}
S["vitb_ft1e-5"] = summarise("results_v2", "baseline_lr1e-5", SEEDS5)
assert abs(S["vitb"]["ft"]["lr"][0] - 3e-5) < 1e-12
S["swinb"] = {k: summarise("results", e, SEEDS5) for k, e in W.items()}
for m in ("vitb", "swinb"):
    S[m + "_vs_ft"] = {k: paired(S[m][k], S[m]["ft"]) for k in S[m] if k != "ft"}
    S[m + "_vs_ft_val"] = {k: paired(S[m][k], S[m]["ft"], "val_seeds") for k in S[m] if k != "ft"}
    S[m + "_vs_all"] = {k: paired(S[m][k], S[m]["attn"]) for k in ("mlp", "both", "first", "even", "odd", "second")}
    S[m + "_vs_all_val"] = {k: paired(S[m][k], S[m]["attn"], "val_seeds") for k in ("first", "even", "odd", "second")}

# full FT learning rates, 5 seeds (ViT-B 1e-4 and its 1e-5 partner: the previous report's runs)
S["ft_lr5"] = {"vitb_1e-4_prev": summarise("results", "baseline", SEEDS5), "vitb_1e-5_prev": summarise("results", "baseline_lr1e-5", SEEDS5),
               "vitb_1e-3_prev": summarise("results", "baseline_lr1e-3", SEEDS5),
               "swinb_1e-4": summarise("results", "swinb_baseline_lr1e-4", SEEDS5), "swinb_1e-5": S["swinb"]["ft"]}
S["ft_lr5"]["vitb_1e-4_minus_1e-5"] = paired(S["ft_lr5"]["vitb_1e-4_prev"], S["ft_lr5"]["vitb_1e-5_prev"])
S["ft_lr5"]["vitb_1e-4_minus_1e-5_val"] = paired(S["ft_lr5"]["vitb_1e-4_prev"], S["ft_lr5"]["vitb_1e-5_prev"], "val_seeds")
S["ft_lr5"]["swinb_1e-4_minus_1e-5"] = paired(S["ft_lr5"]["swinb_1e-4"], S["ft_lr5"]["swinb_1e-5"])
S["ft_lr5"]["swinb_1e-4_minus_1e-5_val"] = paired(S["ft_lr5"]["swinb_1e-4"], S["ft_lr5"]["swinb_1e-5"], "val_seeds")

# learning-rate check, seeds 42/123/456. G1 = ViT-B full FT at 3e-5 (results_v2_lr3e-5, --lr override)
LR = {"vitb": {"ft": {"3e-6": ("results_v2", "baseline_lr3e-6"), "1e-5": ("results_v2", "baseline_lr1e-5"),
                      "3e-5": ("results_v2_lr3e-5", "baseline_lr1e-5"), "1e-4": ("results", "baseline")},
               "lora": {"3e-5": ("results_v2", "lora_attn_r8_lr3e-5"), "1e-4": ("results_v2", "lora_attn_r8"),
                        "3e-4": ("results_v2", "lora_attn_r8_lr3e-4"), "1e-3": ("results_v2", "lora_attn_r8_lr1e-3")}},
      "swinb": {"ft": {r: ("results", f"swinb_baseline_lr{r}") for r in ("3e-6", "1e-5", "3e-5", "1e-4")},
                "lora": {"3e-5": ("results", "swinb_lora_attn_r8_lr3e-5"), "1e-4": ("results", "swinb_lora_attn_r8"),
                         "3e-4": ("results", "swinb_lora_attn_r8_lr3e-4"), "1e-3": ("results", "swinb_lora_attn_r8_lr1e-3")}}}
S["lr3"] = {m: {meth: {r: summarise(f, e, SEEDS3) for r, (f, e) in d.items()} for meth, d in dd.items()} for m, dd in LR.items()}
for m in LR:
    for meth, d in S["lr3"][m].items():
        for r, s in d.items():
            if s is not None:
                want = float(r)
                assert abs(s["lr"][0] - want) < 1e-12, (m, meth, r, s["lr"])
    ft, lo = S["lr3"][m]["ft"], S["lr3"][m]["lora"]
    S["lr3"][m]["diffs"] = {}
    for name, a, b in [(f"ft_{r}_minus_1e-5", ft[r], ft["1e-5"]) for r in ft if r != "1e-5"] + \
                       [(f"lora_{r}_minus_1e-4", lo[r], lo["1e-4"]) for r in lo if r != "1e-4"] + \
                       [(f"lora_{r}_minus_ft_{q}", lo[r], ft[q]) for r in lo for q in ft]:
        if a is not None and b is not None:
            S["lr3"][m]["diffs"][name] = {"acc": paired(a, b), "val": paired(a, b, "val_seeds")}
S["ft_rates5"] = {"vitb_3e-5_minus_1e-5": {"acc": paired(S["vitb"]["ft"], S["vitb_ft1e-5"]),
                                          "val": paired(S["vitb"]["ft"], S["vitb_ft1e-5"], "val_seeds")}}
# attention LoRA rerun (results_v3, 3 seeds) against full FT at 3e-5, same seeds
_r3, _f3 = summarise("results_v3", "lora_attn_r8", SEEDS3), summarise("results_v2_lr3e-5", "baseline_lr1e-5", SEEDS3)
S["rerun_vs_ft"] = {"acc": paired(_r3, _f3), "val": paired(_r3, _f3, "val_seeds"), "rerun": _r3,
                    "rerun_minus_first": paired(_r3, summarise("results_v2", "lora_attn_r8", SEEDS3))}

# rank 16 (ViT-B, previous report's runs, both implementations)
S["rank"] = {}
for t in ("attn", "mlp", "attn_mlp"):
    for impl in ("library", "scratch"):
        a, b = summarise("results", f"lora_{t}_r16", SEEDS5, impl), summarise("results", f"lora_{t}_r8", SEEDS5, impl)
        S["rank"][f"{t}_{impl}"] = {"r8": b, "r16": a, "r16_minus_r8": paired(a, b)}

# target ranking against the learning rate (ViT-B, 3 seeds)
a34 = S["lr3"]["vitb"]["lora"]["3e-4"]
S["target_lr"] = {"attn3e-4_minus_mlp": paired(a34, summarise("results_v2", "lora_mlp_r8", SEEDS3)),
                  "attn3e-4_minus_both": paired(a34, summarise("results_v2", "lora_attn_mlp_r8", SEEDS3))}

# scratch vs library (ViT-B, 7 LoRA configurations)
S["impl"] = {}
for k, (f, e) in V.items():
    if k == "ft": continue
    sc = summarise(f, e, SEEDS5, "scratch")
    S["impl"][k] = {"acc": paired(sc, S["vitb"][k]), "peak_diff": sc["peak"][0] - S["vitb"][k]["peak"][0],
                    "same_stop": sum(x == y for x, y in zip(sc["stop_seeds"], S["vitb"][k]["stop_seeds"])), "scratch": sc}

# memory against batch size (profiler, seed 42) and inference memory
S["sweep"] = {"vitb": {"ft": sweep("results_profile", "baseline"), "attn": sweep("results_v2_profile", "lora_attn_r8"),
                       "mlp": sweep("results_v2_profile", "lora_mlp_r8"), "both": sweep("results_v2_profile", "lora_attn_mlp_r8"),
                       "first": sweep("results_v2_profile", "lora_attn_r8_L0-5"), "even": sweep("results_v2_profile", "lora_attn_r8_L0_2_4_6_8_10"),
                       "odd": sweep("results_v2_profile", "lora_attn_r8_L1_3_5_7_9_11"), "second": sweep("results_v2_profile", "lora_attn_r8_L6-11")},
              "swinb": {"ft": sweep("results_profile", "swinb_baseline_lr1e-4"), **{k: sweep("results_profile", W[k]) for k in W if k != "ft"}}}
# batch sizes 2 and 4 (job 14887, 2026-11-17), merged into the sweeps above; batch 1 and 8 there were re-measured
# by that job and agreed within 0.0003 GiB
BS_EXTRA = {"vitb": {"ft": "baseline", "attn": "lora_attn_r8", "mlp": "lora_mlp_r8", "both": "lora_attn_mlp_r8"},
            "swinb": {"ft": "swinb_baseline_lr1e-4", "attn": "swinb_lora_attn_r8", "mlp": "swinb_lora_mlp_r8",
                      "both": "swinb_lora_attn_mlp_r8"}}
for m, d in BS_EXTRA.items():
    for k, e in d.items():
        extra = sweep("results_mem_profile_bs", e)
        for bs in ("1", "8"):
            assert abs(extra[bs] - S["sweep"][m][k][bs]) < 0.005, (m, k, bs)
        S["sweep"][m][k].update({"2": extra["2"], "4": extra["4"]})
S["infer"] = {"vitb": {"ft": infer("results_profile", "baseline"), **{k: infer(V[k][0] + "_profile", V[k][1]) for k in V if k != "ft"}},
              "swinb": {"ft": infer("results_profile", "swinb_baseline_lr1e-4"), **{k: infer("results_profile", W[k]) for k in W if k != "ft"}}}
S["memsweep"] = {os.path.basename(d): sweep("results_mem_profile", os.path.basename(d))
                 for d in sorted(glob.glob(os.path.join(ROOT, "results_mem_profile", "*")))}
# ViT-B at every first adapted block k (job 14887)
S["memsweep_vitb"] = {os.path.basename(d): sweep("results_mem_profile_vitb", os.path.basename(d))
                      for d in sorted(glob.glob(os.path.join(ROOT, "results_mem_profile_vitb", "*")))}

# memory-budget study training runs (3 seeds)
D = {"deit_all": "deitb_lora_attn_r8", "deit_3": "deitb_lora_attn_r8_L3-11", "deit_6": "deitb_lora_attn_r8_L6-11",
     "vitb_3": "lora_attn_r8_L3-11", "vitb_all": "lora_attn_r8", "vitb_6": "lora_attn_r8_L6-11"}
S["budget"] = {k: summarise("results_v2", e, SEEDS3) for k, e in D.items()}
S["budget"]["vitb_all_rerun"] = summarise("results_v3", "lora_attn_r8", SEEDS3)      # G3, optional
S["budget_diffs"] = {}
for name, a, b in (("deit_3_minus_all", "deit_3", "deit_all"), ("deit_6_minus_all", "deit_6", "deit_all"),
                   ("vitb_3_minus_all", "vitb_3", "vitb_all"), ("vitb_6_minus_all", "vitb_6", "vitb_all"),
                   ("vitb_3_minus_all_rerun", "vitb_3", "vitb_all_rerun"), ("vitb_6_minus_all_rerun", "vitb_6", "vitb_all_rerun")):
    if S["budget"][a] and S["budget"][b]:
        S["budget_diffs"][name] = {"acc": paired(S["budget"][a], S["budget"][b]), "val": paired(S["budget"][a], S["budget"][b], "val_seeds")}

os.makedirs(OUT, exist_ok=True)
json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print("written", os.path.join(OUT, "summary.json"), "| ViT-B full FT reference: lr", S["vitb"]["ft"]["lr"][0])
