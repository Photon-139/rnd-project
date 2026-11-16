"""CPU checks for explicit LoRA layer lists and other ViT sizes (no GPU, no CIFAR needed).

    python tests/test_variants_cpu.py            # ViT-B only, ~2-4 min
    python tests/test_variants_cpu.py --heavy    # also builds ViT-L / ViT-H (several GB RAM, slow)

Companion to tests/golden_vitb.py, which proves the original ViT-B pipeline is unchanged.
This file proves the NEW paths: layer lists (both implementations agree on which blocks
carry adapters, the pairing gate fires when they do not), geometry read from checkpoints,
analytic parameter counts, merge/unmerge with partial adapters, result-JSON fields.
"""
import copy
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train_utils  # noqa: E402
import run as run_mod  # noqa: E402
from experiments import (EXPERIMENTS, ARCH_BASELINE, LoRAExperimentConfig, arch_dims,  # noqa: E402
                         dense_param_counts, trainable_counts, validate_layers, num_blocks, _layers_tag)
from lora import LoRALinear  # noqa: E402
from models import (build_paired_models, build_library_model, build_scratch_model,  # noqa: E402
                    vit_config_from_checkpoint, trainable_signature, check_forward_equivalence,
                    perturbed_adapter_reference, apply_adapter_reference, adapter_reference,
                    export_adapter, load_adapter_file, merge_all, unmerge_all,
                    _param_key, _scratch_module_key, from_pretrained_checked)
from vision_transformer import ViTConfig, ViTForImageClassification as MyViT, load_vit_weights  # noqa: E402

CPU = torch.device("cpu")
PATHS = {"vitb": "hf_vit/vit_base", "vitl": "hf_vit/vit_large", "vith": "hf_vit/vit_huge"}


def _cfg(name, **lora_over):
    cfg = copy.deepcopy(EXPERIMENTS[name])
    for k, v in lora_over.items():
        setattr(cfg.lora, k, v)
    return cfg


# ───────────────────────────── registry ─────────────────────────────

def test_registry():
    for key, cfg in EXPERIMENTS.items():
        assert cfg.name == key, key
        json.dumps(cfg.to_dict())
        assert cfg.arch in ARCH_BASELINE, (key, cfg.arch)
        depth = num_blocks(arch_dims(cfg.model_local_path, strict=True))
        if cfg.lora:
            validate_layers(cfg.lora.layers, depth)
            if cfg.lora.layers is not None:
                assert key.endswith(_layers_tag(cfg.lora.layers)), f"{key} name does not encode its layers"
        else:
            assert "_L" not in key.split("baseline")[-1], key
    for arch, base in ARCH_BASELINE.items():
        assert base in EXPERIMENTS and EXPERIMENTS[base].arch == arch and not EXPERIMENTS[base].lora, base
    # original ViT-B entries: no layer restriction, default arch
    for name in ("lora_attn_r8", "lora_mlp_r16", "lora_attn_mlp_r8"):
        assert EXPERIMENTS[name].lora.layers is None and EXPERIMENTS[name].arch == "vitb"
    print(f"[ok] registry: {len(EXPERIMENTS)} experiments, names/layers/archs consistent")


def test_layers_validation():
    for bad in ([], [-1], [1, 1], [True], ["a"], 3):
        try:
            LoRAExperimentConfig(rank=8, alpha=16, target_attention=["q_proj"], layers=bad)
        except ValueError:
            continue
        raise AssertionError(f"layers={bad!r} should have been rejected")
    ok = LoRAExperimentConfig(rank=8, alpha=16, target_attention=["q_proj"], layers=(3, 4))
    assert ok.layers == [3, 4]
    try:
        validate_layers([12], 12)
        raise AssertionError("layer 12 must be out of range for 12 blocks")
    except ValueError:
        pass
    cfg = _cfg("lora_attn_r8", layers=[11, 12])          # out of range for ViT-B: must fail before any model is built
    for builder in (lambda: build_library_model(cfg, 1), lambda: build_scratch_model(cfg, 1, None)):
        try:
            builder()
            raise AssertionError("out-of-range layer was accepted")
        except ValueError:
            pass
    print("[ok] layer list validation (config-level and model-depth-level)")


# ─────────────────────── geometry and parameter counts ───────────────────────

def test_dims_and_counts():
    from transformers import ViTConfig as HFViTConfig, ViTForImageClassification as HFViT
    # the scratch config derived from ViT-B's config.json must equal the old hardcoded default
    assert vars(vit_config_from_checkpoint(str(ROOT / PATHS["vitb"]), 100)) == vars(ViTConfig(num_labels=100))
    for arch, rel in PATHS.items():
        path = str(ROOT / rel)
        dims = arch_dims(path, strict=True)
        elems, tensors = dense_param_counts(dims, 100)
        with torch.device("meta"):
            mine = MyViT(vit_config_from_checkpoint(path, 100))
            hf = HFViT(HFViTConfig.from_pretrained(path, num_labels=100))
        n_mine = sum(p.numel() for p in mine.parameters())
        n_hf = sum(p.numel() for p in hf.parameters())
        t_mine = len(list(mine.parameters()))
        assert elems == n_mine == n_hf, (arch, elems, n_mine, n_hf)
        assert tensors == t_mine == len(list(hf.parameters())), (arch, tensors, t_mine)
        if arch == "vitb":
            assert elems == 85_875_556 and tensors == 200
        print(f"[ok] {arch}: analytic = scratch = HF parameter count = {elems:,} ({tensors} tensors)")


def test_norm_stats():
    norm = train_utils.transform.transforms[-1]
    for arch, rel in PATHS.items():
        pre = json.loads((ROOT / rel / "preprocessor_config.json").read_text())
        assert pre["image_mean"] == [0.5] * 3 and pre["image_std"] == [0.5] * 3, (arch, pre)
        assert list(norm.mean) == [0.5] * 3 and list(norm.std) == [0.5] * 3
        assert pre["size"] == 224 and arch_dims(str(ROOT / rel))["image_size"] == 224
    print("[ok] all registered checkpoints use mean=std=0.5 / 224px, matching train_utils.transform")


# ───────────────────────── layer-subset structure ─────────────────────────

def _adapted_layers(models):
    lib, scr = models["library"], models["scratch"]
    lib_layers = {_param_key(n)[0] for n, p in lib.named_parameters()
                  if ".lora_A." in n or ".lora_B." in n}
    scr_layers = {_scratch_module_key(n)[0] for n, m in scr.named_modules() if isinstance(m, LoRALinear)}
    return lib_layers, scr_layers


def _trainable_elems(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_layer_subset_structure():
    for name in ("lora_attn_r8_L0-5", "lora_attn_r8_L6-11", "lora_attn_r8_L0_2_4_6_8_10",
                 "lora_attn_r8_L1_3_5_7_9_11", "lora_qv_r8"):
        cfg = copy.deepcopy(EXPERIMENTS[name])
        want = set(cfg.lora.layers) if cfg.lora.layers is not None else set(range(12))
        models = build_paired_models(cfg, seed=42)
        lib_layers, scr_layers = _adapted_layers(models)
        assert lib_layers == scr_layers == want, (name, lib_layers, scr_layers, want)
        dims = arch_dims(cfg.model_local_path)
        exp_elems, exp_tensors = trainable_counts(dims, 100, cfg.to_dict()["lora"])
        for impl, m in models.items():
            sig = trainable_signature(m)
            # 'trainable_count' counts tensors; PEFT's frozen head duplicate is NOT trainable
            assert sig["trainable_count"] == exp_tensors, (name, impl, sig["trainable_count"], exp_tensors)
            assert _trainable_elems(m) == exp_elems, (name, impl, _trainable_elems(m), exp_elems)
        report = run_mod.run_pre_train_checks(models, 42, CPU, 1e-3)
        assert report["ok"] and report["param_check"]["ok"], name
        assert report["forward"]["forward_ok"] and report["forward_lora"]["forward_ok"], name
        assert report["forward_lora"]["lora_adapters_exercised"] == len(want) * len(
            cfg.lora.target_attention + cfg.lora.target_mlp), name
        print(f"[ok] {name}: layers {sorted(want)} identical in both impls; "
              f"{exp_elems:,} trainable elements / {exp_tensors} tensors = analytic; gate passes "
              f"(fwd {report['forward']['forward_max_abs_diff']:.1e}, "
              f"fwd+LoRA {report['forward_lora']['forward_max_abs_diff']:.1e})")


def test_gate_catches_layer_mismatch():
    subset, full = _cfg("lora_attn_r8", layers=[3, 4]), _cfg("lora_attn_r8")
    _, ref_subset = build_library_model(subset, 7)
    _, ref_full = build_library_model(full, 7)
    for ref, scratch_cfg, what in ((ref_subset, full, "scratch has MORE adapted layers than the reference"),
                                   (ref_full, subset, "scratch has FEWER adapted layers than the reference")):
        try:
            build_scratch_model(scratch_cfg, 7, ref)
        except RuntimeError as e:
            assert "FATAL" in str(e), e
        else:
            raise AssertionError(f"mismatch not detected: {what}")
    print("[ok] pairing gate raises when the two implementations disagree on adapted layers (both directions)")


# ───────────────────────── merge / adapters with partial layers ─────────────────────────

def _layer_of_state_key(k):
    m = re.search(r"layer[s]?\.(\d+)\.", k)
    return int(m.group(1)) if m else None


def test_merge_partial_layers():
    cfg = copy.deepcopy(EXPERIMENTS["lora_attn_r8_L6-11"])
    adapted = set(cfg.lora.layers)
    models = build_paired_models(cfg, seed=42)
    ref = perturbed_adapter_reference(models["library"], seed=1, b_std=0.02)
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(5))
    outs = {}
    for impl, m in models.items():
        apply_adapter_reference(m, ref)
        m.eval()
        before = {k: v.detach().clone() for k, v in m.state_dict().items()}
        with torch.no_grad():
            o0 = m(x)
            o0 = o0.logits if hasattr(o0, "logits") else o0
            merge_all(m)
            o1 = m(x)
            o1 = o1.logits if hasattr(o1, "logits") else o1
            mid = {k: v.detach().clone() for k, v in m.state_dict().items()}
            unmerge_all(m)
            o2 = m(x)
            o2 = o2.logits if hasattr(o2, "logits") else o2
        assert (o0 - o1).abs().max() < 1e-4, (impl, (o0 - o1).abs().max())
        assert (o0 - o2).abs().max() < 1e-4, impl
        # blocks WITHOUT adapters must be bit-identical after merging
        untouched = [k for k in before if (_layer_of_state_key(k) is not None
                                           and _layer_of_state_key(k) not in adapted
                                           and "lora" not in k)]
        assert untouched and all(torch.equal(before[k], mid[k]) for k in untouched), impl
        changed = [k for k in before if _layer_of_state_key(k) in adapted and "lora" not in k
                   and not torch.equal(before[k], mid[k])]
        assert changed, f"{impl}: merge changed nothing in the adapted blocks"
        outs[impl] = o1
    assert (outs["library"] - outs["scratch"]).abs().max() < 1e-3
    print("[ok] merge/unmerge with adapters on blocks 6-11: output preserved, other blocks untouched, "
          "merged library == merged scratch")


def test_adapter_roundtrip_partial():
    cfg = copy.deepcopy(EXPERIMENTS["lora_attn_r8_L6-11"])
    models = build_paired_models(cfg, seed=42)
    ref = perturbed_adapter_reference(models["library"], seed=3, b_std=0.02)
    assert {k[0] for k in ref if isinstance(k[0], int)} == set(cfg.lora.layers)
    for m in models.values():
        apply_adapter_reference(m, ref)
    with tempfile.TemporaryDirectory() as d:
        f_lib = export_adapter(models["library"], Path(d) / "lib.pt")
        f_scr = export_adapter(models["scratch"], Path(d) / "scr.pt")
        a, b = load_adapter_file(f_lib), load_adapter_file(f_scr)
        assert set(a) == set(b) == set(ref)
        assert all(torch.equal(a[k], ref[k]) and torch.equal(b[k], ref[k]) for k in ref)
        fresh = build_paired_models(cfg, seed=9)
        for m in fresh.values():
            apply_adapter_reference(m, a)
        rep = check_forward_equivalence(fresh["library"], fresh["scratch"], seed=2, device=CPU, tol=1e-3)
        assert rep["forward_ok"], rep
        exp_elems, _ = trainable_counts(arch_dims(cfg.model_local_path), 100, cfg.to_dict()["lora"])
        size = f_lib.stat().st_size
        assert 4 * exp_elems <= size <= 4 * exp_elems + 64 * 1024, (size, 4 * exp_elems)
    print("[ok] adapter export/import with a block subset: keys, values, size = 4 B x trainable elements")


# ───────────────────────── result JSON ─────────────────────────

def _fake_loaders(batch_size=4, seed=0, data_dir=None, norm=None):
    g = torch.Generator().manual_seed(0)

    def ds(n):
        return TensorDataset(torch.randn(n, 3, 224, 224, generator=g), torch.randint(0, 100, (n,), generator=g))
    return (DataLoader(ds(8), batch_size=4, shuffle=True, generator=torch.Generator()),
            DataLoader(ds(4), batch_size=4), DataLoader(ds(4), batch_size=4))


def test_result_json_fields():
    cfg = copy.deepcopy(EXPERIMENTS["lora_attn_r8_L6-11"])
    original = train_utils.get_loaders
    train_utils.get_loaders = _fake_loaders
    try:
        with tempfile.TemporaryDirectory() as d:
            models = build_paired_models(cfg, seed=42)
            exp_elems, exp_tensors = trainable_counts(arch_dims(cfg.model_local_path), 100,
                                                      cfg.to_dict()["lora"])
            train_utils.train_and_evaluate(
                model=models.pop("library"), run_config=cfg, seed=42, results_dir=d,
                checkpoint_dir=str(Path(d) / "ckpt"), use_logits=True, impl_name="library",
                max_epochs=1)
            data = json.loads(train_utils.result_path(d, cfg.name, "library", 42).read_text())
            assert data["config"]["lora"]["layers"] == list(range(6, 12))
            assert data["config"]["max_epochs"] == 1 == data["training_setup"]["max_epochs"]
            assert data["model_dims"]["hidden_size"] == 768 and data["model_dims"]["num_hidden_layers"] == 12
            # adapter file on disk contains only blocks 6-11
            ad = train_utils.result_path(d, cfg.name, "library", 42).with_name("library_seed42_adapter.pt")
            keys = load_adapter_file(ad)
            assert {k[0] for k in keys if isinstance(k[0], int)} == set(range(6, 12))
            assert abs(data["metrics"]["adapter_export_size_mb"] - 4 * exp_elems / 1024 ** 2) < 0.1
    finally:
        train_utils.get_loaders = original
    print("[ok] result JSON records layers, effective max_epochs and model geometry")


def test_cli_list_and_all():
    out = subprocess.run([sys.executable, str(ROOT / "run.py"), "--list"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    for name in ("lora_attn_r8_L6-11", "vitl_lora_attn_r8"):
        assert name in out.stdout, name
    names = run_mod.expand_experiment_names(["all"])
    assert names and all(EXPERIMENTS[n].arch == "vitb" for n in names), names
    assert "baseline" in names and "lora_attn_r8_L6-11" in names and not any(n.startswith("vitl") for n in names)
    assert run_mod.expand_experiment_names(["vitl_lora_attn_r8"]) == ["vitl_lora_attn_r8"]
    print(f"[ok] run.py --list shows the new experiments; '-e all' = {len(names)} ViT-B entries, no ViT-L")


def test_aggregate_arch_and_missing():
    """aggregate.py: empty architecture is loud; a missing ORIGINAL ViT-B experiment is reported."""
    agg = str(ROOT / "aggregate.py")
    src = ROOT / "results"
    if not (src / "baseline").exists():
        print("[skip] aggregate checks need the existing results/ tree")
        return
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        out = subprocess.run([sys.executable, agg, "--arch", "vitl", "--results", str(src),
                              "--profile", str(d / "nope"), "--merge", str(d / "nope"), "--out", str(d / "o1")],
                             capture_output=True, text=True, cwd=ROOT)
        assert out.returncode == 0 and "WARNING: no runs found for --arch vitl" in out.stdout, out.stdout + out.stderr
        import shutil
        (d / "partial").mkdir()
        shutil.copytree(src / "baseline", d / "partial" / "baseline")
        out = subprocess.run([sys.executable, agg, "--results", str(d / "partial"), "--profile", str(d / "nope"),
                              "--merge", str(d / "nope"), "--out", str(d / "o2")],
                             capture_output=True, text=True, cwd=ROOT)
        assert out.returncode == 0, out.stderr
        report = (d / "o2" / "report.md").read_text()
        assert "lora_mlp_r16: no runs found" in report and "baseline: no runs found" not in report
        assert "lora_qv_r8: no runs found" not in report      # new, never-run entries stay silent by default
    print("[ok] aggregate.py: empty --arch warns; missing original ViT-B experiments are still reported")


# ───────────────────────── heavy: ViT-L / ViT-H ─────────────────────────

def _assert_backbone_loaded(path):
    from transformers import ViTForImageClassification as HFViT
    hf = HFViT.from_pretrained(path, local_files_only=True).state_dict()
    scr = MyViT(vit_config_from_checkpoint(path, 100))
    load_vit_weights(scr, hf)
    n = 0
    for k, v in scr.state_dict().items():
        if k.startswith("classifier"):
            continue
        assert k in hf and hf[k].shape == v.shape and torch.equal(hf[k], v), f"not loaded exactly: {k}"
        n += 1
    return n


def test_guard_rejects_partial_load():
    """The ViT-H directory is a bare ViTModel whose keys HF cannot map: the loader must refuse it."""
    for path, should_fail in ((PATHS["vitb"], False), (PATHS["vith"], True)):
        try:
            from_pretrained_checked(str(ROOT / path), num_labels=100, ignore_mismatched_sizes=True)
            failed = False
        except RuntimeError as e:
            assert "did not load completely" in str(e), e
            failed = True
        assert failed == should_fail, (path, failed)
    print("[ok] from_pretrained_checked accepts ViT-B and rejects the ViT-H directory (backbone would be random)")


def test_heavy_variants():
    for arch, name_list in (("vitl", ["vitl_lora_attn_r8_L12-23", "vitl_baseline_lr1e-4"]),):
        n = _assert_backbone_loaded(str(ROOT / PATHS[arch]))
        print(f"[ok] {arch}: {n} backbone tensors loaded bit-exactly into the scratch model")
        for name in name_list:
            cfg = copy.deepcopy(EXPERIMENTS[name])
            models = build_paired_models(cfg, seed=42)
            exp_elems, exp_tensors = trainable_counts(arch_dims(cfg.model_local_path), 100,
                                                      cfg.to_dict()["lora"] if cfg.lora else None)
            for impl, m in models.items():
                assert _trainable_elems(m) == exp_elems, (name, impl, _trainable_elems(m), exp_elems)
            if cfg.lora and cfg.lora.layers is not None:
                lib_layers, scr_layers = _adapted_layers(models)
                assert lib_layers == scr_layers == set(cfg.lora.layers), name
            report = run_mod.run_pre_train_checks(models, 42, CPU, 1e-3)
            run_mod.enforce_checks(report, f"heavy:{name}")
            print(f"[ok] {name}: gate passes, {exp_elems:,} trainable elements = analytic")
            del models


if __name__ == "__main__":
    test_registry()
    test_layers_validation()
    test_dims_and_counts()
    test_norm_stats()
    test_cli_list_and_all()
    test_guard_rejects_partial_load()
    test_layer_subset_structure()
    test_gate_catches_layer_mismatch()
    test_merge_partial_layers()
    test_adapter_roundtrip_partial()
    test_result_json_fields()
    test_aggregate_arch_and_missing()
    if "--heavy" in sys.argv:
        test_heavy_variants()
    print("ALL OK")
