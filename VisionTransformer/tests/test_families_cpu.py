"""CPU checks for the model-family layer: DeiT-B (paired, same architecture as ViT-B) and
Swin-B (library only: no scratch implementation, hierarchical, ImageNet normalization).

    python tests/test_families_cpu.py

Needs hf_vit/deit_base and hf_vit/swin_base (each test says so and skips if absent). No GPU and no
CIFAR-100. Companion to test_variants_cpu.py (layer lists, ViT-L) and golden_vitb.py (ViT-B unchanged).
"""
import copy
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import aggregate  # noqa: E402
import run as run_mod  # noqa: E402
import train_utils  # noqa: E402
import models as models_mod  # noqa: E402
from experiments import (EXPERIMENTS, ARCH_BASELINE, SWIN_BASE_PATH, DEIT_BASE_PATH, VIT_BASE_PATH,  # noqa: E402
                         LoRAExperimentConfig, IMAGENET_NORM, _layers_tag,
                         arch_dims, dense_param_counts, trainable_counts, num_blocks, model_family,
                         has_scratch_impl, checkpoint_normalization)
from models import (build_paired_models, build_library_model, build_scratch_model, _param_key,  # noqa: E402
                    export_adapter, load_adapter_file, apply_adapter_reference, perturbed_adapter_reference,
                    merged_copy, _swin_lora_targets, trainable_signature, _check_layers)

CPU = torch.device("cpu")
SWIN_EXPS = [n for n, c in EXPERIMENTS.items() if c.arch == "swinb"]
HAVE_SWIN = (Path(SWIN_BASE_PATH) / "config.json").exists()
HAVE_DEIT = (Path(DEIT_BASE_PATH) / "config.json").exists()
DEPTHS = [2, 2, 18, 2]


def _cfg(name, **lora_over):
    cfg = copy.deepcopy(EXPERIMENTS[name])
    for k, v in lora_over.items():
        setattr(cfg.lora, k, v)
    return cfg


def _skip(why):
    print(f"[skip] {why}")


# ───────────────────────────── registry / families ─────────────────────────────

def test_family_registry():
    for key, cfg in EXPERIMENTS.items():
        if not (Path(cfg.model_local_path) / "config.json").exists():
            continue
        fam = model_family(cfg.model_local_path, strict=True)
        assert (fam == "swin") == (cfg.arch == "swinb"), (key, fam, cfg.arch)
        if fam == "swin":
            assert cfg.implementations == ["library"], f"{key}: Swin has no scratch implementation"
            assert not has_scratch_impl(cfg.model_local_path)
        else:
            assert "scratch" in cfg.implementations and has_scratch_impl(cfg.model_local_path), key
    for arch in ("deitb", "swinb"):
        base = EXPERIMENTS[ARCH_BASELINE[arch]]
        assert base.arch == arch and not base.lora and base.lr == 1e-4
    assert not any(EXPERIMENTS[n].arch in ("deitb", "swinb") for n in run_mod.expand_experiment_names(["all"]))
    print(f"[ok] families: Swin entries are library-only, DeiT/ViT paired; '-e all' excludes both new archs")


# ───────────────────────────── geometry / counts ─────────────────────────────

def test_swin_counts_match_hf():
    if not HAVE_SWIN:
        return _skip("no hf_vit/swin_base")
    from transformers import AutoConfig, AutoModelForImageClassification
    dims = arch_dims(SWIN_BASE_PATH, strict=True)
    assert dims["family"] == "swin" and num_blocks(dims) == 24 and dims["hidden_size"] == 1024
    elems, tensors = dense_param_counts(dims, 100)
    with torch.device("meta"):
        hf = AutoModelForImageClassification.from_config(AutoConfig.from_pretrained(SWIN_BASE_PATH, num_labels=100))
    n_hf, t_hf = sum(p.numel() for p in hf.parameters()), len(list(hf.parameters()))
    assert (elems, tensors) == (n_hf, t_hf) == (86_845_724, 425), (elems, tensors, n_hf, t_hf)
    print(f"[ok] Swin-B: analytic dense count = HF meta model = {elems:,} elements / {tensors} tensors")


def test_deit_counts_match_vitb():
    if not HAVE_DEIT:
        return _skip("no hf_vit/deit_base")
    d = arch_dims(DEIT_BASE_PATH, strict=True)
    assert dense_param_counts(d, 100) == (85_875_556, 200) == dense_param_counts(arch_dims(VIT_BASE_PATH, True), 100)
    assert model_family(DEIT_BASE_PATH) == "vit"
    print("[ok] DeiT-B: same geometry and parameter count as ViT-B (it is the same architecture)")


# ───────────────────────────── normalization ─────────────────────────────

def test_normalization_from_checkpoint():
    mean, std, src = checkpoint_normalization(VIT_BASE_PATH, "vitb")
    assert (mean, std, src) == ((0.5,) * 3, (0.5,) * 3, "checkpoint")
    assert train_utils.make_transform(mean, std) is train_utils.transform      # ViT inputs unchanged, same object
    if HAVE_SWIN:
        mean, std, src = checkpoint_normalization(SWIN_BASE_PATH, "swinb")
        assert mean == (0.485, 0.456, 0.406) and std == (0.229, 0.224, 0.225) and src == "checkpoint"
        tf = train_utils.make_transform(mean, std)
        assert tf is not train_utils.transform
        img = Image.fromarray((torch.arange(32 * 32 * 3).reshape(32, 32, 3) % 256).to(torch.uint8).numpy())
        x = tf(img)
        raw = train_utils.transforms.Compose([train_utils.transforms.Resize((224, 224)),
                                              train_utils.transforms.ToTensor()])(img)
        want = (raw - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
        assert torch.allclose(x, want, atol=1e-6)
    if HAVE_DEIT:
        # The HF DeiT file says 0.5 (byte-identical to the ViT one) but the DeiT recipe, and timm's
        # DeiT-B config, use ImageNet statistics: the override must win, and say so.
        file_norm = json.loads((Path(DEIT_BASE_PATH) / "preprocessor_config.json").read_text())["image_mean"]
        mean, std, src = checkpoint_normalization(DEIT_BASE_PATH, "deitb")
        assert (mean, std) == IMAGENET_NORM and src.startswith("override"), (mean, std, src)
        assert file_norm == [0.5] * 3, "the HF file changed: re-check whether the DeiT override is still needed"
        assert train_utils.make_transform(mean, std) is not train_utils.transform
        assert checkpoint_normalization(VIT_BASE_PATH, "vitb")[0] == (0.5,) * 3          # ViT untouched
    with tempfile.TemporaryDirectory() as d:                                   # no preprocessor_config.json
        assert checkpoint_normalization(d, "vitb")[2].startswith("legacy-default")
        try:
            checkpoint_normalization(d, "swinb")
            raise AssertionError("a non-ViT arch must not fall back silently")
        except FileNotFoundError:
            pass
    print("[ok] normalization: read from the checkpoint (ViT 0.5, Swin ImageNet); legacy fallback only for vitb/vitl")


# ───────────────────────────── Swin LoRA structure ─────────────────────────────

def test_swin_key_mapping():
    name = "base_model.model.swin.encoder.layers.2.blocks.5.attention.q_proj.lora_A.default.weight"
    assert _param_key(name, DEPTHS) == (2 + 2 + 5, "q_proj", "lora_A")
    name = "base_model.model.swin.encoder.layers.3.blocks.1.mlp.fc2.lora_B.default.weight"
    assert _param_key(name, DEPTHS) == (23, "fc2", "lora_B")
    try:
        _param_key(name)
        raise AssertionError("a Swin LoRA key without depths must not map")
    except RuntimeError:
        pass
    assert _param_key("base_model.model.vit.encoder.layer.7.attention.attention.query.lora_A.default.weight") \
        == (7, "q_proj", "lora_A")
    print("[ok] Swin LoRA parameter -> global block index (stage offsets), ViT mapping unchanged")


def _lora_layers(model):
    return {_param_key(n, DEPTHS)[0] for n, _ in model.named_parameters() if ".lora_A." in n}


def test_swin_lora_structure():
    if not HAVE_SWIN:
        return _skip("no hf_vit/swin_base")
    cases = [(n, None) for n in ("swinb_lora_attn_r8", "swinb_lora_mlp_r8", "swinb_lora_attn_mlp_r8")]
    cases += [("swinb_lora_attn_r8_L12-23", None),
              ("swinb_lora_attn_r8", [1, 2, 23]),            # last block of stage 0, first of stage 1, last overall
              ("swinb_lora_attn_mlp_r8", [0, 3, 4, 21, 22])]  # touches every stage boundary
    for name, layers in cases:
        cfg = _cfg(name, **({"layers": layers} if layers else {}))
        want = set(range(24)) if cfg.lora.layers is None else set(cfg.lora.layers)
        model, ref = build_library_model(cfg, seed=42)
        assert _lora_layers(model) == want, (name, layers)
        mods = cfg.lora.target_attention + cfg.lora.target_mlp
        n_a = sum(1 for n, _ in model.named_parameters() if ".lora_A." in n)
        assert n_a == len(want) * len(mods), (name, n_a)
        # block widths differ per stage: each adapter must have the width of ITS block
        for n, p in model.named_parameters():
            m = re.search(r"layers\.(\d+)\.blocks\.", n)
            if m and ".lora_A." in n and ".q_proj." in n:
                assert p.shape == (8, 128 * 2 ** int(m.group(1))), (n, tuple(p.shape))
        exp_elems, exp_tensors = trainable_counts(arch_dims(cfg.model_local_path), 100, cfg.to_dict()["lora"])
        sig = trainable_signature(model)
        assert sig["trainable_count"] == exp_tensors, (name, sig["trainable_count"], exp_tensors)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == exp_elems, name
        print(f"[ok] {name}{' layers ' + str(layers) if layers else ''}: adapters on blocks "
              f"{sorted(want)[:4]}{'...' if len(want) > 4 else ''} only; {exp_elems:,} trainable = analytic")
    try:
        _check_layers(_cfg("swinb_lora_attn_r8", layers=[24]))
        raise AssertionError("block 24 must be out of range for 24 Swin blocks")
    except ValueError:
        pass


def test_swin_target_resolution_refuses_wrong_names():
    """A transformers version that names the modules differently must fail loudly, never adapt nothing."""
    if not HAVE_SWIN:
        return _skip("no hf_vit/swin_base")
    model = build_library_model(_cfg("swinb_baseline_lr1e-4"), 1)[0]
    for n, mod in list(model.named_modules()):
        if n.endswith("attention.q_proj") and ".layers.0.blocks.0." in n:
            model.swin.encoder.layers[0].blocks[0].attention.q_proj = torch.nn.Identity()
    try:
        _swin_lora_targets(model, EXPERIMENTS["swinb_lora_attn_r8"].lora, DEPTHS)
        raise AssertionError("renamed/missing module was not detected")
    except RuntimeError as e:
        assert "resolved" in str(e)
    print("[ok] Swin target resolution refuses a model whose modules do not match the expected names")


# ───────────────────────────── library-only gate ─────────────────────────────

def test_no_scratch_for_swin():
    if not HAVE_SWIN:
        return _skip("no hf_vit/swin_base")
    cfg = copy.deepcopy(EXPERIMENTS["swinb_lora_attn_r8"])
    try:
        build_scratch_model(cfg, 1, None)
        raise AssertionError("scratch Swin must not exist")
    except RuntimeError as e:
        assert "no from-scratch implementation" in str(e)
    assert run_mod.resolve_implementations(cfg, "paired", explicit=False) == ["library"]
    for arg in ("paired", "scratch"):
        try:
            run_mod.resolve_implementations(cfg, arg, explicit=True)
            raise AssertionError(f"--impl {arg} must be refused for Swin")
        except SystemExit:
            pass
    print("[ok] Swin: scratch build refused; --impl paired/scratch refused; default resolves to library")


def test_library_only_gate():
    if not HAVE_SWIN:
        return _skip("no hf_vit/swin_base")
    for name in ("swinb_lora_attn_r8_L12-23", "swinb_lora_mlp_r8", "swinb_baseline_lr1e-4"):
        cfg = copy.deepcopy(EXPERIMENTS[name])
        models = build_paired_models(cfg, seed=42)
        assert list(models) == ["library"]
        report = run_mod.run_pre_train_checks(models, 42, CPU, 1e-3, config=cfg)
        lo = report["library_only"]
        assert lo["ok"] and lo["trainable_ok"] and lo["logits_ok"], (name, lo)
        if cfg.lora:
            assert lo["identity_max_abs_diff"] <= 1e-6 and lo["merge_ok"]
            assert lo["merge"]["adapter_effect_max_abs"] > 0, "perturbed adapters changed nothing: vacuous check"
            assert lo["merge"]["adapters_perturbed"] == (12 if "L12" in name else 24) * len(
                cfg.lora.target_attention + cfg.lora.target_mlp)
        run_mod.enforce_checks(report, f"lib-only:{name}")
        summ = run_mod.summarize_verification(report)
        assert summ["library_only_ok"] is True and "trainable_elements" in summ
        print(f"[ok] {name}: library-only gate passes "
              + (f"(merge Δ {lo['merge']['merged_vs_unmerged_max_abs_diff']:.1e}, effect {lo['merge']['adapter_effect_max_abs']:.2f})"
                 if cfg.lora else ""))


def test_library_only_gate_catches_problems():
    if not HAVE_SWIN:
        return _skip("no hf_vit/swin_base")
    cfg = copy.deepcopy(EXPERIMENTS["swinb_lora_attn_r8_L12-23"])
    # (a) something besides head + adapters is trainable
    models = build_paired_models(cfg, seed=1)
    for p in models["library"].base_model.model.swin.layernorm.parameters():
        p.requires_grad_(True)
    report = run_mod.run_pre_train_checks(models, 1, CPU, 1e-3, config=cfg)
    assert report["library_only"]["trainable_ok"] is False
    try:
        run_mod.enforce_checks(report, "t")
        raise AssertionError("gate did not stop a wrong trainable set")
    except SystemExit:
        pass
    # (b) the registry claims different layers than the model has
    models = build_paired_models(cfg, seed=1)
    wrong = _cfg("swinb_lora_attn_r8_L12-23", layers=list(range(0, 12)))
    report = run_mod.run_pre_train_checks(models, 1, CPU, 1e-3, config=wrong)
    assert report["library_only"]["trainable_ok"] is False
    print("[ok] library-only gate rejects an unexpected trainable set and a config/model layer mismatch")


# ───────────────────────────── adapters, merge, PEFT reload ─────────────────────────────

def test_swin_adapter_roundtrip_and_merge():
    if not HAVE_SWIN:
        return _skip("no hf_vit/swin_base")
    cfg = copy.deepcopy(EXPERIMENTS["swinb_lora_attn_r8_L12-23"])
    lib = build_paired_models(cfg, seed=42)["library"]
    ref = perturbed_adapter_reference(lib, seed=3)
    apply_adapter_reference(lib, ref)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "a.pt"
        export_adapter(lib, f)
        back = load_adapter_file(f)
        assert {k[0] for k in back if isinstance(k[0], int)} == set(range(12, 24))
        assert set(back) == set(ref)
        fresh = build_paired_models(cfg, seed=7)["library"]
        apply_adapter_reference(fresh, back)
        from models import adapter_reference
        got = adapter_reference(fresh)
        assert all(torch.equal(got[k], ref[k]) for k in ref)
        # PEFT directory round trip, exactly as run.py's peft_save / peft_load closures do it
        peft_dir = Path(d) / "peft"
        lib.save_pretrained(peft_dir)
        for n, p in fresh.named_parameters():
            if ".lora_B." in n:
                p.data.zero_()
        fresh.load_adapter(str(peft_dir), adapter_name="default")
        got = adapter_reference(fresh)
        assert all(torch.equal(got[k], ref[k]) for k in ref), "PEFT reload did not restore the adapters"
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    lib.eval()
    merged = merged_copy(lib).eval()
    assert not any("lora" in n for n, _ in merged.named_parameters())
    assert sum(p.numel() for p in merged.parameters()) == dense_param_counts(arch_dims(cfg.model_local_path), 100)[0]
    with torch.no_grad():
        assert (lib(x).logits - merged(x).logits).abs().max() < 1e-3
    print("[ok] Swin partial adapters: export/apply round trip, PEFT save/load_adapter, merged copy = dense Swin-B")


# ───────────────────────────── DeiT: paired like ViT-B ─────────────────────────────

def test_deit_paired_gate():
    if not HAVE_DEIT:
        return _skip("no hf_vit/deit_base")
    from test_variants_cpu import _assert_backbone_loaded, _adapted_layers
    n = _assert_backbone_loaded(DEIT_BASE_PATH)
    for name in ("deitb_lora_attn_r8_L6-11", "deitb_baseline_lr1e-4"):
        cfg = copy.deepcopy(EXPERIMENTS[name])
        models = build_paired_models(cfg, seed=42)
        report = run_mod.run_pre_train_checks(models, 42, CPU, 1e-3, config=cfg)
        run_mod.enforce_checks(report, f"deit:{name}")
        assert report["forward"]["forward_ok"]
        if cfg.lora:
            lib_l, scr_l = _adapted_layers(models)
            assert lib_l == scr_l == set(range(6, 12)) and report["forward_lora"]["forward_ok"]
        exp = trainable_counts(arch_dims(cfg.model_local_path), 100, cfg.to_dict()["lora"] if cfg.lora else None)
        for m in models.values():
            assert sum(p.numel() for p in m.parameters() if p.requires_grad) == exp[0], name
        print(f"[ok] {name}: pairing gate passes (fwd {report['forward']['forward_max_abs_diff']:.1e}); "
              f"{n} backbone tensors loaded bit-exactly into the scratch model")


# ───────────────────────────── training path / result JSON ─────────────────────────────

def test_swin_training_path_and_json():
    if not HAVE_SWIN:
        return _skip("no hf_vit/swin_base")
    seen = {}

    def fake_loaders(batch_size=4, seed=0, data_dir=None, norm=None):
        seen["norm"] = norm
        g = torch.Generator().manual_seed(0)

        def ds(n):
            return TensorDataset(torch.randn(n, 3, 224, 224, generator=g), torch.randint(0, 100, (n,), generator=g))
        return (DataLoader(ds(8), batch_size=4, shuffle=True, generator=torch.Generator()),
                DataLoader(ds(4), batch_size=4), DataLoader(ds(4), batch_size=4))

    cfg = copy.deepcopy(EXPERIMENTS["swinb_lora_attn_r8_L12-23"])
    original = train_utils.get_loaders
    train_utils.get_loaders = fake_loaders
    try:
        with tempfile.TemporaryDirectory() as d:
            models = build_paired_models(cfg, seed=42)
            report = run_mod.run_pre_train_checks(models, 42, CPU, 1e-3, config=cfg)
            train_utils.train_and_evaluate(
                model=models.pop("library"), run_config=cfg, seed=42, results_dir=d,
                checkpoint_dir=str(Path(d) / "ckpt"), use_logits=True, impl_name="library", max_epochs=1,
                verification=run_mod.summarize_verification(report))
            path = train_utils.result_path(d, cfg.name, "library", 42)
            data = json.loads(path.read_text())
    finally:
        train_utils.get_loaders = original
    assert seen["norm"][0] == (0.485, 0.456, 0.406), seen
    ts = data["training_setup"]["input_normalization"]
    assert ts["mean"] == [0.485, 0.456, 0.406] and ts["source"] == "checkpoint"
    assert data["config"]["arch"] == "swinb" and data["config"]["lora"]["layers"] == list(range(12, 24))
    assert data["model_dims"]["depths"] == DEPTHS and data["model_dims"]["family"] == "swin"
    assert data["verification"]["library_only_ok"] is True
    # the aggregator accepts a library-only run (CPU memory fields are 0, so patch the one the
    # trainable-count check reads: grads GiB)
    exp_elems = trainable_counts(arch_dims(cfg.model_local_path), 100, cfg.to_dict()["lora"])[0]
    data["metrics"]["static_memory_breakdown_gb"]["grads"] = exp_elems * 4 / aggregate.GiB
    data["_file"] = "x"
    problems = []
    aggregate.derive(data, problems)
    problems = [p for p in problems if "peak" not in p and "truncated run" not in p]   # 1-epoch CPU run
    assert problems == [], problems
    bad = copy.deepcopy(data)
    bad["verification"]["library_only_ok"] = False
    problems = []
    aggregate.derive(bad, problems)
    assert any("library-only verification gate" in p for p in problems)
    print("[ok] Swin end-to-end on CPU: ImageNet normalization used and recorded, dims/arch/layers in JSON, "
          "aggregator accepts the library-only record and flags a failed gate")


def test_scripts_and_cli_library_only():
    if not HAVE_SWIN:
        return _skip("no hf_vit/swin_base")
    name = "swinb_lora_attn_r8_L12-23"
    py = sys.executable
    out = subprocess.run([py, "run.py", "-e", name, "-s", "42", "--verify-only"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stdout[-1500:] + out.stderr[-1500:]
    assert "OK (library-only family" in out.stdout and "built only one implementation" not in out.stdout
    out = subprocess.run([py, "run.py", "-e", name, "--impl", "paired", "--verify-only"], capture_output=True,
                         text=True, cwd=ROOT)
    assert out.returncode != 0 and "Refusing to silently run a partial comparison" in out.stdout + out.stderr
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run([py, "merge_study.py", "--experiment", name, "--cycles", "2", "--n-inputs", "2",
                              "--out", d], capture_output=True, text=True, cwd=ROOT)
        assert out.returncode == 0, out.stdout[-1500:] + out.stderr[-1500:]
        rep = json.loads((Path(d) / name / "seed42.json").read_text())
        c = rep["correctness_fp32"]
        assert "library" in c and "scratch" not in c and "library_vs_scratch_merged" not in c
        assert c["library"]["max_abs_diff_merged_vs_unmerged"] < 1e-3
        assert "scratch" not in rep["cost_fp32"] and "swap_adapter_from_file" in rep["cost_fp32"]
        md = aggregate.merge_section([Path(d)], "swinb")
        assert name in md and "| scratch |" not in md and "| library |" in md
        out = subprocess.run([py, "profile_memory.py", "--experiment", name, "--impl", "library", "--mode",
                              "inference", "--out", d], capture_output=True, text=True, cwd=ROOT)
        assert out.returncode == 0, out.stdout[-1500:] + out.stderr[-1500:]
        out = subprocess.run([py, "profile_memory.py", "--experiment", name, "--impl", "scratch", "--mode",
                              "inference", "--out", d], capture_output=True, text=True, cwd=ROOT)
        assert out.returncode != 0 and "no 'scratch' implementation" in out.stdout + out.stderr
    print("[ok] CLI: library-only verify-only passes, explicit --impl paired refused; merge_study and "
          "profile_memory run on Swin without a scratch model; aggregator renders the merge section")


# ───────────────────────────── placement oracle / fixes from the audit ─────────────────────────────

def _adapted_modules(model):
    """Module paths carrying a LoRA adapter, read from parameter names."""
    return sorted({re.sub(r"^base_model\.model\.", "", n.split(".lora_A.")[0])
                   for n, _ in model.named_parameters() if ".lora_A." in n})


def _hand_written(stage_blocks, projs):
    """Expected module paths written out by hand from (stage, block-in-stage) pairs, independent of
    the global-index arithmetic and of models._SWIN_ROLES."""
    tail = {"q": "attention.q_proj", "k": "attention.k_proj", "v": "attention.v_proj", "o": "attention.o_proj",
            "fc1": "mlp.fc1", "fc2": "mlp.fc2"}
    return sorted(f"swin.encoder.layers.{s}.blocks.{b}.{tail[p]}" for s, b in stage_blocks for p in projs)


def test_swin_placement_against_hand_written_lists():
    if not HAVE_SWIN:
        return _skip("no hf_vit/swin_base")
    # depths 2,2,18,2: global 12..21 = stage 2 blocks 8..17, global 22,23 = stage 3 blocks 0,1
    last_half = [(2, b) for b in range(8, 18)] + [(3, 0), (3, 1)]
    cases = [
        ("swinb_lora_attn_r8_L12-23", None, last_half, ["q", "k", "v"]),
        ("swinb_lora_attn_r8", [1, 2, 23], [(0, 1), (1, 0), (3, 1)], ["q", "k", "v"]),
        ("swinb_lora_mlp_r8", [0, 4, 22], [(0, 0), (2, 0), (3, 0)], ["fc1", "fc2"]),
    ]
    for name, layers, blocks, projs in cases:
        cfg = _cfg(name, **({"layers": layers} if layers else {}))
        model, _ = build_library_model(cfg, seed=1)
        assert _adapted_modules(model) == _hand_written(blocks, projs), name
    # q and v only: the oracle must tell k from v even though the counts are identical
    cfg = _cfg("swinb_lora_attn_r8", target_attention=["q_proj", "v_proj"], layers=[0])
    good = _adapted_modules(build_library_model(cfg, seed=1)[0])
    assert good == _hand_written([(0, 0)], ["q", "v"])
    saved = dict(models_mod._SWIN_ROLES)
    try:
        models_mod._SWIN_ROLES["attention.v_proj"], models_mod._SWIN_ROLES["attention.k_proj"] = "k_proj", "v_proj"
        bad = _adapted_modules(build_library_model(cfg, seed=1)[0])
    finally:
        models_mod._SWIN_ROLES.clear()
        models_mod._SWIN_ROLES.update(saved)
    assert len(bad) == len(good) and bad != good, "a k<->v mix-up in the role table was not detectable"
    print("[ok] Swin adapters sit on exactly the hand-written module lists; a deliberate k<->v swap "
          "(same count) is caught by that comparison")


def test_library_only_requires_config():
    if not HAVE_SWIN:
        return _skip("no hf_vit/swin_base")
    cfg = copy.deepcopy(EXPERIMENTS["swinb_lora_attn_r8_L12-23"])
    models = build_paired_models(cfg, seed=1)
    try:
        run_mod.run_pre_train_checks(models, 1, CPU, 1e-3)
        raise AssertionError("a Swin model without config= must not skip every check")
    except RuntimeError as e:
        assert "needs config=" in str(e)
    print("[ok] run_pre_train_checks refuses a Swin model when called without its config")


def test_layers_are_canonical():
    a = LoRAExperimentConfig(rank=8, alpha=16, target_attention=["q_proj"], layers=[1, 0])
    b = LoRAExperimentConfig(rank=8, alpha=16, target_attention=["q_proj"], layers=[0, 1])
    assert a.layers == b.layers == [0, 1]
    assert _layers_tag([1, 0]) == _layers_tag([0, 1]) == "_L0-1"
    assert _layers_tag([4, 2, 0]) == "_L0_2_4"
    print("[ok] layer lists are canonical: [1, 0] and [0, 1] are one experiment with one name")


def test_merge_study_eval_uses_checkpoint_normalization():
    """merge_study --eval-test must feed the model the normalization it was trained with."""
    if not HAVE_SWIN:
        return _skip("no hf_vit/swin_base")
    import merge_study
    seen = []

    def fake_loaders(batch_size=64, seed=0, data_dir=None, norm=None):
        seen.append(norm)
        g = torch.Generator().manual_seed(0)
        ds = TensorDataset(torch.randn(4, 3, 224, 224, generator=g), torch.randint(0, 100, (4,), generator=g))
        return None, None, DataLoader(ds, batch_size=4)

    original, argv = train_utils.get_loaders, sys.argv
    train_utils.get_loaders = fake_loaders
    try:
        with tempfile.TemporaryDirectory() as d:
            for name, want in (("swinb_lora_attn_r8_L12-23", (IMAGENET_NORM[0], IMAGENET_NORM[1])),
                               ("lora_attn_r8_L6-11", None)):
                seen.clear()
                sys.argv = ["merge_study.py", "--experiment", name, "--cycles", "1", "--n-inputs", "1",
                            "--eval-test", "--eval-max-batches", "1", "--out", d]
                merge_study.main()
                assert seen == [want], (name, seen)
    finally:
        train_utils.get_loaders, sys.argv = original, argv
    print("[ok] merge_study --eval-test: Swin evaluated with ImageNet statistics, ViT through the unchanged legacy call")


if __name__ == "__main__":
    test_family_registry()
    test_swin_counts_match_hf()
    test_deit_counts_match_vitb()
    test_normalization_from_checkpoint()
    test_swin_key_mapping()
    test_swin_lora_structure()
    test_swin_target_resolution_refuses_wrong_names()
    test_no_scratch_for_swin()
    test_library_only_gate()
    test_library_only_gate_catches_problems()
    test_swin_adapter_roundtrip_and_merge()
    test_deit_paired_gate()
    test_swin_training_path_and_json()
    test_scripts_and_cli_library_only()
    test_swin_placement_against_hand_written_lists()
    test_library_only_requires_config()
    test_layers_are_canonical()
    test_merge_study_eval_uses_checkpoint_normalization()
    print("ALL OK")
