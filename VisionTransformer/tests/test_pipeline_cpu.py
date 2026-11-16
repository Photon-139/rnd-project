"""CPU checks for the pipeline changes (no GPU, no CIFAR needed).

    python tests/test_pipeline_cpu.py

1. adapter export/import round trip across BOTH implementations
2. tiny end-to-end train_and_evaluate run (fake data): result schema,
   provenance, adapter export, checkpoint deletion
3. overwrite guard aborts before any training
"""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train_utils  # noqa: E402
from experiments import EXPERIMENTS  # noqa: E402
from models import (build_paired_models, adapter_reference, export_adapter,  # noqa: E402
                    load_adapter_file, apply_adapter_reference, check_forward_equivalence)


def _random_adapter(model, seed=0, b_std=0.02):
    ref = adapter_reference(model)
    g = torch.Generator().manual_seed(seed)
    return {k: (torch.randn(v.shape, generator=g) * b_std if k[-1] == "lora_B" else v.clone())
            for k, v in ref.items()}


def test_adapter_roundtrip():
    cfg = copy.deepcopy(EXPERIMENTS["lora_attn_r8"])
    models = build_paired_models(cfg, seed=42)
    lib, scr = models["library"], models["scratch"]
    ref = _random_adapter(lib)
    apply_adapter_reference(lib, ref)
    apply_adapter_reference(scr, ref)

    with tempfile.TemporaryDirectory() as d:
        f_lib = export_adapter(lib, Path(d) / "lib.pt")
        f_scr = export_adapter(scr, Path(d) / "scr.pt")
        a, b = load_adapter_file(f_lib), load_adapter_file(f_scr)
        assert set(a) == set(b) == set(ref), "key sets differ"
        for k in ref:
            assert torch.equal(a[k], ref[k]) and torch.equal(b[k], ref[k]), k
        size_mb = f_lib.stat().st_size / 1024 ** 2
        assert abs(size_mb - f_scr.stat().st_size / 1024 ** 2) < 0.01
        assert 1.5 < size_mb < 2.5, f"attn r8 adapter should be ~2 MiB, got {size_mb}"

        # loads into a FRESH model of each implementation and reproduces the outputs
        fresh = build_paired_models(cfg, seed=7)  # different head init
        for m in fresh.values():
            apply_adapter_reference(m, load_adapter_file(f_lib))
    rep = check_forward_equivalence(fresh["library"], fresh["scratch"], seed=1,
                                    device=torch.device("cpu"), tol=1e-3)
    assert rep["forward_ok"], rep
    print(f"[ok] adapter round trip ({len(ref)} tensors, {size_mb:.2f} MiB, "
          f"fresh-model forward diff {rep['forward_max_abs_diff']:.2e})")


def _fake_loaders(batch_size=4, seed=0, data_dir=None):
    g = torch.Generator().manual_seed(0)

    def ds(n):
        return TensorDataset(torch.randn(n, 3, 224, 224, generator=g),
                             torch.randint(0, 100, (n,), generator=g))
    train = DataLoader(ds(8), batch_size=4, shuffle=True, generator=torch.Generator())
    return train, DataLoader(ds(4), batch_size=4), DataLoader(ds(4), batch_size=4)


def test_end_to_end_tiny():
    cfg = copy.deepcopy(EXPERIMENTS["lora_attn_r8"])
    cfg.max_epochs = 1
    train_utils.get_loaders = _fake_loaders
    with tempfile.TemporaryDirectory() as d:
        models = build_paired_models(cfg, seed=42)
        for impl in ("library", "scratch"):
            res = train_utils.train_and_evaluate(
                model=models.pop(impl), run_config=cfg, seed=42, results_dir=d,
                checkpoint_dir=str(Path(d) / "ckpt"), use_logits=(impl == "library"),
                impl_name=impl)
            jf = train_utils.result_path(d, cfg.name, impl, 42)
            data = json.loads(jf.read_text())
            m = data["metrics"]
            assert data["provenance"]["result_schema_version"] == 2
            assert len(data["provenance"]["source_sha256"]) == 64
            for k in ("inference_weights_gb", "inference_resident_overhead_gb",
                      "inference_memory_clean", "adapter_export_size_mb"):
                assert k in m, k
            assert 1.5 < m["adapter_export_size_mb"] < 2.5, m["adapter_export_size_mb"]
            adapter = jf.with_name(f"{impl}_seed42_adapter.pt")
            assert adapter.exists()
            ck = list((Path(d) / "ckpt").rglob("*"))
            assert not [p for p in ck if p.is_file() or p.name.endswith("_peft")], ck
            print(f"[ok] tiny run {impl}: adapter {m['adapter_export_size_mb']:.2f} MiB, "
                  f"git_dirty={data['provenance']['git_dirty']}")


def test_overwrite_guard():
    with tempfile.TemporaryDirectory() as d:
        target = train_utils.result_path(d, "lora_attn_r8", "library", 42)
        target.parent.mkdir(parents=True)
        target.write_text("{}")
        out = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), "-e", "lora_attn_r8", "-s", "42",
             "--results-dir", d], capture_output=True, text=True, cwd=ROOT)
        assert out.returncode == 1 and "refusing to overwrite" in out.stdout, out.stdout + out.stderr
        assert "Starting Training" not in out.stdout
        assert target.read_text() == "{}"
    print("[ok] overwrite guard aborts before training")


if __name__ == "__main__":
    test_adapter_roundtrip()
    test_end_to_end_tiny()
    test_overwrite_guard()
    print("ALL OK")
