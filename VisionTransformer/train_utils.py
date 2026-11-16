"""
Shared training loop, data loading, metrics collection, and environment logging.
"""

import os
import sys
import gc
import json
import time
import shutil
import socket
import hashlib
import platform
import datetime
import subprocess
from pathlib import Path
from typing import Optional, Union, Any

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader

from experiments import DATA_DIR

# Default constants
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 5
DEFAULT_MIN_DELTA = 1e-3

_REPO_ROOT = Path(__file__).resolve().parent

# The vit-base checkpoint's own preprocessor_config.json (hf_vit/vit_base/)
# specifies image_mean = image_std = 0.5. The ImageNet statistics previously
# used here do NOT match the checkpoint's training distribution; both
# implementations see the same inputs either way, so the paired comparison is
# unaffected, but absolute accuracy is not comparable to runs made before this
# change.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

_LEGACY_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def make_transform(mean, std):
    """The input pipeline for a checkpoint that wants `mean`/`std`. The legacy 0.5/0.5 case
    returns the module-level `transform` itself, so ViT-B and ViT-L inputs are unchanged."""
    mean, std = tuple(float(v) for v in mean), tuple(float(v) for v in std)
    if mean == (0.5, 0.5, 0.5) and std == (0.5, 0.5, 0.5):
        return transform
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])


_cached_loaders: dict[tuple, tuple[DataLoader, DataLoader, DataLoader]] = {}
# NOTE: the cached train loader's shuffle generator is reseeded per run (see
# train_and_evaluate), so cached loaders are single-threaded sequential use
# only — parallel seed runs inside one process would need per-run loaders.


def get_loaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    data_dir: Optional[str] = None,
    norm: Optional[tuple] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns train (45k), val (5k), and test (10k) CIFAR-100 loaders.
    Cached by (batch_size, seed, data_dir, normalization) so repeated calls are fast.
    `norm` is (mean, std) from the checkpoint; None keeps the legacy 0.5/0.5.
    """
    root = data_dir or DATA_DIR
    tf = transform if norm is None else make_transform(*norm)
    cache_key = (batch_size, seed, root) if tf is transform else (batch_size, seed, root, tuple(norm[0]), tuple(norm[1]))
    if cache_key in _cached_loaders:
        return _cached_loaders[cache_key]

    full_trainset = datasets.CIFAR100(root=root, train=True, download=False, transform=tf)
    testset = datasets.CIFAR100(root=root, train=False, download=False, transform=tf)

    # Deterministic split based on seed
    trainset, valset = random_split(
        full_trainset, [45000, 5000], generator=torch.Generator().manual_seed(seed)
    )

    pin_memory = torch.cuda.is_available()
    # Explicit generator for the shuffle order (dedicated, like every other RNG
    # use in the pipeline). Reseeded per run in train_and_evaluate, so batch
    # order is identical across implementations and immune to any other global
    # RNG consumption in the process.
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=pin_memory, generator=torch.Generator())
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)

    _cached_loaders[cache_key] = (train_loader, val_loader, test_loader)
    return train_loader, val_loader, test_loader


def capture_environment(model: Optional[nn.Module] = None) -> dict[str, Any]:
    """Capture environment metadata for full reproducibility."""
    env_info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env_info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        })
    try:
        import transformers
        env_info["transformers_version"] = transformers.__version__
    except Exception:
        env_info["transformers_version"] = None
    try:
        import peft
        env_info["peft_version"] = peft.__version__
    except Exception:
        env_info["peft_version"] = None
    if model is not None:
        cfg = getattr(model, "config", None)
        env_info["attn_implementation"] = (
            getattr(cfg, "_attn_implementation", None) if cfg is not None else None
        )
    env_info["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    env_info["slurm_cpus_per_task"] = os.environ.get("SLURM_CPUS_PER_TASK")
    env_info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    env_info["pytorch_cuda_alloc_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    env_info["hostname"] = socket.gethostname()
    return env_info


RESULT_SCHEMA_VERSION = 2
_PROVENANCE_FILES = ["run.py", "train_utils.py", "models.py", "vision_transformer.py",
                     "lora.py", "experiments.py"]


def code_provenance() -> dict:
    """Which code produced a result: SHA-256 over the six source files (always
    available) plus the git commit and dirty flag when the repo is under git."""
    h = hashlib.sha256()
    for name in _PROVENANCE_FILES:
        try:
            h.update((_REPO_ROOT / name).read_bytes())
        except OSError:
            h.update(b"<missing:" + name.encode() + b">")

    def _git(*args):
        try:
            out = subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True,
                                 text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    status = _git("status", "--porcelain", "--", *_PROVENANCE_FILES)
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_sha256": h.hexdigest(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": (bool(status) if status is not None else None),
        "result_schema_version": RESULT_SCHEMA_VERSION,
    }


def result_path(results_dir: Optional[str], exp_name: str, impl_name: str, seed: int) -> Path:
    """Where a run's result JSON lives. Single source of truth for the
    directory layout. Always nests under the experiment name: a shared
    --results-dir across experiments would otherwise collide on
    {impl}_seed{N}.json."""
    out_dir = (Path(results_dir) / exp_name) if results_dir else (_REPO_ROOT / "results" / exp_name)
    return out_dir / f"{impl_name}_seed{seed}.json"


def _default_save(model: nn.Module, path: Union[str, Path]) -> Path:
    """Saves a full state-dict file and returns the path it wrote."""
    torch.save(model.state_dict(), path)
    return Path(path)


def _default_load(model: nn.Module, path: Union[str, Path]) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def static_memory_bytes(model: nn.Module) -> dict:
    """Exact static GPU footprint of the training setup, in GB.

    Components (all dtype-safe via element_size, so fp32/bf16 both work):
        params        every parameter of the model
        grads         gradients — AdamW only creates state for the TRAINABLE
                      parameters that were passed to it
        adamw_states  exp_avg + exp_avg_sq moments for those same trainable
                      parameters (default AdamW, no amsgrad)

    This part is identical for both implementations of an experiment (same
    parameter counts — enforced by the paired verification gate), so
    subtracting it from a measured peak isolates the implementation-dependent
    ACTIVATION memory, which is the only part that can differ.
    """
    params = [(p.numel(), p.element_size()) for p in model.parameters()]
    trainable = [(p.numel(), p.element_size()) for p in model.parameters() if p.requires_grad]

    params_gb = sum(n * s for n, s in params) / (1024 ** 3)
    grads_gb = sum(n * s for n, s in trainable) / (1024 ** 3)
    adamw_gb = 2 * grads_gb  # exp_avg + exp_avg_sq
    total_gb = params_gb + grads_gb + adamw_gb

    return {
        "static_memory_gb": total_gb,
        "params_gb": params_gb,
        "grads_gb": grads_gb,
        "adamw_states_gb": adamw_gb,
        "trainable_params": len(trainable),
        "total_params": len(params),
    }


def _probe_inference(model: nn.Module, device, seed: int, warmup: int = 3, iterations: int = 20) -> dict:
    """Single-sample (batch-1) inference timing and memory footprint.

    Deployment-representative metrics: what one forward at batch 1 costs in
    time and memory, as opposed to the batch-64 training figures.

    Methodology notes (read before trusting the numbers):
      - Input is FIXED and synthetic: randn(1, 3, 224, 224) from a DEDICATED
        CPU generator, so the global RNG (and paired batch shuffle) is never
        touched. Only tensor SHAPES matter for timing/memory, so synthetic
        values are equivalent to real images for this purpose.
      - `warmup` untimed forwards run first (kernel autotune, cuDNN
        workspaces, allocator warm-up), then `iterations` timed forwards with
        cuda.synchronize() around each. Mean/median/max are reported because
        single-forward latency at batch 1 is noise-sensitive.
      - Memory: the CUDA peak counter is RESET here and read after the probe,
        so these figures are isolated from the training/run peaks. Because
        reset re-initializes the peak to the CURRENT allocation, the reported
        peak INCLUDES the model's resident weights — i.e. it is the total
        memory needed to run one inference, which is the deployment number.
      - CALLER CONTRACT: must be called AFTER the training/run peak figures
        have been captured (this resets the CUDA peak counter) and AFTER the
        caller released grads, optimizer state, loop temporaries and the
        allocator cache. `inference_memory_clean` records whether that held.
    """
    report = {
        "batch": 1,
        "warmup_iters": warmup,
        "timed_iters": iterations,
        "batch1_forward_sec_mean": 0.0,
        "batch1_forward_sec_median": 0.0,
        "batch1_forward_sec_max": 0.0,
        "inference_peak_memory_allocated_gb": 0.0,
        "inference_peak_memory_reserved_gb": 0.0,
        "inference_alloc_before_probe_gb": 0.0,
        # Cleanliness evidence for the probe's baseline: resident bytes at the
        # reset point minus the model's own parameter+buffer bytes. ~0 when the
        # training state was released; a probe that cannot show this must not
        # be aggregated (inference_memory_clean False / None).
        "inference_weights_gb": (sum(p.numel() * p.element_size() for p in model.parameters())
                                 + sum(b.numel() * b.element_size() for b in model.buffers())) / (1024 ** 3),
        "inference_resident_overhead_gb": 0.0,
        "inference_memory_clean": None,
    }
    was_training = model.training
    model.eval()

    # Reset the CUDA peak counter HERE: the probe must measure only its own
    # allocations, not the checkpoint reload or the batch-64 test evaluation
    # that precede it. Without this reset the reported peak is the max over
    # reload + test-eval + probe — dominated by the batch-64 evaluation.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        # memory_allocated (not max_memory_allocated): right after a reset the
        # two coincide, but memory_allocated is correct regardless of when this
        # line is read, so the number can't be broken by future reordering.
        report["inference_alloc_before_probe_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
        report["inference_resident_overhead_gb"] = (report["inference_alloc_before_probe_gb"]
                                                    - report["inference_weights_gb"])
        report["inference_memory_clean"] = report["inference_resident_overhead_gb"] < 0.02

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 3, 224, 224, generator=g).to(device)

    times = []
    try:
        with torch.no_grad():
            for i in range(warmup + iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                if i >= warmup:
                    times.append(elapsed)
    finally:
        if was_training:
            model.train()
        if torch.cuda.is_available():
            report["inference_peak_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
            report["inference_peak_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024 ** 3)

    if times:
        report["batch1_forward_sec_mean"] = sum(times) / len(times)
        ordered = sorted(times)
        mid = len(ordered) // 2
        report["batch1_forward_sec_median"] = (ordered[mid] if len(ordered) % 2
                                               else (ordered[mid - 1] + ordered[mid]) / 2)
        report["batch1_forward_sec_max"] = ordered[-1]
    return report


def _checkpoint_size_mb(checkpoint_path: Optional[Union[str, Path]]) -> Optional[float]:
    """Size on disk of the best checkpoint, in MB. Handles both layouts:
    a state-dict file (.pt) and a PEFT adapter directory (recursive).

    This equals the deployable artifact size in each mode: the full
    fine-tuned weights, or the LoRA adapters (plus head). None when no
    checkpoint exists.
    """
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size / (1024 ** 2)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 2)


def _gb_str(gb: float) -> str:
    """Format a memory figure: GB with two decimals, or MB when small enough
    that GB rounding would print a misleading 0.00 (e.g. LoRA grads are ~2MB)."""
    return f"{gb:.2f} GB" if gb >= 0.01 else f"{gb * 1024:.1f} MB"


def train_and_evaluate(
    model: nn.Module,
    run_config: Any,
    seed: int = DEFAULT_SEED,
    results_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    use_logits: bool = False,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    max_epochs: Optional[int] = None,
    patience: Optional[int] = None,
    min_delta: Optional[float] = None,
    verification: Optional[dict] = None,
    save_fn=None,
    load_fn=None,
    impl_name: str = "model",
    # Legacy arguments support (e.g. model, model_name, run_name, checkpoint_dir, use_logits, lr, ...)
    *args,
    **kwargs,
) -> dict[str, Any]:
    """
    Train and evaluate a model with full determinism, per-batch timings,
    and structured JSON output.

    Measurement scopes (read before quoting any number):
        forward  per batch: model(images) only
        backward per batch: loss computation + loss.backward() (the loss sits
                 inside the backward window — equal for both implementations,
                 but know it's there)
        optim    per batch: optimizer.step() only
        per-epoch peak memory: train + validation phases of that epoch
                 (checkpoint save excluded — it is CPU-side and negligible)
        batch order: a pure function of the seed. The train loader's shuffle
                 generator is reseed per run, so both implementations see the
                 same batches and no other RNG use can perturb the order.
        Timing: time.perf_counter() throughout (monotonic, single clock for
                 both batch timings and epoch totals)

    First-class metrics (all stored in the result JSON):
        metrics.avg_batch_forward_sec      mean forward time per training batch
        metrics.avg_batch_backward_sec     mean backward time per training batch
        metrics.avg_batch_optim_sec        mean optimizer.step() time per batch
        metrics.peak_memory_gb_train       peak GPU memory during training +
                                           validation only
        metrics.peak_memory_gb_run         peak GPU memory for the WHOLE run,
                                           including best-checkpoint reload and
                                           test evaluation
        metrics.static_memory_gb           exact static footprint: params +
                                           grads + AdamW moments (identical for
                                           both implementations)
        metrics.activation_peak_gb_train   peak minus static. NOT pure activation
                                           memory: static includes a full set of
                                           gradients that are not yet live at the
                                           activation high-water mark, so for full
                                           fine-tuning this understates activations
                                           by ~the gradient size (~0.32 GiB). Report
                                           it as peak_minus_static; exact numbers
                                           come from profile_memory.py. (LoRA: the
                                           gradient term is a few MB, negligible.)
                                           Covers training + validation
        metrics.activation_peak_gb_run     same, for the whole run
        metrics.batch1_forward_sec_*       single-sample (batch-1) inference
                                           forward time (mean/median/max over
                                           timed iterations after warmup) —
                                           deployment-representative
        metrics.inference_peak_memory_allocated_gb
                                        peak GPU memory for the batch-1 probe,
                                        INCLUDING the resident model weights;
                                        measured on its own counter after the
                                        run peaks (never affects them)
        metrics.inference_peak_memory_reserved_gb
                                        reserved counterpart of the above
        metrics.best_checkpoint_size_mb    deployable artifact size on disk:
                                        full fine-tuned weights or LoRA
                                        adapters, whichever this run produced
        per_epoch[i]                       per-epoch history; each entry also
                                           carries forward_time_sec,
                                           backward_time_sec,
                                           optim_step_time_sec (summed over
                                           that epoch's batches),
                                           num_train_batches, and
                                           peak_memory_allocated_gb /
                                           peak_memory_reserved_gb for that
                                           epoch (each epoch resets the CUDA
                                           peak counter, so per-epoch values
                                           are per-epoch, not cumulative)
    """
    # ── Parse config & parameters (handles new and legacy callers) ──
    exp_name = "experiment"
    model_name_meta = "vit"
    config_dict = {}

    if hasattr(run_config, "to_dict"):
        config_dict = run_config.to_dict()
        exp_name = getattr(run_config, "name", exp_name)
        model_name_meta = getattr(run_config, "model_name", model_name_meta)
        if lr is None:
            lr = getattr(run_config, "lr", 1e-4)
        if batch_size is None:
            batch_size = getattr(run_config, "batch_size", DEFAULT_BATCH_SIZE)
        if max_epochs is None:
            max_epochs = getattr(run_config, "max_epochs", DEFAULT_EPOCHS)
        if patience is None:
            patience = getattr(run_config, "patience", DEFAULT_PATIENCE)
        if min_delta is None:
            min_delta = getattr(run_config, "min_delta", DEFAULT_MIN_DELTA)
    elif isinstance(run_config, dict):
        config_dict = run_config
        exp_name = run_config.get("name", exp_name)
        model_name_meta = run_config.get("model_name", model_name_meta)
        if lr is None:
            lr = run_config.get("lr", 1e-4)
        if batch_size is None:
            batch_size = run_config.get("batch_size", DEFAULT_BATCH_SIZE)
        if max_epochs is None:
            max_epochs = run_config.get("max_epochs", DEFAULT_EPOCHS)
        if patience is None:
            patience = run_config.get("patience", DEFAULT_PATIENCE)
        if min_delta is None:
            min_delta = run_config.get("min_delta", DEFAULT_MIN_DELTA)
    elif isinstance(run_config, str):
        # Legacy caller: train_and_evaluate(model, model_name, run_name, checkpoint_dir, ...)
        # NOTE the legacy signature passes the RUN NAME positionally where the
        # modern signature takes `seed`; this branch exists only for the
        # archived drivers in old scripts/. New callers use ExperimentConfig.
        model_name_meta = run_config
        legacy_run_name = seed if isinstance(seed, str) else (args[0] if args else "run")
        exp_name = str(legacy_run_name)
        if len(args) > 1:
            checkpoint_dir = str(args[1])
        if lr is None:
            lr = kwargs.get("lr", 1e-4)
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        if max_epochs is None:
            max_epochs = DEFAULT_EPOCHS
        if patience is None:
            patience = DEFAULT_PATIENCE
        if min_delta is None:
            min_delta = DEFAULT_MIN_DELTA
        seed = DEFAULT_SEED

    # `is None` (not `or`): an explicit 0 / 0.0 is a legitimate value and must
    # not be silently replaced by the default.
    if lr is None:
        lr = 1e-4
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if max_epochs is None:
        max_epochs = DEFAULT_EPOCHS
    if patience is None:
        patience = DEFAULT_PATIENCE
    if min_delta is None:
        min_delta = DEFAULT_MIN_DELTA

    if save_fn is None:
        save_fn = _default_save
    if load_fn is None:
        load_fn = _default_load

    # Record the values this run ACTUALLY used. The config's `seeds` list and
    # `lr` are registry defaults; when the CLI overrides them (--seeds, --lr),
    # the stored config would otherwise disagree with what trained.
    if isinstance(config_dict, dict):
        config_dict["seeds_run"] = [seed]
        config_dict["lr"] = lr
        config_dict["max_epochs"] = max_epochs

    # ── Reproducibility & RNG setup ──
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = next(model.parameters()).device
    # Input normalization comes from the checkpoint (ViT/ViT-L: 0.5, Swin: ImageNet statistics).
    # Dict/legacy callers without a model path keep the original 0.5/0.5.
    norm, norm_source = None, "legacy-default"
    if isinstance(config_dict, dict) and config_dict.get("model_local_path"):
        from experiments import checkpoint_normalization
        mean, std, norm_source = checkpoint_normalization(str(config_dict["model_local_path"]),
                                                          str(config_dict.get("arch", "vitb")))
        norm = (tuple(mean), tuple(std))
    # The legacy 0.5/0.5 case calls get_loaders exactly as before (no extra argument).
    loader_kwargs = {"norm": norm} if (norm is not None and norm != _LEGACY_NORM) else {}
    train_loader, val_loader, test_loader = get_loaders(batch_size=batch_size, seed=seed, **loader_kwargs)

    # Reseed the loader's own shuffle generator for THIS run: the loaders are
    # cached and shared between implementations, so without this the second
    # implementation would continue the first one's generator state and see a
    # different batch order. Reseeding here makes the order a pure function of
    # the seed, independent of the global RNG.
    sampler_generator = getattr(getattr(train_loader, "sampler", None), "generator", None)
    if sampler_generator is not None:
        sampler_generator.manual_seed(seed)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Provenance of the optimization setup. These settings are NOT part of
    # ExperimentConfig, so they are recorded here from the actual objects
    # (weight decay read from the optimizer itself, never hardcoded).
    training_setup = {
        "optimizer": type(optimizer).__name__,
        "optimizer_weight_decay": optimizer.defaults.get("weight_decay"),
        "lr_scheduler": "none",
        "criterion": type(criterion).__name__,
        # Effective epoch cap. Differs from the registry value only for deliberately
        # truncated runs (--max-epochs), which aggregate.py flags.
        "max_epochs": max_epochs,
        "input_normalization": {"mean": list(norm[0]) if norm else [0.5, 0.5, 0.5],
                                "std": list(norm[1]) if norm else [0.5, 0.5, 0.5],
                                "source": norm_source},
    }

    # Exact static footprint: parameters + gradients (trainable only) + AdamW
    # moments (trainable only). Subtracted from the measured peaks below to
    # report the implementation-dependent activation memory.
    static_mem = static_memory_bytes(model)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = -1
    stopping_epoch = max_epochs
    best_checkpoint = None

    # Always nest under the experiment name: a shared --checkpoint-dir across
    # experiments (or concurrent SLURM jobs) would otherwise collide on
    # {impl}_seed{N} checkpoint files, and a reload could load another
    # experiment's weights.
    ckpt_folder = (Path(checkpoint_dir) / exp_name) if checkpoint_dir else (_REPO_ROOT / "checkpoints" / exp_name)
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_folder / f"{impl_name}_seed{seed}_best.pt"

    print("=" * 60)
    print(f"Starting Training: {exp_name} | Impl: {impl_name} | Seed: {seed} | LR: {lr}")
    print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")
    print("=" * 60)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    per_epoch_stats = []
    epoch_train_times = []
    epoch_val_times = []
    all_forward_times = []
    all_backward_times = []
    all_optim_times = []
    # Per-epoch memory peaks (allocated / reserved), collected because each
    # epoch resets the CUDA peak counter so the run-level peak is the max over
    # epochs. See the epoch loop below for what is recorded.
    epoch_allocated_peaks = []
    epoch_reserved_peaks = []

    for epoch in range(max_epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()

        model.train()
        train_loss = 0.0
        train_correct = 0

        # Slice markers: so this epoch's forward/backward/optim batch timings
        # can be summed from the run-level lists below and stored per epoch.
        epoch_fwd_start = len(all_forward_times)
        epoch_bwd_start = len(all_backward_times)
        epoch_opt_start = len(all_optim_times)

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            logits = outputs.logits if use_logits else outputs
            loss = criterion(logits, labels)

            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()

            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            all_forward_times.append(t1 - t0)
            all_backward_times.append(t2 - t1)
            all_optim_times.append(t3 - t2)

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()

        train_acc = train_correct / len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_train_time = time.perf_counter() - epoch_start_time
        epoch_train_times.append(epoch_train_time)

        # ── Validation ──
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_val_start = time.perf_counter()

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                logits = outputs.logits if use_logits else outputs
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_val_time = time.perf_counter() - epoch_val_start
        epoch_val_times.append(epoch_val_time)

        # ── First-class per-epoch metrics: batch timings + memory peak ──
        # The CUDA peak counter is reset after each epoch's capture, so
        # epoch_alloc_peak is THIS epoch's peak, not a cumulative one.
        if torch.cuda.is_available():
            epoch_alloc_peak = torch.cuda.max_memory_allocated()
            epoch_reserved_peak = torch.cuda.max_memory_reserved()
            torch.cuda.reset_peak_memory_stats()
        else:
            epoch_alloc_peak = 0
            epoch_reserved_peak = 0
        epoch_allocated_peaks.append(epoch_alloc_peak)
        epoch_reserved_peaks.append(epoch_reserved_peak)

        epoch_fwd_total = sum(all_forward_times[epoch_fwd_start:])
        epoch_bwd_total = sum(all_backward_times[epoch_bwd_start:])
        epoch_opt_total = sum(all_optim_times[epoch_opt_start:])
        epoch_num_batches = len(all_forward_times) - epoch_fwd_start

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc * 100,
            "val_loss": val_loss,
            "val_acc": val_acc * 100,
            "train_time_sec": epoch_train_time,
            "val_time_sec": epoch_val_time,
            # Per-epoch batch-phase timings (summed over this epoch's batches;
            # divide by num_train_batches for the per-batch mean).
            "forward_time_sec": epoch_fwd_total,
            "backward_time_sec": epoch_bwd_total,
            "optim_step_time_sec": epoch_opt_total,
            "num_train_batches": epoch_num_batches,
            # Per-epoch GPU memory peaks.
            "peak_memory_allocated_gb": epoch_alloc_peak / (1024 ** 3),
            "peak_memory_reserved_gb": epoch_reserved_peak / (1024 ** 3),
        }
        per_epoch_stats.append(epoch_info)

        print(
            f"Epoch {epoch+1:02d}/{max_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Train Time: {epoch_train_time:.2f}s | Val Time: {epoch_val_time:.2f}s | "
            # Per-BATCH milliseconds (epoch totals / num_train_batches) so this
            # line is directly comparable with the per-batch run averages.
            f"Fwd/Bwd/Opt: {epoch_fwd_total / epoch_num_batches * 1000:.1f}/"
            f"{epoch_bwd_total / epoch_num_batches * 1000:.1f}/"
            f"{epoch_opt_total / epoch_num_batches * 1000:.2f} ms/batch | "
            f"Peak Mem: {epoch_alloc_peak/(1024**3):.2f}/"
            f"{epoch_reserved_peak/(1024**3):.2f} GB (alloc/res)"
        )

        # ── Early stopping check ──
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            saved_path = save_fn(model, checkpoint_path)
            if saved_path is not None:
                best_checkpoint = saved_path
            best_epoch = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopping_epoch = epoch + 1
                print(f"Early stopping triggered at epoch {stopping_epoch}.")
                break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    # Peak memory over training + validation only (BEFORE checkpoint reload and
    # test evaluation). The per-epoch captures above reset the CUDA peak
    # counter, so the training peak is the MAX over epoch peaks. The
    # whole-run figure (peak_memory_run_gb, below) additionally covers the
    # checkpoint reload and test evaluation.
    peak_memory_train_gb = (max(epoch_allocated_peaks) / (1024 ** 3)) if epoch_allocated_peaks else 0.0
    peak_memory_reserved_train_gb = (max(epoch_reserved_peaks) / (1024 ** 3)) if epoch_reserved_peaks else 0.0

    # ── Test evaluation with best checkpoint ──
    # A saver may report the exact path it wrote (PEFT writes a DIRECTORY, not
    # the .pt-style path it was given). Savers that return None keep the old
    # behavior: reload from the given path if it exists.
    if best_checkpoint is not None and Path(best_checkpoint).exists():
        model = load_fn(model, best_checkpoint)
    elif checkpoint_path.exists():
        model = load_fn(model, checkpoint_path)
    model.eval()
    test_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = (outputs.logits if use_logits else outputs).argmax(dim=-1)
            test_correct += (preds == labels).sum().item()

    test_acc = (test_correct / len(test_loader.dataset)) * 100

    # Peak memory for the WHOLE run, including best-checkpoint reload and test
    # evaluation. In practice this equals the training peak (reload + no_grad
    # eval cannot exceed it), so it is kept for completeness, not as separate
    # evidence. The CUDA counter currently covers only the reload + test phase
    # (the epoch captures reset it), so combine it with the training peak.
    peak_memory_run_gb = (max(max(epoch_allocated_peaks, default=0),
                              torch.cuda.max_memory_allocated()) / (1024 ** 3)) if torch.cuda.is_available() else 0.0
    peak_memory_reserved_run_gb = (max(max(epoch_reserved_peaks, default=0),
                                       torch.cuda.max_memory_reserved()) / (1024 ** 3)) if torch.cuda.is_available() else 0.0

    # Implementation-dependent memory: measured peak minus the static footprint
    # (params + grads + AdamW moments). Clamped at 0 — CPU runs measure 0.0
    # peaks while the static footprint is a real property of the model.
    activation_peak_train_gb = max(0.0, peak_memory_train_gb - static_mem["static_memory_gb"])
    activation_peak_run_gb = max(0.0, peak_memory_run_gb - static_mem["static_memory_gb"])

    # ── Deployment-representative probe: batch-1 inference ──
    # Must run AFTER the peak captures above: _probe_inference resets the CUDA
    # peak counter, which would otherwise erase the training/run peak figures.
    # Release everything that exists only because we trained, so the probe's
    # baseline is the model's own weights: loop temporaries, gradients, AdamW
    # state, then the allocator cache. (Before this existed the probe read
    # weights + grads + AdamW state + temporaries.) Peak figures above are
    # already captured.
    images = labels = outputs = logits = loss = preds = None
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    probe_report = _probe_inference(model, device, seed)
    checkpoint_size_mb = _checkpoint_size_mb(
        best_checkpoint if best_checkpoint is not None else checkpoint_path
    )
    total_train_time = sum(epoch_train_times)
    total_val_time = sum(epoch_val_times)

    result_file = result_path(results_dir, exp_name, impl_name, seed)
    result_file.parent.mkdir(parents=True, exist_ok=True)

    # Per-task artifact of LoRA runs: head + adapters in the canonical,
    # implementation-independent format (see models.export_adapter). Small
    # (MBs), kept next to the result JSON. Needed later to test merge/unmerge
    # on TRAINED adapters. Never fails the run: training hours are not
    # forfeited over an export problem.
    adapter_file = None
    adapter_size_mb = None
    try:
        from models import has_lora_params, export_adapter
        if has_lora_params(model):
            adapter_file = export_adapter(
                model, result_file.with_name(f"{impl_name}_seed{seed}_adapter.pt"))
            adapter_size_mb = adapter_file.stat().st_size / (1024 ** 2)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: adapter export failed: {exc}")

    # Invariant: the per-epoch time slices must account for every batch timing.
    # Guards the slice-marker bookkeeping against future refactors.
    assert abs(sum(e["forward_time_sec"] for e in per_epoch_stats) - sum(all_forward_times)) < 1e-6
    assert abs(sum(e["backward_time_sec"] for e in per_epoch_stats) - sum(all_backward_times)) < 1e-6
    assert abs(sum(e["optim_step_time_sec"] for e in per_epoch_stats) - sum(all_optim_times)) < 1e-6
    avg_train_time = total_train_time / len(epoch_train_times) if epoch_train_times else 0.0
    avg_val_time = total_val_time / len(epoch_val_times) if epoch_val_times else 0.0

    avg_fwd = (sum(all_forward_times) / len(all_forward_times)) if all_forward_times else 0.0
    avg_bwd = (sum(all_backward_times) / len(all_backward_times)) if all_backward_times else 0.0
    avg_opt = (sum(all_optim_times) / len(all_optim_times)) if all_optim_times else 0.0

    num_samples_trained = len(train_loader.dataset) * len(epoch_train_times)
    train_throughput = (num_samples_trained / total_train_time) if total_train_time > 0 else 0.0

    print(f"\n--- Final Results ({exp_name} | {impl_name} | seed {seed}) ---")
    print(f"Total Time (train+eval+probe): {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Total Training Time        : {total_train_time:.2f}s")
    print(f"Average Epoch Train Time   : {avg_train_time:.2f}s")
    print(f"Average Epoch Val Time     : {avg_val_time:.2f}s")
    print(f"Per-Batch Fwd / Bwd / Opt  : {avg_fwd*1000:.2f}ms / {avg_bwd*1000:.2f}ms / {avg_opt*1000:.2f}ms")
    print(f"Training Throughput        : {train_throughput:.2f} samples/sec")
    print(f"Best Epoch / Stopping Epoch: {best_epoch} / {stopping_epoch}")
    print(f"Final Test Accuracy        : {test_acc:.2f}%")
    print(f"Peak GPU Memory (train+val / whole run): "
          f"{peak_memory_train_gb:.2f} / {peak_memory_run_gb:.2f} GB allocated, "
          f"{peak_memory_reserved_train_gb:.2f} / {peak_memory_reserved_run_gb:.2f} GB reserved")
    print(f"Static Footprint            : {static_mem['static_memory_gb']:.2f} GB "
          f"(params {_gb_str(static_mem['params_gb'])} + grads {_gb_str(static_mem['grads_gb'])} "
          f"+ adamw {_gb_str(static_mem['adamw_states_gb'])})")
    print(f"Activation Peak (allocated, train+val / whole run): "
          f"{activation_peak_train_gb:.2f} / {activation_peak_run_gb:.2f} GB")
    print(f"Batch-1 Inference           : {probe_report['batch1_forward_sec_mean'] * 1000:.2f} ms mean / "
          f"{probe_report['batch1_forward_sec_median'] * 1000:.2f} ms median "
          f"({probe_report['timed_iters']} iters)")
    print(f"Inference Peak Memory       : {probe_report['inference_peak_memory_allocated_gb']:.2f} GB alloc / "
          f"{probe_report['inference_peak_memory_reserved_gb']:.2f} GB reserved (incl. model weights)")
    if checkpoint_size_mb is not None:
        print(f"Best Checkpoint Size        : {checkpoint_size_mb:.1f} MB")
    print("=" * 60)

    # Transformer geometry of the checkpoint this run used (None if unreadable;
    # must not be able to crash a finished run).
    try:
        from experiments import arch_dims
        model_dims = dict(arch_dims(str(config_dict.get("model_local_path")), True))
    except Exception:
        model_dims = None

    result: dict[str, Any] = {
        "experiment": exp_name,
        "implementation": impl_name,
        "seed": seed,
        "config": config_dict,
        "model_dims": model_dims,
        "environment": capture_environment(model),
        "training_setup": training_setup,
        # Paired-init verification evidence for THIS run (init tensors,
        # trainable-set match, forward equivalence). Produced by run.py's
        # pre-training gate; None only for legacy direct callers of this
        # function, which bypass that gate.
        "verification": verification,
        "provenance": code_provenance(),
        "metrics": {
            "test_acc": test_acc,
            "best_epoch": best_epoch,
            "stopping_epoch": stopping_epoch,
            "total_time_sec": total_time,
            "total_train_time_sec": total_train_time,
            "total_val_time_sec": total_val_time,
            "avg_epoch_train_time_sec": avg_train_time,
            "avg_epoch_val_time_sec": avg_val_time,
            "avg_batch_forward_sec": avg_fwd,
            "avg_batch_backward_sec": avg_bwd,
            "avg_batch_optim_sec": avg_opt,
            "train_throughput_samples_per_sec": train_throughput,
            "peak_memory_gb_train": peak_memory_train_gb,
            "peak_memory_reserved_gb_train": peak_memory_reserved_train_gb,
            "peak_memory_gb_run": peak_memory_run_gb,
            "peak_memory_reserved_gb_run": peak_memory_reserved_run_gb,
            # Memory decomposition: the static part (params + grads + AdamW
            # moments) is identical for both implementations; the activation
            # peak is the implementation-dependent remainder.
            "static_memory_gb": static_mem["static_memory_gb"],
            "static_memory_breakdown_gb": {
                "params": static_mem["params_gb"],
                "grads": static_mem["grads_gb"],
                "adamw_states": static_mem["adamw_states_gb"],
            },
            "activation_peak_gb_train": activation_peak_train_gb,
            "activation_peak_gb_run": activation_peak_run_gb,
            # Deployment-representative probe (batch-1 inference), measured on
            # its own memory counter AFTER the run peaks — does not affect them.
            # The inference peak includes the resident model weights.
            "batch1_forward_sec_mean": probe_report["batch1_forward_sec_mean"],
            "batch1_forward_sec_median": probe_report["batch1_forward_sec_median"],
            "batch1_forward_sec_max": probe_report["batch1_forward_sec_max"],
            "inference_peak_memory_allocated_gb": probe_report["inference_peak_memory_allocated_gb"],
            "inference_peak_memory_reserved_gb": probe_report["inference_peak_memory_reserved_gb"],
            "inference_alloc_before_probe_gb": probe_report["inference_alloc_before_probe_gb"],
            "inference_weights_gb": probe_report["inference_weights_gb"],
            "inference_resident_overhead_gb": probe_report["inference_resident_overhead_gb"],
            "inference_memory_clean": probe_report["inference_memory_clean"],
            # Size of the exported head+adapter file (LoRA runs only; identical
            # semantics for both implementations, unlike best_checkpoint_size_mb
            # which is a full state dict for scratch).
            "adapter_export_size_mb": adapter_size_mb,
            # Deployable artifact size on disk (best checkpoint), MB.
            "best_checkpoint_size_mb": checkpoint_size_mb,
        },
        "per_epoch": per_epoch_stats,
    }

    # ── Save structured JSON output ──
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {result_file}")

    # ── Delete the best checkpoint ──
    # The checkpoint's only consumers were the best-model reload and the test
    # evaluation above; best_checkpoint_size_mb was recorded before this point,
    # so no metric depends on the artifact surviving. Retraining a known
    # configuration is roughly an hour, so keeping ~344MB per run across a
    # multi-day sweep is pure waste. Deletion failure never fails the run.
    if best_checkpoint is not None:
        ckpt_path = Path(best_checkpoint)
        try:
            if ckpt_path.is_dir():
                shutil.rmtree(ckpt_path)
                print(f"Checkpoint deleted (directory): {ckpt_path}")
            elif ckpt_path.exists():
                ckpt_path.unlink()
                print(f"Checkpoint deleted (file): {ckpt_path}")
        except OSError as exc:
            print(f"Warning: could not delete checkpoint {ckpt_path}: {exc}")

    # For legacy callers compatibility, also populate top-level fields
    result.update({
        "total_time": total_time,
        "train_time": total_train_time,
        "val_time": total_val_time,
        "avg_train_time": avg_train_time,
        "avg_val_time": avg_val_time,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "peak_memory_gb": peak_memory_train_gb,
        "peak_reserved_gb": peak_memory_reserved_train_gb,
    })

    return result
