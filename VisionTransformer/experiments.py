"""
Experiment configuration definitions and registry.

All experiment configs live in the EXPERIMENTS dict. Adding a new experiment
is just adding another entry — no new files needed.
"""

from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Optional, List
import json
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
# Weights (hf_vit/<model>) and data (cifar100) are looked up inside this folder first, then under
# $ASSETS_ROOT if it is set. Each location can also be set directly (VIT_BASE_PATH, DATA_DIR, ...).
_REPO_ROOT = Path(__file__).resolve().parent

def _resolve_path(rel_path: str) -> str:
    local_candidate = _REPO_ROOT / rel_path
    if local_candidate.exists():
        return str(local_candidate)
    assets_root = os.environ.get("ASSETS_ROOT")
    if assets_root and (Path(assets_root) / rel_path).exists():
        return str(Path(assets_root) / rel_path)
    return str(local_candidate)

VIT_BASE_PATH = os.environ.get(
    "VIT_BASE_PATH",
    _resolve_path("hf_vit/vit_base")
)

VIT_LARGE_PATH = os.environ.get(
    "VIT_LARGE_PATH",
    _resolve_path("hf_vit/vit_large")
)

VIT_HUGE_PATH = os.environ.get(
    "VIT_HUGE_PATH",
    _resolve_path("hf_vit/vit_huge")
)

DEIT_BASE_PATH = os.environ.get(
    "DEIT_BASE_PATH",
    _resolve_path("hf_vit/deit_base")
)

SWIN_BASE_PATH = os.environ.get(
    "SWIN_BASE_PATH",
    _resolve_path("hf_vit/swin_base")
)

DATA_DIR = os.environ.get(
    "DATA_DIR",
    _resolve_path("cifar100")
)


# ── Model geometry (read from the checkpoint, never hardcoded) ────────
#
# The scratch model, the analytic parameter counts in aggregate.py and the layer
# validation all take their dimensions from the checkpoint's own config.json, so
# a new model size is one registry entry plus its weights directory.

# Reference full-fine-tuning baseline per architecture: what LoRA runs are compared
# against in aggregate.py ("LoRA − full fine-tuning"). ViT-B's is the lr 1e-4 baseline.
ARCH_BASELINE = {"vitb": "baseline", "vitl": "vitl_baseline_lr1e-4",
                 "deitb": "deitb_baseline_lr1e-4", "swinb": "swinb_baseline_lr1e-4"}

# Geometry of the original ViT-B/16 checkpoint, used only when a checkpoint's
# config.json is unavailable (e.g. aggregating on a machine without hf_vit/).
_VITB_FALLBACK = dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                      intermediate_size=3072, patch_size=16, image_size=224, num_channels=3,
                      layer_norm_eps=1e-12, qkv_bias=True, hidden_act="gelu")


def _read_config(model_local_path: str) -> Optional[dict]:
    cfg_file = Path(model_local_path) / "config.json"
    return json.loads(cfg_file.read_text()) if cfg_file.exists() else None


def model_family(model_local_path: str, strict: bool = False) -> str:
    """'vit' (ViT and DeiT: one architecture, different pretraining) or 'swin', from the
    checkpoint's config.json. strict=False treats a missing file as the original ViT-B."""
    raw = _read_config(model_local_path)
    if raw is None:
        if strict:
            raise FileNotFoundError(f"no config.json under {model_local_path}")
        return "vit"
    mt = raw.get("model_type", "vit")
    if mt not in ("vit", "swin"):
        raise ValueError(f"unsupported model_type {mt!r} in {model_local_path}/config.json "
                         f"(supported: 'vit' [ViT, DeiT], 'swin')")
    return mt


# Families that have a from-scratch implementation in this repo (vision_transformer.py +
# lora.py). Families without one run library-only: no paired comparison, no pairing gate.
FAMILIES_WITH_SCRATCH = ("vit",)


def has_scratch_impl(model_local_path: str) -> bool:
    return model_family(model_local_path, strict=True) in FAMILIES_WITH_SCRATCH


@lru_cache(maxsize=None)
def arch_dims(model_local_path: str, strict: bool = False) -> dict:
    """Transformer geometry from <path>/config.json. strict=False falls back to the
    ViT-B geometry when the file is missing (aggregation only); model building
    uses strict=True so a wrong path can never silently build a ViT-B.

    A Swin checkpoint returns the Swin geometry (it has stages, so `depths` is a list);
    use `num_blocks(dims)` for the number of LoRA-addressable blocks of either family."""
    raw = _read_config(model_local_path)
    if raw is None:
        if strict:
            raise FileNotFoundError(f"no config.json under {model_local_path}")
        return dict(_VITB_FALLBACK)
    if model_family(model_local_path) == "swin":
        return _swin_dims(raw)
    dims = {k: raw.get(k, v) for k, v in _VITB_FALLBACK.items()}
    return dims


def _swin_dims(raw: dict) -> dict:
    depths = list(raw["depths"])
    return dict(
        family="swin", embed_dim=raw["embed_dim"], depths=depths, num_heads=list(raw["num_heads"]),
        window_size=raw["window_size"], mlp_ratio=raw.get("mlp_ratio", 4.0),
        patch_size=raw["patch_size"], image_size=raw["image_size"], num_channels=raw.get("num_channels", 3),
        qkv_bias=raw.get("qkv_bias", True), layer_norm_eps=raw.get("layer_norm_eps", 1e-5),
        hidden_act=raw.get("hidden_act", "gelu"), drop_path_rate=raw.get("drop_path_rate", 0.0),
        hidden_size=raw["embed_dim"] * 2 ** (len(depths) - 1),      # width of the last stage
    )


def num_blocks(dims: dict) -> int:
    """Number of transformer blocks LoRA `layers` can address: ViT/DeiT depth, or for Swin
    the blocks of all stages counted in one running 0-indexed sequence."""
    return sum(dims["depths"]) if "depths" in dims else dims["num_hidden_layers"]


def swin_block_table(dims: dict) -> list:
    """[(stage, block_in_stage, width, heads)] in global block order."""
    out = []
    for s, (depth, heads) in enumerate(zip(dims["depths"], dims["num_heads"])):
        out += [(s, b, dims["embed_dim"] * 2 ** s, heads) for b in range(depth)]
    return out


def validate_layers(layers: Optional[List[int]], num_layers: int) -> None:
    """Range check against a concrete model depth (LoRAExperimentConfig itself
    cannot know the depth)."""
    if layers is None:
        return
    bad = [i for i in layers if not (0 <= i < num_layers)]
    if bad:
        raise ValueError(f"LoRA layers {bad} are out of range for a model with {num_layers} "
                         f"blocks (valid: 0..{num_layers - 1}, 0-indexed)")


# Legacy input normalization (the original ViT-B runs): mean = std = 0.5.
LEGACY_NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Architectures whose checkpoint metadata is known to be wrong. The Hugging Face
# facebook/deit-base-patch16-224 preprocessor_config.json says mean = std = 0.5 (it matches the ViT
# files byte for byte), but DeiT was trained with the ImageNet statistics: timm's DeiT-B config
# (timm/deit_base_patch16_224.fb_in1k) lists 0.485/0.456/0.406 and 0.229/0.224/0.225. The override is
# applied BEFORE the file is read and is recorded as the source in every result JSON.
NORM_OVERRIDE = {"deitb": (IMAGENET_NORM[0], IMAGENET_NORM[1],
                           "override: ImageNet statistics of the DeiT recipe (the HF preprocessor_config.json "
                           "says 0.5, a copy of the ViT file; timm's DeiT-B config uses ImageNet)")}


def checkpoint_normalization(model_local_path: str, arch: str = "vitb") -> tuple:
    """((mean), (std), source) the checkpoint's own preprocessor_config.json asks for.

    Every checkpoint is fine-tuned on inputs normalized the way it was pretrained, so this is
    read from the checkpoint and never assumed (Swin uses the ImageNet statistics, the ViT
    checkpoints 0.5). A missing file is tolerated only for the two original ViT archs, where
    it falls back to the legacy 0.5/0.5 and says so in `source`; anything else must have it."""
    if arch in NORM_OVERRIDE:
        return NORM_OVERRIDE[arch]
    f = Path(model_local_path) / "preprocessor_config.json"
    if f.exists():
        raw = json.loads(f.read_text())
        if not raw.get("do_normalize", True):
            raise ValueError(f"{f}: do_normalize is false; this pipeline always normalizes")
        return tuple(raw["image_mean"]), tuple(raw["image_std"]), "checkpoint"
    if arch in ("vitb", "vitl"):
        return LEGACY_NORM[0], LEGACY_NORM[1], "legacy-default (no preprocessor_config.json)"
    raise FileNotFoundError(f"{f} is required for arch {arch!r} (it defines the input normalization)")


_PROJ_DIMS = {"q_proj": "dd", "k_proj": "dd", "v_proj": "dd", "o_proj": "dd", "fc1": "dm", "fc2": "md"}


def _swin_dense_counts(dims: dict, num_labels: int) -> tuple:
    e, p, ch, w = dims["embed_dim"], dims["patch_size"], dims["num_channels"], dims["window_size"]
    table = (2 * w - 1) ** 2                         # relative-position-bias table rows
    elems = (ch * p * p * e + e) + 2 * e             # patch projection (conv) + embedding LayerNorm
    tensors = 4
    for s, b, d, h in swin_block_table(dims):
        m = int(d * dims["mlp_ratio"])
        elems += 2 * d + 4 * (d * d + d) + table * h + 2 * d + (d * m + m) + (m * d + d)
        tensors += 17
    for s in range(len(dims["depths"]) - 1):         # patch merging after every stage but the last
        d = dims["embed_dim"] * 2 ** s
        elems += 4 * d * 2 * d + 2 * 4 * d           # bias-free reduction Linear(4d -> 2d) + LayerNorm(4d)
        tensors += 3
    dl = dims["hidden_size"]
    elems += 2 * dl + (dl * num_labels + num_labels)
    return elems, tensors + 4


def dense_param_counts(dims: dict, num_labels: int) -> tuple:
    """(elements, tensors) of the whole classification model: every parameter
    trainable, as in full fine-tuning."""
    if "depths" in dims:
        return _swin_dense_counts(dims, num_labels)
    d, m, L = dims["hidden_size"], dims["intermediate_size"], dims["num_hidden_layers"]
    p, img, ch = dims["patch_size"], dims["image_size"], dims["num_channels"]
    n_tokens = (img // p) ** 2 + 1
    per_layer = 2 * d + 4 * (d * d + d) + 2 * d + (d * m + m) + (m * d + d)
    elems = (ch * p * p * d + d) + d + n_tokens * d + L * per_layer + 2 * d + (d * num_labels + num_labels)
    return elems, 8 + 16 * L


def trainable_counts(dims: dict, num_labels: int, lora: Optional[dict]) -> tuple:
    """(elements, tensors) expected to be trainable. `lora` is the LoRA part of a
    config dict (rank, target_attention, target_mlp, optional layers) or None
    for full fine-tuning."""
    if not lora:
        return dense_param_counts(dims, num_labels)
    L = num_blocks(dims)
    layers = lora.get("layers")
    validate_layers(layers, L)
    selected = list(range(L)) if layers is None else list(layers)
    mods = list(lora["target_attention"]) + list(lora["target_mlp"])
    if "depths" in dims:                              # width, and so adapter size, varies per block
        table = swin_block_table(dims)
        widths = [(table[i][2], int(table[i][2] * dims["mlp_ratio"])) for i in selected]
    else:
        widths = [(dims["hidden_size"], dims["intermediate_size"])] * len(selected)
    elems = dims["hidden_size"] * num_labels + num_labels
    for d, m in widths:
        in_out = {"dd": 2 * d, "dm": d + m, "md": m + d}
        elems += lora["rank"] * sum(in_out[_PROJ_DIMS[x]] for x in mods)
    return elems, 2 + 2 * len(selected) * len(mods)


@dataclass
class LoRAExperimentConfig:
    """LoRA-specific settings for an experiment."""
    rank: int
    alpha: int
    target_attention: List[str] = field(default_factory=list)
    target_mlp: List[str] = field(default_factory=list)
    dropout: float = 0.0
    bias: str = "none"
    # Which transformer blocks get adapters: None = every block (the original
    # behavior), otherwise an explicit 0-indexed list such as [6, 7, 8, 9, 10, 11].
    # Deliberately explicit: no presets, so the JSON of every run states exactly
    # which blocks were adapted. The upper bound depends on the model depth and is
    # checked when the model is built (models.validate_layers).
    layers: Optional[List[int]] = None

    def __post_init__(self):
        if self.layers is None:
            return
        if (not isinstance(self.layers, (list, tuple)) or len(self.layers) == 0
                or not all(isinstance(i, int) and not isinstance(i, bool) for i in self.layers)):
            raise ValueError(f"layers must be None or a non-empty list of ints, got {self.layers!r}")
        if min(self.layers) < 0:
            raise ValueError(f"layers are 0-indexed and must be >= 0, got {list(self.layers)!r}")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError(f"layers contains duplicates: {list(self.layers)!r}")
        self.layers = sorted(self.layers)          # canonical: [1, 0] and [0, 1] are the same experiment


@dataclass
class ExperimentConfig:
    """
    Complete configuration for one experiment (may run across multiple seeds).

    Fields:
        name:             Unique identifier used for file paths and logging.
        model_name:       HuggingFace model identifier (for metadata/fallback).
        model_local_path: Absolute path to local pretrained weights.
        num_labels:       Number of output classes.
        lr:               Learning rate.
        batch_size:       Training batch size.
        max_epochs:       Maximum training epochs before forced stop.
        patience:         Early stopping patience (epochs without improvement).
        min_delta:        Minimum val loss improvement to count as progress.
        seeds:            List of random seeds to run.
        lora:             LoRA config, or None for full fine-tuning baseline.
        implementations:  List of implementations to run (e.g. ["scratch", "library"]).
        arch:             Architecture tag ("vitb" | "vitl" | "deitb" | "swinb"). Used by aggregate.py to
                          keep each architecture in its own tables and to pick the reference baseline.
                          Architectures whose family has no scratch implementation (Swin) must list
                          implementations=["library"].
    """
    name: str
    model_name: str
    model_local_path: str
    num_labels: int = 100
    lr: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 5
    min_delta: float = 1e-3
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    lora: Optional[LoRAExperimentConfig] = None
    implementations: List[str] = field(default_factory=lambda: ["scratch", "library"])
    arch: str = "vitb"

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict (for structured logging)."""
        return asdict(self)


# ── Experiment Registry ──────────────────────────────────────────
#
# Each entry defines one experiment. The runner will execute it across
# all specified seeds, for both scratch and library implementations.
#
# To add a new experiment: just add another key-value pair below.

EXPERIMENTS: dict[str, ExperimentConfig] = {

    # ── Full Fine-Tuning Baseline ────────────────────────────────
    "baseline": ExperimentConfig(
        name="baseline",
        model_name="google/vit-base-patch16-224",
        model_local_path=VIT_BASE_PATH,
    ),

    # ── Full fine-tuning LR sweep (same recipe, different LR) ─────
    # Separate registry names so results/checkpoints never collide with the
    # lr=1e-4 `baseline`, and the LR is part of the experiment identity.
    # Same 5 default seeds as every other experiment, so the sweep is directly
    # comparable (paired per seed) with the lr=1e-4 baseline and the LoRA runs.
    "baseline_lr1e-3": ExperimentConfig(
        name="baseline_lr1e-3",
        model_name="google/vit-base-patch16-224",
        model_local_path=VIT_BASE_PATH,
        lr=1e-3,
    ),
    "baseline_lr1e-5": ExperimentConfig(
        name="baseline_lr1e-5",
        model_name="google/vit-base-patch16-224",
        model_local_path=VIT_BASE_PATH,
        lr=1e-5,
    ),

    # ── LoRA: Attention Only (Q, K, V) ───────────────────────────
    "lora_attn_r8": ExperimentConfig(
        name="lora_attn_r8",
        model_name="google/vit-base-patch16-224",
        model_local_path=VIT_BASE_PATH,
        lora=LoRAExperimentConfig(
            rank=8, alpha=16,
            target_attention=["q_proj", "k_proj", "v_proj"],
        ),
    ),
    "lora_attn_r16": ExperimentConfig(
        name="lora_attn_r16",
        model_name="google/vit-base-patch16-224",
        model_local_path=VIT_BASE_PATH,
        lora=LoRAExperimentConfig(
            rank=16, alpha=32,
            target_attention=["q_proj", "k_proj", "v_proj"],
        ),
    ),

    # ── LoRA: MLP Only (fc1, fc2) ────────────────────────────────
    "lora_mlp_r8": ExperimentConfig(
        name="lora_mlp_r8",
        model_name="google/vit-base-patch16-224",
        model_local_path=VIT_BASE_PATH,
        lora=LoRAExperimentConfig(
            rank=8, alpha=16,
            target_mlp=["fc1", "fc2"],
        ),
    ),
    "lora_mlp_r16": ExperimentConfig(
        name="lora_mlp_r16",
        model_name="google/vit-base-patch16-224",
        model_local_path=VIT_BASE_PATH,
        lora=LoRAExperimentConfig(
            rank=16, alpha=32,
            target_mlp=["fc1", "fc2"],
        ),
    ),

    # ── LoRA: Attention + MLP ────────────────────────────────────
    "lora_attn_mlp_r8": ExperimentConfig(
        name="lora_attn_mlp_r8",
        model_name="google/vit-base-patch16-224",
        model_local_path=VIT_BASE_PATH,
        lora=LoRAExperimentConfig(
            rank=8, alpha=16,
            target_attention=["q_proj", "k_proj", "v_proj"],
            target_mlp=["fc1", "fc2"],
        ),
    ),
    "lora_attn_mlp_r16": ExperimentConfig(
        name="lora_attn_mlp_r16",
        model_name="google/vit-base-patch16-224",
        model_local_path=VIT_BASE_PATH,
        lora=LoRAExperimentConfig(
            rank=16, alpha=32,
            target_attention=["q_proj", "k_proj", "v_proj"],
            target_mlp=["fc1", "fc2"],
        ),
    ),
}


# ── Additions: explicit layer lists and other ViT sizes ──────────────
#
# Everything above is the original ViT-B registry and is left untouched. The
# entries below are generated so that names cannot disagree with their data:
# a layer-specific experiment's name is derived from its `layers` list.

_ATTN_QKV = ["q_proj", "k_proj", "v_proj"]
_MLP = ["fc1", "fc2"]


def _layers_tag(layers: Optional[List[int]]) -> str:
    """'' for all layers; '0-5' for a contiguous run; '0_2_4' otherwise."""
    if layers is None:
        return ""
    ls = sorted(layers)
    if len(ls) > 1 and ls == list(range(ls[0], ls[-1] + 1)):
        return f"_L{ls[0]}-{ls[-1]}"
    return "_L" + "_".join(str(i) for i in ls)


def _add_exp(name: str, *, model_name: str, path: str, arch: str, lr: float = 1e-4,
             rank: Optional[int] = None, attn: Optional[List[str]] = None,
             mlp: Optional[List[str]] = None, layers: Optional[List[int]] = None,
             implementations: Optional[List[str]] = None) -> None:
    if name in EXPERIMENTS:
        raise ValueError(f"duplicate experiment name {name!r}")
    lora = None
    if rank is not None:
        lora = LoRAExperimentConfig(rank=rank, alpha=2 * rank, target_attention=list(attn or []),
                                    target_mlp=list(mlp or []), layers=layers)
    extra = {} if implementations is None else {"implementations": list(implementations)}
    EXPERIMENTS[name] = ExperimentConfig(name=name, model_name=model_name,
                                         model_local_path=path, lr=lr, lora=lora, arch=arch, **extra)


_B = dict(model_name="google/vit-base-patch16-224", path=VIT_BASE_PATH, arch="vitb")
_L = dict(model_name="google/vit-large-patch16-224", path=VIT_LARGE_PATH, arch="vitl")

# ViT-B: q/v only, and attention r8 on explicit block subsets (0-indexed, 12 blocks).
# Four lists of equal size (6 blocks) that separate "which blocks" from "how many":
# first half, second half, even blocks, odd blocks.
_add_exp("lora_qv_r8", **_B, rank=8, attn=["q_proj", "v_proj"])
for _layers in ([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]):
    _add_exp(f"lora_attn_r8{_layers_tag(_layers)}", **_B, rank=8, attn=_ATTN_QKV, layers=_layers)

# ViT-L (24 blocks, same pretraining family as ViT-B). The full-fine-tuning learning
# rate is NOT assumed to transfer from ViT-B, hence a small sweep.
for _lr, _tag in ((1e-5, "1e-5"), (3e-5, "3e-5"), (1e-4, "1e-4")):
    _add_exp(f"vitl_baseline_lr{_tag}", **_L, lr=_lr)
_add_exp("vitl_lora_attn_r8", **_L, rank=8, attn=_ATTN_QKV)
_add_exp("vitl_lora_mlp_r8", **_L, rank=8, mlp=_MLP)
_add_exp("vitl_lora_attn_mlp_r8", **_L, rank=8, attn=_ATTN_QKV, mlp=_MLP)
_add_exp(f"vitl_lora_attn_r8{_layers_tag(list(range(12, 24)))}", **_L, rank=8, attn=_ATTN_QKV,
         layers=list(range(12, 24)))

# DeiT-B: the same architecture as ViT-B with a different pretraining recipe (ImageNet-1k only,
# no 21k stage), so it runs through the same paired pipeline, scratch implementation included.
_D = dict(model_name="facebook/deit-base-patch16-224", path=DEIT_BASE_PATH, arch="deitb")
for _lr, _tag in ((1e-5, "1e-5"), (3e-5, "3e-5"), (1e-4, "1e-4")):
    _add_exp(f"deitb_baseline_lr{_tag}", **_D, lr=_lr)
_add_exp("deitb_lora_attn_r8", **_D, rank=8, attn=_ATTN_QKV)
_add_exp("deitb_lora_mlp_r8", **_D, rank=8, mlp=_MLP)
_add_exp("deitb_lora_attn_mlp_r8", **_D, rank=8, attn=_ATTN_QKV, mlp=_MLP)
_add_exp(f"deitb_lora_attn_r8{_layers_tag(list(range(6, 12)))}", **_D, rank=8, attn=_ATTN_QKV,
         layers=list(range(6, 12)))

# Swin-B (hierarchical, windowed attention): LIBRARY ONLY. There is no scratch Swin, so these
# runs have no paired comparison and no pairing gate; models.library_only_checks stands in.
# `layers` addresses its 24 blocks (stage depths 2,2,18,2) as one running 0-indexed sequence.
_S = dict(model_name="microsoft/swin-base-patch4-window7-224", path=SWIN_BASE_PATH, arch="swinb",
          implementations=["library"])
for _lr, _tag in ((1e-5, "1e-5"), (3e-5, "3e-5"), (1e-4, "1e-4")):
    _add_exp(f"swinb_baseline_lr{_tag}", **_S, lr=_lr)
_add_exp("swinb_lora_attn_r8", **_S, rank=8, attn=_ATTN_QKV)
_add_exp("swinb_lora_mlp_r8", **_S, rank=8, mlp=_MLP)
_add_exp("swinb_lora_attn_mlp_r8", **_S, rank=8, attn=_ATTN_QKV, mlp=_MLP)
# Attention r8 on block subsets, the same four-way design as ViT-B: first half, second half, even
# blocks, odd blocks. The halves are not balanced in Swin (later blocks are wider: 233,472 vs 344,064
# adapter parameters); even and odd take one block of every pair in every stage (288,768 each).
for _layers in (list(range(0, 12)), list(range(12, 24)), list(range(0, 24, 2)), list(range(1, 24, 2))):
    _add_exp(f"swinb_lora_attn_r8{_layers_tag(_layers)}", **_S, rank=8, attn=_ATTN_QKV, layers=_layers)

# Learning-rate check: full FT below the earlier range (its best, 1e-5, was the smallest tried) and
# LoRA either side of 1e-4 (the only rate run before). The rate is chosen on validation loss.
# ViT-B entries are run library-only (--impl library); the question is the rate, not the implementation.
_add_exp("swinb_baseline_lr3e-6", **_S, lr=3e-6)
_add_exp("baseline_lr3e-6", **_B, lr=3e-6)
for _lr, _tag in ((3e-5, "3e-5"), (3e-4, "3e-4"), (1e-3, "1e-3")):
    _add_exp(f"swinb_lora_attn_r8_lr{_tag}", **_S, lr=_lr, rank=8, attn=_ATTN_QKV)
    _add_exp(f"lora_attn_r8_lr{_tag}", **_B, lr=_lr, rank=8, attn=_ATTN_QKV)

# Memory-budget study (2026-11-10): attention r8 on every block from a first adapted block k to the
# last ("suffix" placements). Peak training memory is set mainly by k (backpropagation must reach it), so
# these sample the memory and accuracy curves between the measured k = 0, 1 and 6 (ViT-B/DeiT-B) and
# k = 0, 1 and 12 (Swin-B). Most are profiled only (no training); DeiT-B k = 3 and ViT-B k = 3 are trained.
for _k in (1, 2, 3, 4, 5, 8, 10):
    _add_exp(f"deitb_lora_attn_r8{_layers_tag(list(range(_k, 12)))}", **_D, rank=8, attn=_ATTN_QKV,
             layers=list(range(_k, 12)))
_add_exp(f"lora_attn_r8{_layers_tag(list(range(3, 12)))}", **_B, rank=8, attn=_ATTN_QKV,
         layers=list(range(3, 12)))
# ViT-B at the other start blocks (2026-11-17), profiled only, so that the memory-against-start-block figure
# shows ViT-B itself (k = 3 and 6 are registered above and earlier).
for _k in (1, 2, 4, 5, 8, 10):
    _add_exp(f"lora_attn_r8{_layers_tag(list(range(_k, 12)))}", **_B, rank=8, attn=_ATTN_QKV,
             layers=list(range(_k, 12)))
for _k in (2, 4, 8, 16, 22):      # 2, 4, 22: first block of stages 2, 3, 4
    _add_exp(f"swinb_lora_attn_r8{_layers_tag(list(range(_k, 24)))}", **_S, rank=8, attn=_ATTN_QKV,
             layers=list(range(_k, 24)))

# ViT-H is deliberately NOT registered. hf_vit/vit_huge is a bare ViTModel checkpoint
# (keys without the "vit." prefix) and this transformers version then leaves the whole
# backbone randomly initialized; models.from_pretrained_checked refuses such a load.
# Using it needs a prefix-remapping loader first. It is also patch 14 / 21k-only, so not
# like-for-like with B and L.
