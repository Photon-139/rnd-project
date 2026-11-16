"""
Model builder for paired experiments.

HOW THIS FILE WORKS (read top to bottom):

    1. build_library_model(config, seed)
           Builds the HuggingFace model (+ optional PEFT LoRA), then EXTRACTS
           every tensor that must be identical into a flat dict called `ref`.
           Keys of `ref` are scratch-side keys, e.g.
               ("classifier", "weight")
               (7, "q_proj", "lora_A")

    2. build_scratch_model(config, seed, ref)
           Builds the scratch model, then LOADS every tensor in `ref` into it.
           Any key in `ref` that cannot be applied raises immediately.

    3. build_paired_models(config, seed)
           Glue only: builds the library model first (it is the reference),
           then the scratch model.

    4. summarize_init(models)
           Independent verification: re-reads BOTH models and compares every
           transferred tensor. Returns a plain dict report. Never trains.

The `ref` dict is the entire "copying" mechanism. It is an ordinary dict of
tensors, so it can be printed, counted, or saved to disk.

There are no warnings in the paired flow: if anything cannot be found, mapped,
or copied, this file raises. A silently skipped tensor would mean an unpaired
run that looks paired, which is the one failure mode this system must never have.
"""

import re
import hashlib
import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, SwinForImageClassification
from peft import LoraConfig as PeftLoraConfig, get_peft_model

from vision_transformer import ViTForImageClassification as MyViT, ViTConfig, load_vit_weights
from lora import LoRAConfig as ScratchLoRAConfig, LoRATarget, LoRAModuleSpec, LoRALinear, apply_lora
from experiments import (ExperimentConfig, LoRAExperimentConfig, arch_dims, validate_layers, num_blocks,
                         model_family, has_scratch_impl, swin_block_table, trainable_counts)

# Projection names, as they appear inside module paths. The library-side list
# also accepts the query/key/value spellings and canonicalizes them, so both
# naming conventions map onto the same scratch-side keys.
_PROJ_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
_LIB_PROJ_NAMES = _PROJ_NAMES + ["query", "key", "value"]
_CANON = {"query": "q_proj", "key": "k_proj", "value": "v_proj"}


# ─────────────────────────────────────────────────────────────────────
# Config translation (unchanged from the previous version)
# ─────────────────────────────────────────────────────────────────────

def _to_peft_config(lora_cfg: LoRAExperimentConfig, explicit_targets: Optional[list] = None) -> PeftLoraConfig:
    """Convert experiment LoRA config to PEFT LoraConfig.

    explicit_targets (Swin): full module paths, one per adapted linear. Swin's blocks are nested
    (stage -> block), which PEFT's flat `layers_to_transform` cannot address, so the block
    selection is resolved to exact module names beforehand (see _swin_lora_targets)."""
    target_modules = list(lora_cfg.target_attention) + list(lora_cfg.target_mlp)
    extra = {}
    if explicit_targets is not None:
        target_modules = list(explicit_targets)
    elif lora_cfg.layers is not None:
        # Only passed when a subset is requested, so the all-layers path builds
        # exactly the PEFT config it always did.
        extra["layers_to_transform"] = list(lora_cfg.layers)
    return PeftLoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        target_modules=target_modules,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        modules_to_save=["classifier"],
        **extra,
    )


def vit_config_from_checkpoint(model_local_path: str, num_labels: int) -> ViTConfig:
    """Scratch ViTConfig built from the checkpoint's own config.json (strict: a
    missing file is an error, never a silent ViT-B)."""
    d = arch_dims(model_local_path, strict=True)
    return ViTConfig(
        hidden_size=d["hidden_size"], num_hidden_layers=d["num_hidden_layers"],
        num_attention_heads=d["num_attention_heads"], intermediate_size=d["intermediate_size"],
        hidden_act=d["hidden_act"], layer_norm_eps=d["layer_norm_eps"],
        image_size=d["image_size"], patch_size=d["patch_size"], num_channels=d["num_channels"],
        qkv_bias=d["qkv_bias"], num_labels=num_labels,
    )


def hf_classifier_class(model_local_path: str):
    """HF classification head class for the checkpoint's family (DeiT is stored as a ViT)."""
    return SwinForImageClassification if model_family(model_local_path, strict=True) == "swin" \
        else ViTForImageClassification


def from_pretrained_checked(model_local_path: str, **kwargs):
    """ViTForImageClassification.from_pretrained that REFUSES a partially random model.

    HF reports keys it could not map as missing/unexpected and silently random-initializes
    the missing tensors. The only tolerated gap is the classification head (new num_labels).
    A checkpoint whose backbone keys do not map (the ViT-H directory is a bare ViTModel with
    un-prefixed keys) would otherwise train and "work" from random weights."""
    model, info = hf_classifier_class(model_local_path).from_pretrained(
        model_local_path, local_files_only=True, output_loading_info=True, **kwargs)
    missing = sorted(k for k in info["missing_keys"] if not k.startswith("classifier."))
    unexpected = sorted(k for k in info["unexpected_keys"] if not k.startswith("pooler."))
    if missing or unexpected:
        raise RuntimeError(
            f"[Build] FATAL: checkpoint {model_local_path} did not load completely: "
            f"{len(missing)} model tensors missing from the checkpoint (e.g. {missing[:3]}), "
            f"{len(unexpected)} checkpoint tensors unused (e.g. {unexpected[:3]}). "
            f"The backbone would be partly random.")
    return model


def _check_layers(config: ExperimentConfig) -> None:
    """Fail before building anything if the requested LoRA layers do not exist."""
    if config.lora is not None:
        validate_layers(config.lora.layers, num_blocks(arch_dims(config.model_local_path, strict=True)))


def _to_scratch_lora_config(lora_cfg: LoRAExperimentConfig) -> ScratchLoRAConfig:
    """Convert experiment LoRA config to scratch LoRAConfig."""
    targets = []
    if lora_cfg.target_attention or lora_cfg.target_mlp:
        targets.append(
            LoRATarget(
                spec=LoRAModuleSpec(
                    r=lora_cfg.rank,
                    alpha=lora_cfg.alpha,
                    dropout=lora_cfg.dropout,
                ),
                layers="all" if lora_cfg.layers is None else list(lora_cfg.layers),
                attention=list(lora_cfg.target_attention),
                mlp=list(lora_cfg.target_mlp),
            )
        )
    return ScratchLoRAConfig(
        targets=targets,
        bias=lora_cfg.bias,
        modules_to_save=["classifier"],
    )


_SWIN_BLOCK_RE = re.compile(r"layers\.(\d+)\.blocks\.(\d+)\.")
# Tail of a Swin block's linear -> canonical projection name (transformers >= 5 module names).
_SWIN_ROLES = {"attention.q_proj": "q_proj", "attention.k_proj": "k_proj", "attention.v_proj": "v_proj",
               "attention.o_proj": "o_proj", "mlp.fc1": "fc1", "mlp.fc2": "fc2"}


def _swin_lora_targets(model: nn.Module, lora_cfg: LoRAExperimentConfig, depths: list) -> list:
    """Exact module paths of the Swin linears to adapt: the requested projections in the requested
    blocks (global 0-indexed, across stages). Refuses a result of the wrong size, so a transformers
    version that names the modules differently cannot silently adapt nothing (or the wrong thing)."""
    want = set(lora_cfg.target_attention) | set(lora_cfg.target_mlp)
    n_blocks = sum(depths)
    blocks = set(range(n_blocks)) if lora_cfg.layers is None else set(lora_cfg.layers)
    offsets = [sum(depths[:i]) for i in range(len(depths))]
    found = []
    for name, mod in model.named_modules():
        m = _SWIN_BLOCK_RE.search(name)
        if m is None or not isinstance(mod, nn.Linear):
            continue
        role = _SWIN_ROLES.get(name[m.end():])
        if role in want and offsets[int(m.group(1))] + int(m.group(2)) in blocks:
            found.append(name)
    expected = len(blocks) * len(want)
    if len(found) != expected:
        raise RuntimeError(f"[Build] FATAL: resolved {len(found)} Swin LoRA target modules, expected "
                           f"{expected} ({len(blocks)} blocks x {sorted(want)}). The module names of this "
                           f"transformers version are not the ones this code knows ({sorted(_SWIN_ROLES)}).")
    return found


def _depths_of(model: nn.Module) -> Optional[list]:
    """Swin stage depths of a (possibly PEFT-wrapped) model, or None for a non-hierarchical one."""
    for getter in (lambda: model.config, lambda: model.get_base_model().config):
        try:
            depths = getattr(getter(), "depths", None)
        except Exception:
            continue
        if depths:
            return list(depths)
    return None


# ─────────────────────────────────────────────────────────────────────
# Key mapping helpers
#
# A "key" identifies one transferred tensor:
#     ("classifier", "weight" | "bias")
#     (layer_index, projection_name, "lora_A" | "lora_B")
# ─────────────────────────────────────────────────────────────────────

def _param_key(name: str, depths: Optional[list] = None):
    """Map a LIBRARY parameter path to a transfer key, or None.

    Transferred tensors are exactly: the classifier head, and LoRA A/B matrices.
    Everything else (backbone, layernorms, ...) returns None and is never
    transferred. An unmappable LoRA or classifier parameter is FATAL: silently
    skipping such a parameter is how an unpaired run sneaks through.
    """
    if ".lora_A." in name or ".lora_B." in name:
        side = "lora_A" if ".lora_A." in name else "lora_B"
        layer = re.search(r"layer[s]?\.(\d+)", name)
        swin = _SWIN_BLOCK_RE.search(name)
        proj = next((p for p in _LIB_PROJ_NAMES if p in name.split(".")), None)
        if layer is None or proj is None:
            raise RuntimeError(
                f"[Build] FATAL: cannot map library LoRA parameter '{name}' "
                f"(layer={layer}, proj={proj})."
            )
        if swin is not None:
            # "layers.N" is the STAGE here; the transfer key uses the global block index.
            if not depths:
                raise RuntimeError(f"[Build] FATAL: '{name}' is a Swin block but no stage depths were given.")
            return sum(depths[:int(swin.group(1))]) + int(swin.group(2)), _CANON.get(proj, proj), side
        return int(layer.group(1)), _CANON.get(proj, proj), side

    if "classifier" in name and "lora" not in name:
        if name.endswith("weight"):
            return ("classifier", "weight")
        if name.endswith("bias"):
            return ("classifier", "bias")
        raise RuntimeError(f"[Build] FATAL: unexpected classifier parameter name '{name}'.")

    return None


def _scratch_module_key(name: str):
    """Map a SCRATCH module path to (layer_index, projection_name), or None."""
    layer = re.search(r"layer[s]?\.(\d+)", name)
    proj = next((p for p in _PROJ_NAMES if p in name.split(".")), None)
    if layer is None or proj is None:
        return None
    return int(layer.group(1)), proj


# ─────────────────────────────────────────────────────────────────────
# STEP 1: build the LIBRARY model and snapshot the reference tensors
# ─────────────────────────────────────────────────────────────────────

def _extract_reference_tensors(model: nn.Module) -> dict:
    """Snapshot every tensor the scratch model must share, into a flat dict.

    PEFT's modules_to_save wrapper stores the head twice (the wrapped copy and
    the original module). When both are present the modules_to_save copy wins,
    because that is the copy training actually updates.
    """
    ref = {}
    depths = _depths_of(model)
    for name, param in model.named_parameters():
        key = _param_key(name, depths)
        if key is None:
            continue
        # PEFT stores the head twice under modules_to_save: the wrapped
        # trainable copy and the frozen original. The wrapped copy must win.
        # This condition achieves that in EITHER traversal order: a later
        # non-wrapped copy is skipped (key already set), and a wrapped copy
        # always overwrites (the 'modules_to_save' guard never skips it).
        if key[0] == "classifier" and key in ref and "modules_to_save" not in name:
            continue
        ref[key] = param.detach().clone()
    return ref


def build_library_model(config: ExperimentConfig, seed: int):
    """Build the library (HuggingFace + optional PEFT) model.

    Returns (model, ref) where `ref` is the flat snapshot the scratch model
    will be initialized from.
    """
    _check_layers(config)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = from_pretrained_checked(
        config.model_local_path,
        num_labels=config.num_labels,
        ignore_mismatched_sizes=True,
    )
    if config.lora:
        explicit = None
        if model_family(config.model_local_path) == "swin":
            explicit = _swin_lora_targets(model, config.lora, list(model.config.depths))
        model = get_peft_model(model, _to_peft_config(config.lora, explicit))

    ref = _extract_reference_tensors(model)
    n_head = sum(1 for k in ref if k[0] == "classifier")
    n_lora = sum(1 for k in ref if isinstance(k[0], int))

    if not ref:
        raise RuntimeError("[Build] FATAL: extracted zero reference tensors from library model.")
    if n_head != 2:
        raise RuntimeError(
            f"[Build] FATAL: expected exactly 2 classifier tensors (weight, bias) in the "
            f"reference snapshot, found {n_head}. The library head was not found under the "
            f"expected parameter naming."
        )
    if config.lora and n_lora == 0:
        raise RuntimeError("[Build] FATAL: LoRA config produced zero LoRA parameters in library model.")
    if not config.lora and n_lora != 0:
        raise RuntimeError("[Build] FATAL: no LoRA configured, but library model exposes LoRA parameters.")

    print(f"[Build] Library model built. Reference snapshot: {len(ref)} tensors "
          f"({n_head} classifier, {n_lora} LoRA A/B).")
    return model, ref


# ─────────────────────────────────────────────────────────────────────
# STEP 2: build the SCRATCH model and load the reference tensors
# ─────────────────────────────────────────────────────────────────────

def _load_reference_tensors(model: nn.Module, ref: dict) -> None:
    """Copy every tensor in `ref` into the scratch model. Missing keys are FATAL."""
    applied = set()

    # --- classifier head: direct attribute access, checkable in one glance ---
    if ("classifier", "weight") in ref or ("classifier", "bias") in ref:
        if not hasattr(model, "classifier"):
            raise RuntimeError("[Build] FATAL: ref has a classifier head but the scratch model has none.")
        if ("classifier", "weight") not in ref or ("classifier", "bias") not in ref:
            raise RuntimeError("[Build] FATAL: ref must contain both classifier weight and bias.")
        for which in ("weight", "bias"):
            key = ("classifier", which)
            target = getattr(model.classifier, which)
            if target.shape != ref[key].shape:
                raise RuntimeError(
                    f"[Build] FATAL: classifier {which} shape mismatch: "
                    f"scratch {tuple(target.shape)} vs library {tuple(ref[key].shape)}."
                )
        model.classifier.weight.data.copy_(ref[("classifier", "weight")])
        model.classifier.bias.data.copy_(ref[("classifier", "bias")])
        applied.add(("classifier", "weight"))
        applied.add(("classifier", "bias"))

    # --- LoRA tensors: walk every LoRALinear in the scratch model ---
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        stem = _scratch_module_key(name)
        if stem is None:
            raise RuntimeError(f"[Build] FATAL: cannot map scratch LoRA module '{name}'.")
        for side in ("lora_A", "lora_B"):
            key = (stem[0], stem[1], side)
            if key not in ref:
                raise RuntimeError(
                    f"[Build] FATAL: no reference tensor for {key} (scratch module '{name}')."
                )
            target = getattr(module, side).weight
            if target.shape != ref[key].shape:
                raise RuntimeError(
                    f"[Build] FATAL: shape mismatch for {key}: "
                    f"scratch {tuple(target.shape)} vs library {tuple(ref[key].shape)}."
                )
            target.data.copy_(ref[key])
            applied.add(key)

    # --- final accounting: every ref key must have been applied exactly once ---
    unapplied = set(ref.keys()) - applied
    if unapplied:
        raise RuntimeError(
            f"[Build] FATAL: {len(unapplied)} reference tensors were never applied, "
            f"e.g. {sorted(unapplied, key=str)[:5]}."
        )

    print(f"[Build] Matched init loaded into scratch: {len(applied)} tensors "
          f"({sum(1 for k in applied if k[0] == 'classifier')} classifier, "
          f"{sum(1 for k in applied if isinstance(k[0], int))} LoRA A/B).")


def build_scratch_model(config: ExperimentConfig, seed: int, ref: Optional[dict]) -> nn.Module:
    """Build the scratch ViT and load matched initialization from `ref`.

    ref=None means no library model was built. That is only acceptable without
    LoRA (a debugging smoke-test); a LoRA run without a reference would train
    adapters from unpaired random init, so it is refused.
    """
    if not has_scratch_impl(config.model_local_path):
        raise RuntimeError(
            f"[Build] FATAL: {config.name!r} uses a {model_family(config.model_local_path)!r} checkpoint, which "
            f"has no from-scratch implementation. Set implementations=['library'] (run with --impl library).")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Scratch model + pretrained backbone. The classifier head is skipped
    #    here by design (load_vit_weights reinitializes shape-mismatched keys);
    #    the head arrives in step 3 from the library reference.
    _check_layers(config)
    model = MyViT(vit_config_from_checkpoint(config.model_local_path, config.num_labels))
    hf_weights = from_pretrained_checked(config.model_local_path).state_dict()
    load_vit_weights(model, hf_weights)

    # 2. Scratch LoRA. Must run BEFORE loading ref, because the LoRA modules
    #    do not exist until this call creates them.
    if config.lora:
        apply_lora(model, _to_scratch_lora_config(config.lora))

    # 3. Load matched initialization (classifier head + LoRA A/B).
    if ref is None:
        if config.lora:
            raise RuntimeError(
                "[Build] FATAL: LoRA run requires a library reference for matched init, but no "
                "library model was built. This happens when the experiment config does not "
                "include 'library' in its implementations list. Run with --impl paired, or add "
                "'library' to the config's implementations."
            )
        print("[Build] WARNING: scratch-only run (no library reference). Classifier head "
              "is NOT matched to any library model — debugging use only.")
        return model

    _load_reference_tensors(model, ref)
    return model


# ─────────────────────────────────────────────────────────────────────
# STEP 3: glue
# ─────────────────────────────────────────────────────────────────────

def build_paired_models(config: ExperimentConfig, seed: int) -> dict[str, nn.Module]:
    """Build the requested models with matched initialization.

    The library model is ALWAYS built first: it is the reference the scratch
    model's initialization is taken from. If only the scratch model is
    requested there is no reference, and build_scratch_model refuses LoRA
    configs in that case.
    """
    models: dict[str, nn.Module] = {}
    ref = None

    if "library" in config.implementations:
        models["library"], ref = build_library_model(config, seed)

    if "scratch" in config.implementations:
        models["scratch"] = build_scratch_model(config, seed, ref)

    return models


# ─────────────────────────────────────────────────────────────────────
# STEP 4: independent verification
# ─────────────────────────────────────────────────────────────────────

def _scratch_key_tensors(model: nn.Module) -> dict:
    """Read the transferred tensors back out of a SCRATCH model, by key.

    Deliberately implemented differently from the library-side extraction: the
    scratch side is anchored on module types (isinstance LoRALinear) and direct
    attributes, so a name-mapping bug cannot hide inside one shared helper.
    """
    out = {}
    if hasattr(model, "classifier"):
        out[("classifier", "weight")] = model.classifier.weight.detach()
        out[("classifier", "bias")] = model.classifier.bias.detach()
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        stem = _scratch_module_key(name)
        if stem is None:
            raise RuntimeError(f"[Verify] FATAL: cannot map scratch LoRA module '{name}'.")
        out[(stem[0], stem[1], "lora_A")] = module.lora_A.weight.detach()
        out[(stem[0], stem[1], "lora_B")] = module.lora_B.weight.detach()
    return out


def summarize_init(models: dict) -> dict:
    """Verify that the two implementations hold identical tensors.

    Re-reads BOTH models (no reuse of the load-time data) and compares every
    transferred tensor: classifier head + all LoRA A/B. Pure function — never
    modifies the models. Returns a plain dict report:

        compared / equal        how many tensors were compared and matched
        max_abs_diff            largest absolute difference found
        mismatched / *_only     details of anything that did not line up
        ok                      True only if everything matched exactly
    """
    report = {
        "compared": 0,
        "equal": 0,
        "mismatched": [],
        "in_library_only": [],
        "in_scratch_only": [],
        "max_abs_diff": 0.0,
        "ok": False,
    }
    if "library" not in models or "scratch" not in models:
        report["note"] = "single-implementation run; nothing to compare"
        return report

    lib = _extract_reference_tensors(models["library"])
    scr = _scratch_key_tensors(models["scratch"])

    for key in sorted(lib.keys(), key=str):
        if key not in scr:
            report["in_library_only"].append(str(key))
            continue
        a = lib[key]
        b = scr[key]
        report["compared"] += 1
        if a.shape != b.shape:
            report["mismatched"].append({
                "key": str(key), "reason": "shape",
                "library": tuple(a.shape), "scratch": tuple(b.shape),
            })
            continue
        diff = float((a - b).abs().max()) if a.numel() > 0 else 0.0
        report["max_abs_diff"] = max(report["max_abs_diff"], diff)
        if torch.equal(a, b):
            report["equal"] += 1
        else:
            report["mismatched"].append({"key": str(key), "max_abs_diff": diff})

    report["in_scratch_only"] = [str(k) for k in scr.keys() if k not in lib]
    report["ok"] = (
        report["compared"] > 0
        and report["equal"] == report["compared"]
        and not report["in_library_only"]
        and not report["in_scratch_only"]
    )
    return report


def trainable_signature(model: nn.Module) -> dict:
    """Fingerprint of a model's TRAINABLE parameter set.

    The paired comparison is only fair if both implementations optimize the
    SAME parameters (e.g. PEFT and apply_lora freezing the same things).

    Deliberately TRAINABLE-ONLY: PEFT's modules_to_save wrapper keeps an extra
    FROZEN copy of the head (original_module) that the scratch model does not
    replicate, so comparing total/frozen counts would reject every LoRA run
    for a benign structural difference. The trainable multiset is what the
    optimizer actually sees.

    The shape multiset is included (not just the hash) so a mismatch prints
    something actionable instead of just two differing hashes.
    """
    trainable = sorted(tuple(p.shape) for p in model.parameters() if p.requires_grad)
    digest = hashlib.sha256(repr(trainable).encode("utf-8")).hexdigest()
    return {
        "trainable_count": len(trainable),
        "trainable_shape_multiset": trainable,
        "trainable_shapes_hash": digest,
    }


def check_forward_equivalence(lib: nn.Module, scr: nn.Module, seed: int, device, tol: float) -> dict:
    """Compare the two implementations' OUTPUTS on one fixed input, eval mode.

    summarize_init proves the PARAMETERS are identical; this proves the two
    codebases compute the same FUNCTION. A small non-zero difference is
    expected and acceptable (mathematically equal, different kernel order);
    a large difference means a structural mismatch (wrong tensor placement,
    divergent forward code) and must stop the run.

    Uses a DEDICATED CPU generator: the global RNG must stay untouched because
    the training loop's paired batch shuffle depends on it (train_utils.py:171).
    """
    report = {
        "forward_max_abs_diff": 0.0,
        "forward_ok": False,
        "tolerance": tol,
    }
    if lib is None or scr is None:
        report["note"] = "single-implementation run; nothing to compare"
        return report

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 3, 224, 224, generator=g).to(device)

    lib_was_training, scr_was_training = lib.training, scr.training
    lib.eval()
    scr.eval()
    try:
        with torch.no_grad():
            out_lib = lib(x)
            out_scr = scr(x)
            if hasattr(out_lib, "logits"):
                out_lib = out_lib.logits
            if hasattr(out_scr, "logits"):
                out_scr = out_scr.logits
            if out_lib.shape != out_scr.shape:
                report["forward_shape_mismatch"] = {
                    "library": tuple(out_lib.shape),
                    "scratch": tuple(out_scr.shape),
                }
                return report
            diff = float((out_lib - out_scr).abs().max())
    finally:
        if lib_was_training:
            lib.train()
        if scr_was_training:
            scr.train()

    report["forward_max_abs_diff"] = diff
    report["forward_ok"] = diff <= tol
    return report


def has_lora_params(model: nn.Module) -> bool:
    """True if the model exposes LoRA adapters (PEFT wrappers or scratch LoRALinear)."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            return True
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            return True
    return False


def check_forward_equivalence_with_lora(lib: nn.Module, scr: nn.Module, seed: int, device, tol: float) -> dict:
    """Forward-equivalence check with every LoRA adapter ACTIVE.

    LoRA B initializes to ZEROS, so the plain forward check cannot see the
    adapter path at all: a wrong alpha/r scaling, a wrong scaling factor, or a
    faulty LoRA forward in the scratch implementation would still pass. This
    variant perturbs every LoRA B to the SAME non-zero random tensor — on DEEP
    COPIES, so the real models are untouched — and then compares outputs. With
    every adapter contributing, the LoRA forward path is now covered.

    One random point per adapter is a spot check, not a proof; it catches
    structural mistakes (missing scaling, wrong composition), not every bug.

    Dedicated CPU generators only: the global RNG must stay untouched because
    the training loop's paired batch shuffle depends on it (train_utils.py:171).
    """
    report = {
        "forward_max_abs_diff": 0.0,
        "forward_ok": False,
        "tolerance": tol,
        "lora_adapters_exercised": 0,
    }
    if lib is None or scr is None:
        report["note"] = "single-implementation run; nothing to compare"
        return report

    lib_c = copy.deepcopy(lib).to(device)
    scr_c = copy.deepcopy(scr).to(device)
    g = torch.Generator().manual_seed(seed + 1)

    # Library side: every LoRA B parameter, by transfer key.
    lib_b: dict = {}
    depths = _depths_of(lib_c)
    for name, param in lib_c.named_parameters():
        key = _param_key(name, depths)
        if key is not None and key[0] != "classifier" and key[2] == "lora_B":
            lib_b[key] = param

    # Scratch side: (layer, proj) -> LoRALinear module.
    scr_lora: dict = {}
    for name, module in scr_c.named_modules():
        if isinstance(module, LoRALinear):
            stem = _scratch_module_key(name)
            if stem is None:
                raise RuntimeError(f"[Verify] FATAL: cannot map scratch LoRA module '{name}'.")
            scr_lora[stem] = module

    if not lib_b or not scr_lora:
        report["note"] = "no LoRA parameters found; LoRA-path check skipped"
        return report

    count = 0
    for key in sorted(lib_b.keys(), key=str):
        stem = (key[0], key[1])
        if stem not in scr_lora:
            raise RuntimeError(
                f"[Verify] FATAL: scratch model has no LoRA module for {stem}, "
                f"but the library model does ({key}). The two adapter sets are not aligned."
            )
        values = torch.randn(lib_b[key].shape, generator=g)
        lib_b[key].data.copy_(values.to(lib_b[key].device))
        scr_lora[stem].lora_B.weight.data.copy_(values.to(scr_lora[stem].lora_B.weight.device))
        count += 1
    report["lora_adapters_exercised"] = count

    lib_c.eval()
    scr_c.eval()
    with torch.no_grad():
        x = torch.randn(4, 3, 224, 224, generator=torch.Generator().manual_seed(seed)).to(device)
        out_lib = lib_c(x)
        out_scr = scr_c(x)
        if hasattr(out_lib, "logits"):
            out_lib = out_lib.logits
        if hasattr(out_scr, "logits"):
            out_scr = out_scr.logits
        if out_lib.shape != out_scr.shape:
            report["forward_shape_mismatch"] = {
                "library": tuple(out_lib.shape),
                "scratch": tuple(out_scr.shape),
            }
            return report
        diff = float((out_lib - out_scr).abs().max())

    del lib_c, scr_c
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    report["forward_max_abs_diff"] = diff
    report["forward_ok"] = diff <= tol
    return report


def library_only_checks(model: nn.Module, config: ExperimentConfig, seed: int, device, tol: float) -> dict:
    """Pre-training gate for a family with NO scratch implementation (Swin).

    There is no second implementation to pair against, so the cross-implementation checks do
    not exist. What can still be proved on the model alone:
      * trainable_ok     the trainable parameters are exactly the analytic expectation
                         (elements and tensors), i.e. LoRA landed on the requested modules and
                         nothing else (or, for full fine-tuning, everything) is trainable
      * identity_ok      with LoRA B = 0 (its init) the adapter path changes nothing
      * merge_ok         with every B perturbed, folding the adapters into the base weights
                         leaves the logits unchanged (within `tol`) and unmerging restores them
      * logits finite and of the expected shape
    Weaker than the paired gate (no second opinion on the function itself); the report says so
    by being a different key, `library_only`, never `forward`."""
    dims = arch_dims(config.model_local_path, strict=True)
    lora = config.to_dict()["lora"]
    exp_elems, exp_tensors = trainable_counts(dims, config.num_labels, lora)
    got_elems = sum(p.numel() for p in model.parameters() if p.requires_grad)
    got_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    report = {
        "trainable": {"elements": got_elems, "expected_elements": exp_elems,
                      "tensors": got_tensors, "expected_tensors": exp_tensors},
        "trainable_ok": (got_elems, got_tensors) == (exp_elems, exp_tensors),
        "tolerance": tol, "ok": False,
    }
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 3, 224, 224, generator=g).to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(x).logits
            report["logits_ok"] = bool(torch.isfinite(out).all()) and tuple(out.shape) == (2, config.num_labels)
            if lora:
                with model.disable_adapter():
                    out_off = model(x).logits
                report["identity_max_abs_diff"] = float((out - out_off).abs().max())
                report["identity_ok"] = report["identity_max_abs_diff"] <= 1e-6
    finally:
        if was_training:
            model.train()
    if lora:
        m = copy.deepcopy(model).to(device).eval()
        gb = torch.Generator().manual_seed(seed + 1)
        n_b = 0
        for name, param in m.named_parameters():
            if ".lora_B." in name:
                param.data.copy_((torch.randn(param.shape, generator=gb) * 0.1).to(param.device))
                n_b += 1
        with torch.no_grad():
            out_u = m(x).logits
            m.merge_adapter()
            out_m = m(x).logits
            m.unmerge_adapter()
            out_r = m(x).logits
            with m.disable_adapter():
                out_base = m(x).logits
        report["merge"] = {"adapters_perturbed": n_b,
                           "adapter_effect_max_abs": float((out_u - out_base).abs().max()),
                           "merged_vs_unmerged_max_abs_diff": float((out_m - out_u).abs().max()),
                           "unmerged_restore_max_abs_diff": float((out_r - out_u).abs().max())}
        report["merge_ok"] = (n_b > 0 and report["merge"]["adapter_effect_max_abs"] > 0
                              and report["merge"]["merged_vs_unmerged_max_abs_diff"] <= tol
                              and report["merge"]["unmerged_restore_max_abs_diff"] <= tol)
        del m
    checks = [report["trainable_ok"], report["logits_ok"]]
    if lora:
        checks += [report["identity_ok"], report["merge_ok"]]
    report["ok"] = all(checks)
    return report


# ─────────────────────────────────────────────────────────────────────
# STEP 5: adapter export / import (canonical, implementation-independent)
#
# The deployable per-task artifact of a LoRA run is exactly the transferred
# tensor set: classifier head + every LoRA A/B. It is saved under string keys
# ("classifier/weight", "7/q_proj/lora_A") so the same file loads into EITHER
# implementation, and its size is the true adapter size for both.
# ─────────────────────────────────────────────────────────────────────

def _key_to_str(key) -> str:
    return "/".join(str(k) for k in key)


def _str_to_key(text: str):
    parts = text.split("/")
    if parts[0] == "classifier":
        return ("classifier", parts[1])
    return (int(parts[0]), parts[1], parts[2])


def adapter_reference(model: nn.Module) -> dict:
    """Tuple-keyed snapshot of the classifier head + LoRA A/B of either implementation."""
    if isinstance(model, MyViT):
        return {k: v.clone() for k, v in _scratch_key_tensors(model).items()}
    return _extract_reference_tensors(model)


def export_adapter(model: nn.Module, path) -> Path:
    """Save the trainable per-task artifact (head + LoRA A/B) to `path`."""
    ref = adapter_reference(model)
    torch.save({_key_to_str(k): v.detach().cpu().clone() for k, v in ref.items()}, path)
    return Path(path)


def load_adapter_file(path) -> dict:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    return {_str_to_key(k): v for k, v in raw.items()}


def apply_adapter_reference(model: nn.Module, ref: dict) -> None:
    """Copy `ref` into a model of either implementation. Every key must land."""
    if isinstance(model, MyViT):
        _load_reference_tensors(model, ref)
        return
    applied = set()
    depths = _depths_of(model)
    for name, param in model.named_parameters():
        key = _param_key(name, depths)
        if key is None or key not in ref:
            continue
        if "original_module" in name:
            continue  # PEFT's frozen duplicate head: never used in forward
        if param.shape != ref[key].shape:
            raise RuntimeError(f"[Adapter] FATAL: shape mismatch for {key}: "
                               f"{tuple(param.shape)} vs {tuple(ref[key].shape)}.")
        param.data.copy_(ref[key])
        applied.add(key)
    missing = set(ref) - applied
    if missing:
        raise RuntimeError(f"[Adapter] FATAL: {len(missing)} adapter tensors not applied to the library "
                           f"model, e.g. {sorted(missing, key=str)[:5]}.")



# ─────────────────────────────────────────────────────────────────────
# STEP 6: merge / unmerge helpers (shared by profile_memory.py and
# merge_study.py). All work on either implementation.
# ─────────────────────────────────────────────────────────────────────

def perturbed_adapter_reference(model: nn.Module, seed: int = 0, b_std: float = 0.02) -> dict:
    """Adapter snapshot with every LoRA B replaced by seeded N(0, b_std^2).

    LoRA B initializes to zeros, which makes merging a no-op. A perturbed B
    exercises the real merge arithmetic without needing a trained adapter
    (valid for memory/latency/math checks; says nothing about trained values).
    """
    ref = adapter_reference(model)
    g = torch.Generator().manual_seed(seed)
    out = {}
    for key in sorted(ref, key=str):
        v = ref[key]
        out[key] = (torch.randn(v.shape, generator=g) * b_std) if key[-1] == "lora_B" else v.clone()
    return out


def lora_base_weights(model: nn.Module) -> dict:
    """{module_name: base weight tensor} for every LoRA-wrapped linear (both implementations)."""
    return {name: m.base_layer.weight for name, m in model.named_modules()
            if hasattr(m, "base_layer") and hasattr(m, "lora_A")}


def merge_all(model: nn.Module) -> None:
    """Fold every adapter into its base weight IN PLACE (reversible with unmerge_all)."""
    if isinstance(model, MyViT):
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.merge()
    else:
        model.merge_adapter()


def unmerge_all(model: nn.Module) -> None:
    if isinstance(model, MyViT):
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.unmerge()
    else:
        model.unmerge_adapter()


def merged_copy(model: nn.Module) -> nn.Module:
    """Deep copy with adapters folded in AND removed: what a merged deployment
    would load (a plain dense ViT, no adapter parameters)."""
    m = copy.deepcopy(model)
    if isinstance(m, MyViT):
        for name, mod in list(m.named_modules()):
            if isinstance(mod, LoRALinear):
                mod.merge()
                m.set_submodule(name, mod.base_layer)
        return m
    return m.merge_and_unload()
