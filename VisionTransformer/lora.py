from dataclasses import dataclass, field
from typing import Literal, Union, List
import torch.nn as nn
import torch
import math

@dataclass(frozen=True)
class LoRAModuleSpec:
    """
    Fully compiled, immutable runtime configuration for a single target module.
    """
    r: int
    alpha: int
    dropout: float = 0.0
    init_lora_weights: bool = True

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError(f"`r` must be positive, got {self.r}.")

        if self.alpha <= 0:
            raise ValueError(f"`alpha` must be positive, got {self.alpha}.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"`dropout` must be between 0 and 1, got {self.dropout}."
            )

    @property
    def scaling(self) -> float:
        return self.alpha / self.r
  

@dataclass
class LoRATarget:
    """
    Describes one group of modules to apply LoRA to, with per-group rank/alpha.

    layers:     "all" or a list of layer indices (e.g. [3, 7]) this target applies to.
    attention:  attention projection names to wrap, e.g. ["q_proj", "v_proj"].
                Valid names: "q_proj", "k_proj", "v_proj", "o_proj".
    mlp:        MLP layer names to wrap, e.g. ["fc1"].
                Valid names: "fc1", "fc2".
    r:          LoRA rank for this group.
    alpha:      LoRA alpha (scaling) for this group.
    """
    spec: LoRAModuleSpec
    layers: Union[Literal["all"], List[int]] = "all"
    attention: List[str] = field(default_factory=list)
    mlp: List[str] = field(default_factory=list)

    _VALID_ATTN = {"q_proj", "k_proj", "v_proj", "o_proj"}
    _VALID_MLP = {"fc1", "fc2"}

    def __post_init__(self):
        if not self.attention and not self.mlp:
            raise ValueError("LoRATarget must specify at least one of `attention` or `mlp` module names.")

        bad_attn = set(self.attention) - self._VALID_ATTN
        if bad_attn:
            raise ValueError(f"Invalid attention module name(s) {bad_attn}; valid options are {self._VALID_ATTN}.")

        bad_mlp = set(self.mlp) - self._VALID_MLP
        if bad_mlp:
            raise ValueError(f"Invalid mlp module name(s) {bad_mlp}; valid options are {self._VALID_MLP}.")
        
        if self.layers != "all":
            if not isinstance(self.layers, list) or not all(isinstance(i, int) for i in self.layers):
                raise ValueError(f"`layers` must be \"all\" or a list of ints, got {self.layers!r}.")


@dataclass
class LoRAConfig:
    """
    Top-level LoRA configuration.

    targets:        list of LoRATarget groups, each with their own layers/modules/r/alpha.
    lora_dropout:   dropout applied inside each LoRA adapter.
    bias:           "none"      -> no biases are trainable
                    "all"       -> all biases in the model are trainable
                    "lora_only" -> only biases on LoRA-wrapped modules are trainable
    modules_to_save: modules kept fully trainable (not LoRA-wrapped, not frozen) and
                      included whole in the saved checkpoint. Typically the classifier
                      head after changing num_labels.
    """
    targets: List[LoRATarget]
    bias: Literal["none", "all", "lora_only"] = "none"
    modules_to_save: List[str] = field(default_factory=lambda: ["classifier"])

    def __post_init__(self):
        if not self.targets:
            raise ValueError("LoRAConfig requires at least one LoRATarget in `targets`.")


def resolve_targets(
    model,
    config: LoRAConfig,
    layer_container_path: str = "vit.layers",
    attention_attr: str = "attention",
    mlp_attr: str = "mlp",
    on_conflict: Literal["raise", "override", "warn"] = "raise",
) -> tuple[dict[str, LoRAModuleSpec], dict[str, bool]]:
    """
    Returns:
      module_specs: {module_path: LoRAModuleSpec} for every LoRA-wrapped module.
      bias_trainable: {module_path: bool} for every nn.Linear in the model
                       (wrapped or not) indicating whether its bias should train,
                       derived from config.bias.
    """
    layer_container = model.get_submodule(layer_container_path)
    num_layers = len(layer_container)
    module_specs: dict[str, LoRAModuleSpec] = {}

    def _add(path: str, spec: LoRAModuleSpec):
        submodule = model.get_submodule(path)  # raises AttributeError if path is wrong
        if not isinstance(submodule, nn.Linear):
            raise TypeError(
                f"Target module '{path}' is a {type(submodule).__name__}, not nn.Linear. "
                f"LoRA can only be applied to nn.Linear layers."
            )
        if path in module_specs:
            if on_conflict == "raise":
                raise ValueError(
                    f"Conflicting LoRATarget entries both resolve to '{path}' "
                    f"(existing: r={module_specs[path].r}, alpha={module_specs[path].alpha}; "
                    f"new: r={spec.r}, alpha={spec.alpha}). "
                    f"Remove the overlap or pass on_conflict=\"override\"/\"warn\"."
                )
            elif on_conflict == "warn":
                print(f"Warning: '{path}' targeted by multiple LoRATarget entries; using r={spec.r}, alpha={spec.alpha}.")
        module_specs[path] = spec

    for target in config.targets:
        layer_idxs = range(num_layers) if target.layers == "all" else target.layers

        for i in layer_idxs:
            if not (0 <= i < num_layers):
                raise ValueError(f"Layer index {i} out of range for model with {num_layers} layers.")

            for name in target.attention:
                _add(f"{layer_container_path}.{i}.{attention_attr}.{name}", target.spec)

            for name in target.mlp:
                _add(f"{layer_container_path}.{i}.{mlp_attr}.{name}", target.spec)

    modules_to_save_resolved = set()

    for name in config.modules_to_save:
        model.get_submodule(name)
        modules_to_save_resolved.add(name)
    # Derive bias trainability for every nn.Linear in the model, per config.bias policy.
    bias_trainable: dict[str, bool] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.bias is not None and name not in modules_to_save_resolved:
            if config.bias == "all":
                bias_trainable[name] = True
            elif config.bias == "lora_only":
                bias_trainable[name] = name in module_specs
            else:  # "none"
                bias_trainable[name] = False

    return module_specs, bias_trainable

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, spec: LoRAModuleSpec, train_bias: bool = False):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                f"LoRALinear requires nn.Linear, got "
                f"{type(base_layer).__name__}"
            )

        self.base_layer = base_layer

        self.r = spec.r
        self.alpha = spec.alpha
        self.scaling = spec.scaling

        self.dropout = nn.Dropout(spec.dropout)

        self.lora_A = nn.Linear(
            base_layer.in_features, 
            self.r,
            bias=False
        )
        self.lora_B = nn.Linear(
            self.r,
            base_layer.out_features,
            bias=False
        )

        self.merged = False

        self.base_layer.weight.requires_grad = False

        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = train_bias

        if spec.init_lora_weights:
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @torch.no_grad()
    def merge(self):
        if self.merged:
            return
        self.base_layer.weight.add_(self.scaling*(self.lora_B.weight @ self.lora_A.weight))
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        self.base_layer.weight.sub_(self.scaling*(self.lora_B.weight @ self.lora_A.weight))
        self.merged = False

    def forward(self, x):
        base_output = self.base_layer(x)
        if self.merged:
            return base_output

        lora_output = self.lora_B(
            self.lora_A(
                self.dropout(x)
            )
        )

        return base_output + self.scaling*lora_output

def apply_lora(model: nn.Module, config: LoRAConfig):
    module_specs, bias_trainable = resolve_targets(model, config)

    # Freeze everything first.
    for param in model.parameters():
        param.requires_grad = False

    # Replace targeted Linear layers with LoRALinear.
    for path, spec in module_specs.items():
        base_layer = model.get_submodule(path)
        train_bias = bias_trainable.get(path, False)

        lora_layer = LoRALinear(
            base_layer=base_layer,
            spec=spec,
            train_bias=train_bias,
        )

        model.set_submodule(path, lora_layer)

    # Apply the global bias policy.
    for name, trainable in bias_trainable.items():
        module = model.get_submodule(name)

        if isinstance(module, LoRALinear):
            module = module.base_layer

        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear at '{name}', got {type(module).__name__}"
            )

        if module.bias is not None:
            module.bias.requires_grad = trainable

    # Make modules_to_save fully trainable.
    for name in config.modules_to_save:
        module = model.get_submodule(name)

        for param in module.parameters():
            param.requires_grad = True

    return module_specs, bias_trainable