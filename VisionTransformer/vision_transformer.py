import torch
import torch.nn as nn

class ViTConfig:
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        num_labels=10,
        debug=False
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.num_labels = num_labels
        self.debug = debug
        self.num_patches = (image_size // patch_size) ** 2


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches = config.num_patches
        self.projection = nn.Conv2d(
            self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.debug = config.debug
    def forward(self, pixel_values):
        if self.debug:
            print("[ViTPatchEmbeddings] PatchEmbeddings input:", pixel_values.shape)
        batch_size, num_channels, height, width = pixel_values.shape

        if(num_channels!=self.num_channels):
            raise ValueError(f"Expected {self.num_channels} channels, got {num_channels}")

        if(height!=self.image_size or width!=self.image_size):
            raise ValueError(f"Input size ({height}x{width}) does not match the expect size of {self.image_size}x{self.image_size}")

        pixel_values = self.projection(pixel_values) # (batch_size, hidden_size, H/patch_size, W/patch_size)
        # (batch_size, hidden_size, H/patch_size, W/patch_size) -> (batch_size, hidden_size, (HW)/patch_size^2)
        # (batch_size, hidden_size, (HW)/patch_size^2) -> (batch_size, (HW)/patch_size^2), hidden_size)
        if self.debug:
            print("[ViTPatchEmbeddings] After patch projection:", pixel_values.shape)
        pixel_values = pixel_values.flatten(2).transpose(1, 2)
        if self.debug:
            print("[ViTPatchEmbeddings] After flatten + transpose:", pixel_values.shape)
            print("\n==================\n")
        return pixel_values

class ViTEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.patch_embeddings = ViTPatchEmbeddings(config=config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob, inplace=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1+self.patch_embeddings.num_patches, config.hidden_size))
        self.debug = config.debug

    def forward(self, pixel_values):
        x = self.patch_embeddings(pixel_values)
        if self.debug:
            print("[ViTEmbeddings] Patch embeddings:", x.shape)
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)
        if self.debug:
            print("[ViTEmbeddings] CLS Token:", tokens_batch.shape)
        x = torch.cat((tokens_batch, x), dim=1)
        if self.debug:
            print("[ViTEmbeddings] After adding CLS:", x.shape)
        x = x + self.position_embeddings
        if self.debug:
            print("[ViTEmbeddings] After adding position embedding:", x.shape)
        x = self.dropout(x)
        if self.debug:
            print("\n==================\n")
        return x

class ViTMLP(nn.Module):
    def __init__(self, config:ViTConfig, debug=False):
        super().__init__()
        self.debug = debug and config.debug
        self.fc1 = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=True)
        self.activation_fn = nn.GELU()

    def forward(self, hidden_states):
        if self.debug:
            print(f"[ViTMLP] fc1 dimensions: {self.fc1.weight.shape}, fc2 dimensions: {self.fc2.weight.shape}")
        hidden_states = self.fc1(hidden_states)
        if self.debug:
            print("[ViTMLP] Dimension after fc1:", hidden_states.shape)
            print("[ViTMLP] Activation function:", self.activation_fn)
        hidden_states = self.activation_fn(hidden_states)
        
        hidden_states = self.fc2(hidden_states)
        if self.debug:
            print("[ViTMLP] Dimension after fc2:", hidden_states.shape)
            print("\n==================\n")

        return hidden_states

class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig, debug=False, layer_idx=-1):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_probs_dropout_prob
        self.scaling = self.head_dim**-0.5
        self.is_causal = False
        self.debug=debug and config.debug

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(self, hidden_states):

        if self.layer_idx>0 and self.debug:
            print(f"[ViTAttnetion Layer: {self.layer_idx}] Dimension information same between layer, skipping prints")
        if self.debug:
            print(f"[ViTAttnetion Layer: {self.layer_idx}] Dimension of W_q = {self.q_proj.weight.shape}, Dimension of W_k = {self.k_proj.weight.shape}, Dimension of W_o = {self.o_proj.weight.shape}")
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if self.debug:
            print(f"[ViTAttnetion Layer: {self.layer_idx}] Q:", query_states.shape)
            print(f"[ViTAttnetion Layer: {self.layer_idx}] K:", key_states.shape)
            print(f"[ViTAttnetion Layer: {self.layer_idx}] V:", value_states.shape)
        query_states = query_states.view(
            query_states.shape[0],
            query_states.shape[1],
            self.num_attention_heads,
            self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            key_states.shape[0],
            key_states.shape[1],
            self.num_attention_heads,
            self.head_dim
        ).transpose(1, 2)

        value_states = value_states.view(
            value_states.shape[0],
            value_states.shape[1],
            self.num_attention_heads,
            self.head_dim
        ).transpose(1, 2)
        if self.debug:
            print(f"[ViTAttnetion Layer: {self.layer_idx}] Q after head split:", query_states.shape)
            print(f"[ViTAttnetion Layer: {self.layer_idx}] K after head split:", key_states.shape)
            print(f"[ViTAttnetion Layer: {self.layer_idx}] V after head split:", value_states.shape)

        # Fused scaled-dot-product attention — the same path the library (HF)
        # implementation uses via sdpa. Computes softmax(QK^T * scale) @ V
        # WITHOUT materializing the per-layer attention matrix, which the eager
        # version did at ~119 MB/layer at batch 64 — that was the ~1.3 GB
        # activation-memory gap between the two implementations. Attention
        # weights are therefore no longer materialized: None is returned for
        # them (matching HF's sdpa path), and ViTLayer — the only caller —
        # discards them. Dropout on attention weights is applied inside the
        # kernel, only while training, matching the previous eager semantics.
        attention_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
        )
        attention_weights = None
        if self.debug:
            print(f"[ViTAttnetion Layer: {self.layer_idx}] Scaling factor:", self.scaling)
        if self.debug:
            print(f"[ViTAttnetion Layer: {self.layer_idx}] Attention output per head:", attention_output.shape)
        attention_output = attention_output.transpose(1, 2).contiguous()

        if self.debug:
            print(f"[ViTAttnetion Layer: {self.layer_idx}] After transpose:", attention_output.shape)
        attention_output = attention_output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        if self.debug:
            print(f"[ViTAttnetion Layer: {self.layer_idx}] After head concatenation:", attention_output.shape)
        attention_output = self.o_proj(attention_output)
        if self.debug:
            print(f"[ViTAttnetion Layer: {self.layer_idx}] After O projection:", attention_output.shape)
            print("\n==================\n")
        return attention_output, attention_weights

class ViTLayer(nn.Module):
    def __init__(self, config: ViTConfig, debug=False, layer_idx=-1):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = ViTAttention(config, debug=debug, layer_idx=layer_idx)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True)
        self.mlp = ViTMLP(config, debug=debug)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob, inplace=False)
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)

        attn_output, _ = self.attention(hidden_states)

        attn_output = self.dropout(attn_output)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + residual

        return hidden_states

class ViTModel(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.embeddings = ViTEmbeddings(config)
        self.layers = nn.ModuleList([
            ViTLayer(config, debug=(i==0), layer_idx=i) for i in range(config.num_hidden_layers)
        ])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=True)
    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.layernorm(hidden_states)
        return hidden_states

class ViTForImageClassification(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(in_features=config.hidden_size, out_features=config.num_labels, bias=True)
        self.debug = config.debug
    def forward(self, image):
        hidden_states = self.vit(image)
        cls_output = hidden_states[:, 0, :]

        logits = self.classifier(cls_output)
        if self.debug:
            print("[ViTForImageClassification] Classifier dimension:", self.classifier.weight.shape)
            print("[ViTForImageClassification] Dimension after passing through classifier:", logits.shape)
            print("\n==================\n")
        return logits
def load_vit_weights(model, hf_state_dict, strict_classifier=False):
    model_state = model.state_dict()
    reinitialized = []
    filtered_state = {}
    ckpt_only = []

    classifier_keys = {"classifier.weight", "classifier.bias"}

    for name, param in hf_state_dict.items():
        if name not in model_state:
            ckpt_only.append(name)
            continue

        # Shape mismatch check
        if param.shape != model_state[name].shape:
            if strict_classifier and name in classifier_keys:
                raise ValueError(
                    f"Classifier mismatch for '{name}': checkpoint shape {tuple(param.shape)} "
                    f"does not match model shape {tuple(model_state[name].shape)}."
                )
            reinitialized.append((name, tuple(param.shape), tuple(model_state[name].shape)))
            continue

        filtered_state[name] = param

    # Load matched parameters
    load_result = model.load_state_dict(filtered_state, strict=False)

    # Separate parameters missing from checkpoint vs. skipped due to shape mismatch
    reinit_names = {r[0] for r in reinitialized}
    truly_missing = [k for k in load_result.missing_keys if k not in reinit_names]

    # Diagnostics
    if reinitialized or truly_missing:
        print(f"Some weights of {model.__class__.__name__} were not initialized from the checkpoint:")
        for name, ckpt_shape, model_shape in reinitialized:
            print(f"  - {name}: checkpoint shape {ckpt_shape} != model shape {model_shape} (reinitialized)")
        for name in truly_missing:
            print(f"  - {name} (missing from checkpoint)")

    if ckpt_only:
        print(f"Unused checkpoint keys ({len(ckpt_only)} keys): {ckpt_only}")

    return model

"""
ViTForImageClassification(
  (vit): ViTModel(
    (embeddings): ViTEmbeddings(
      (patch_embeddings): ViTPatchEmbeddings(
        (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
      )
      (dropout): Dropout(p=0.0, inplace=False)
    )
    (layers): ModuleList(
      (0-11): 12 x ViTLayer(
        (attention): ViTAttention(
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (o_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (mlp): ViTMLP(
          (activation_fn): GELUActivation()
          (fc1): Linear(in_features=768, out_features=3072, bias=True)
          (fc2): Linear(in_features=3072, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
      )
    )
    (layernorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
  )
  (classifier): Linear(in_features=768, out_features=10, bias=True)
)
ViTForImageClassification
    vit: ViTModel
        embeddings: ViTEmbeddings
            patch_embeddings: ViTPatchEmbeddings
                projection
            dropout
        layers: ModuleList
            12xViTLayer
                attention: ViTAttention
                    q_proj
                    k_proj
                    v_proj
                    o_proj
                layernorm_before
                layernorm_after
                mlp: ViTMLP
                    activation_fn
                    fc1
                    fc2
                dropout
        layernorm
    classifier: Linear

"""