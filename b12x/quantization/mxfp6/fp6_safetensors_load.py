"""Load a b12x MX-FP6 (W6A6) safetensors checkpoint into kernel-ready weights.

Read side of :mod:`b12x.quantization.mxfp6.fp6_safetensors_export`. Given
the per-expert ModelOpt-mirror keys (``.weight`` / ``.weight_scale`` /
``.weight_scale_2`` / ``.input_scale``), this reconstructs the exact
:class:`~b12x.quantization.mxfp6.fp6_moe_weights.FP6MoEWeights` layout
that the fused FP6 MoE path consumes:

* FC1 codes are row-stacked ``[up; gate]`` (the kernel's gated-silu convention).
* Block scales are stored **unswizzled** on disk and swizzled here via the same
  :func:`swizzle_block_scale` used by the offline quantizer, so the result is
  byte-identical to the validated ``.pt`` path.

Because the per-row FP6 packing and per-block UE8M0 scale are computed
independently, quantizing ``gate``/``up`` separately on disk and re-stacking here
reproduces the single-matrix offline path bit-for-bit.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from b12x._lib.intrinsics import swizzle_block_scale
from .fp6_checkpoint import (
    INPUT_SCALE_SUFFIX,
    WEIGHT_SCALE_2_SUFFIX,
    WEIGHT_SCALE_SUFFIX,
    WEIGHT_SUFFIX,
    activation_format_for_source,
    weight_format_for_source,
)
from .fp6_dense_weights import FP6DenseWeight
from .fp6_moe_weights import FP6MoEWeights

_UNIT_TOL = 1e-3


def _swizzle_stacked(unswizzled: torch.Tensor) -> torch.Tensor:
    """Swizzle stacked ``(E, rows, blocks)`` UE8M0 bytes into the cutlass layout.

    Identical call to the offline ``_swizzled_block_scales`` (minus the block-max
    step, since the bytes already hold the UE8M0 exponents), so the output matches
    the ``.pt`` artifact byte-for-byte.
    """
    return swizzle_block_scale(unswizzled.view(torch.float8_e8m0fnu)).view(torch.uint8)


def _source_format_from_config(quant_config: dict) -> str:
    """Validate checkpoint identity and return its runtime format selector."""
    if not isinstance(quant_config, dict):
        raise TypeError("quantization_config must be a dictionary")
    required_identity = {
        "quant_method": "modelopt",
        "quant_algo": "W6A6",
        "group_size": 32,
        "scale_dtype": "uint8_ue8m0",
    }
    for key, expected in required_identity.items():
        if key not in quant_config:
            raise ValueError(f"quantization_config is missing required {key!r}")
        if quant_config[key] != expected:
            raise ValueError(
                f"quantization_config {key!r} must be {expected!r}, "
                f"got {quant_config[key]!r}"
            )
    for key in ("weight_format", "activation_format"):
        if key not in quant_config:
            raise ValueError(f"quantization_config is missing required {key!r}")
        if not isinstance(quant_config[key], str):
            raise TypeError(f"quantization_config {key!r} must be a string")
    wf = quant_config["weight_format"].lower()
    af = quant_config["activation_format"].lower()
    if wf == "e2m3" and af == "e3m2":
        return "mxfp6_default"
    if wf == "e2m3" and af == "e2m3":
        return "mxfp6_e2m3"
    if wf == "e3m2" and af == "e3m2":
        return "mxfp6_e3m2"
    if wf == "e2m3" and af == "e4m3":
        return "mxfp6_w6a8"
    raise ValueError(
        "unsupported FP6 quantization format combination: "
        f"weight_format={wf!r}, activation_format={af!r}"
    )


def _validate_fp6_weight_pair(
    name: str,
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[int, int]:
    """Validate an encoded linear before swizzle, concatenation, or transfer."""
    if not isinstance(packed, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name} weight and weight_scale must be tensors")
    if packed.dtype != torch.uint8:
        raise TypeError(f"{name}.weight must be uint8, got {packed.dtype}")
    if scale.dtype != torch.uint8:
        raise TypeError(f"{name}.weight_scale must be uint8, got {scale.dtype}")
    if packed.ndim != 2 or not packed.is_contiguous():
        raise ValueError(
            f"{name}.weight must be contiguous rank 2, got "
            f"shape={tuple(packed.shape)} stride={tuple(packed.stride())}"
        )
    if scale.ndim != 2 or not scale.is_contiguous():
        raise ValueError(
            f"{name}.weight_scale must be contiguous rank 2, got "
            f"shape={tuple(scale.shape)} stride={tuple(scale.stride())}"
        )
    rows, packed_k = (int(dim) for dim in packed.shape)
    if rows <= 0 or packed_k <= 0 or (packed_k * 4) % 3:
        raise ValueError(
            f"{name}.weight has invalid packed FP6 shape {tuple(packed.shape)}"
        )
    k = packed_k * 4 // 3
    if k % 32:
        raise ValueError(f"{name}.weight decodes to K={k}, not divisible by 32")
    expected_scale = (rows, k // 32)
    if tuple(scale.shape) != expected_scale:
        raise ValueError(
            f"{name}.weight_scale shape {tuple(scale.shape)} does not match "
            f"packed weight; expected {expected_scale}"
        )
    return rows, k


def _validate_unit_scalar(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not torch.is_floating_point(tensor) or tensor.numel() != 1:
        raise ValueError(
            f"{name} must be one floating-point value, got "
            f"dtype={tensor.dtype} shape={tuple(tensor.shape)}"
        )
    value = tensor.reshape(()).float()
    if not bool(torch.isfinite(value)) or not bool(
        torch.allclose(value, torch.ones_like(value), atol=_UNIT_TOL)
    ):
        raise NotImplementedError(
            f"{name} must be finite and unit for the pure-MX W6A6 contract"
        )
    return value


def load_fp6_moe_weights_from_safetensors(
    get_tensor: Callable[[str], torch.Tensor],
    prefix: str,
    num_experts: int,
    *,
    activation: str = "silu",
    source_format: str = "mxfp6_default",
    device: torch.device | str = "cuda",
    gate_name: str = "gate_proj",
    up_name: str = "up_proj",
    down_name: str = "down_proj",
) -> FP6MoEWeights:
    """Reconstruct :class:`FP6MoEWeights` for one MoE layer from per-expert keys.

    ``get_tensor(key)`` returns the on-disk tensor for a full key. ``prefix`` is
    the per-layer module path (e.g. ``model...layers.3.mlp``); expert keys are
    read as ``{prefix}.experts.{e}.{proj}.{weight|weight_scale|...}``.
    """
    is_gated = activation == "silu"
    weight_fmt = weight_format_for_source(source_format)

    if isinstance(num_experts, bool) or not isinstance(num_experts, int):
        raise TypeError("num_experts must be a positive integer")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    encoded: list[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = []
    expected_geometry: tuple[int, int] | None = None

    for e in range(num_experts):
        base = f"{prefix}.experts.{e}"
        gate_name_full = f"{base}.{gate_name}"
        down_name_full = f"{base}.{down_name}"
        gate_w = get_tensor(gate_name_full + WEIGHT_SUFFIX)
        gate_s = get_tensor(gate_name_full + WEIGHT_SCALE_SUFFIX)
        down_w = get_tensor(down_name_full + WEIGHT_SUFFIX)
        down_s = get_tensor(down_name_full + WEIGHT_SCALE_SUFFIX)
        n, k = _validate_fp6_weight_pair(gate_name_full, gate_w, gate_s)
        down_k, down_n = _validate_fp6_weight_pair(
            down_name_full, down_w, down_s
        )
        if (down_k, down_n) != (k, n):
            raise ValueError(
                f"{down_name_full} geometry {(down_k, down_n)} is not the "
                f"inverse of gate geometry {(n, k)}"
            )
        if expected_geometry is None:
            expected_geometry = (n, k)
        elif expected_geometry != (n, k):
            raise ValueError(
                f"expert {e} geometry {(n, k)} does not match "
                f"expert 0 geometry {expected_geometry}"
            )
        up_w: torch.Tensor | None = None
        up_s: torch.Tensor | None = None
        if is_gated:
            up_name_full = f"{base}.{up_name}"
            up_w = get_tensor(up_name_full + WEIGHT_SUFFIX)
            up_s = get_tensor(up_name_full + WEIGHT_SCALE_SUFFIX)
            if _validate_fp6_weight_pair(up_name_full, up_w, up_s) != (n, k):
                raise ValueError(
                    f"{up_name_full} geometry must match gate geometry {(n, k)}"
                )
            _validate_unit_scalar(
                up_name_full + WEIGHT_SCALE_2_SUFFIX,
                get_tensor(up_name_full + WEIGHT_SCALE_2_SUFFIX),
            )
            _validate_unit_scalar(
                up_name_full + INPUT_SCALE_SUFFIX,
                get_tensor(up_name_full + INPUT_SCALE_SUFFIX),
            )
        _validate_unit_scalar(
            gate_name_full + WEIGHT_SCALE_2_SUFFIX,
            get_tensor(gate_name_full + WEIGHT_SCALE_2_SUFFIX),
        )
        _validate_unit_scalar(
            down_name_full + WEIGHT_SCALE_2_SUFFIX,
            get_tensor(down_name_full + WEIGHT_SCALE_2_SUFFIX),
        )
        _validate_unit_scalar(
            gate_name_full + INPUT_SCALE_SUFFIX,
            get_tensor(gate_name_full + INPUT_SCALE_SUFFIX),
        )
        _validate_unit_scalar(
            down_name_full + INPUT_SCALE_SUFFIX,
            get_tensor(down_name_full + INPUT_SCALE_SUFFIX),
        )
        encoded.append((gate_w, gate_s, up_w, up_s, down_w, down_s))

    w1_codes: list[torch.Tensor] = []
    w2_codes: list[torch.Tensor] = []
    w1_scales: list[torch.Tensor] = []
    w2_scales: list[torch.Tensor] = []
    for gate_w, gate_s, up_w, up_s, down_w, down_s in encoded:
        if up_w is not None and up_s is not None:
            w1_codes.append(torch.cat([up_w, gate_w], dim=0))
            w1_scales.append(torch.cat([up_s, gate_s], dim=0))
        else:
            w1_codes.append(gate_w)
            w1_scales.append(gate_s)
        w2_codes.append(down_w)
        w2_scales.append(down_s)

    dev = torch.device(device)
    w1_fp6 = torch.stack(w1_codes, dim=0).to(dev).contiguous()
    w2_fp6 = torch.stack(w2_codes, dim=0).to(dev).contiguous()
    w1_blockscale = (
        _swizzle_stacked(torch.stack(w1_scales, dim=0)).to(dev).contiguous()
    )
    w2_blockscale = (
        _swizzle_stacked(torch.stack(w2_scales, dim=0)).to(dev).contiguous()
    )

    # Pure-MX contract: per-block UE8M0 carries the range, so dequant alphas and
    # activation global scales are all 1.0 (matches the validated FP6MoEWeights).
    experts = int(num_experts)
    n = int(w2_fp6.shape[2] * 4 // 3)
    k = int(w2_fp6.shape[1])
    ones_e = torch.ones(experts, dtype=torch.float32, device=dev)
    ones_1 = torch.ones(1, dtype=torch.float32, device=dev)
    return FP6MoEWeights(
        w1_fp6=w1_fp6,
        w1_blockscale=w1_blockscale,
        w1_alphas=ones_e.clone(),
        w2_fp6=w2_fp6,
        w2_blockscale=w2_blockscale,
        w2_alphas=ones_e.clone(),
        a1_gscale=ones_1.clone(),
        a2_gscale=ones_1.clone(),
        num_experts=experts,
        k=k,
        n=n,
        weight_fmt=weight_fmt,
        source_format=source_format,
        activation=activation,
    )


def load_fp6_moe_checkpoint(
    model_path: str,
    *,
    activation: str = "silu",
    device: torch.device | str = "cuda",
    limit_layers: Optional[int] = None,
) -> dict[int, FP6MoEWeights]:
    """Load every routed-MoE layer of an FP6 safetensors checkpoint.

    Returns ``{layer_index: FP6MoEWeights}``. ``source_format`` is recovered from
    ``config.json``'s ``quantization_config``; ``activation`` defaults to silu.
    """
    from .model_fp6 import SafetensorsModel, discover_moe_experts

    model = SafetensorsModel(model_path)
    quant_config = model.config.get("quantization_config")
    source_format = _source_format_from_config(quant_config)
    scheme = discover_moe_experts(model)
    if scheme is None:
        raise ValueError(f"no FP6 MoE experts discovered under {model_path}")

    layers = scheme.layers if limit_layers is None else scheme.layers[:limit_layers]
    out: dict[int, FP6MoEWeights] = {}
    for layer in layers:
        prefix = scheme.prefix_template.format(L=layer)
        out[layer] = load_fp6_moe_weights_from_safetensors(
            model.get_tensor,
            prefix,
            scheme.num_experts,
            activation=activation,
            source_format=source_format,
            device=device,
            gate_name=scheme.gate_name,
            up_name=scheme.up_name,
            down_name=scheme.down_name,
        )
    return out


def load_fp6_dense_weight_from_safetensors(
    get_tensor: Callable[[str], torch.Tensor],
    name: str,
    *,
    source_format: str = "mxfp6_default",
    device: torch.device | str = "cuda",
) -> FP6DenseWeight:
    """Reconstruct an :class:`FP6DenseWeight` for one linear from its FP6 keys.

    Reads ``{name}.weight`` / ``.weight_scale`` / ``.weight_scale_2``, swizzles the
    unswizzled on-disk UE8M0 scales into the cutlass layout
    :meth:`FP6DenseWeight.scale_view` expects, and carries the (unit) weight global
    scale. The result drops straight into :func:`dense_fp6_linear`, whose
    ``alpha = 1/(a_gscale * weight_global_scale)`` is correct for the unit scale.
    """
    packed = get_tensor(name + WEIGHT_SUFFIX)
    wscale = get_tensor(name + WEIGHT_SCALE_SUFFIX)
    out_f, in_f = _validate_fp6_weight_pair(name, packed, wscale)
    wgs = _validate_unit_scalar(
        name + WEIGHT_SCALE_2_SUFFIX,
        get_tensor(name + WEIGHT_SCALE_2_SUFFIX),
    ).reshape(1)
    _validate_unit_scalar(
        name + INPUT_SCALE_SUFFIX,
        get_tensor(name + INPUT_SCALE_SUFFIX),
    )
    fmt = weight_format_for_source(source_format)
    act_fmt = activation_format_for_source(source_format)
    scale_storage = (
        swizzle_block_scale(wscale.view(torch.float8_e8m0fnu)).reshape(-1).view(torch.uint8)
    )
    dev = torch.device(device)
    return FP6DenseWeight(
        packed=packed.to(dev).contiguous(),
        scale_storage=scale_storage.to(dev).contiguous(),
        global_scale=wgs.to(dev),
        out_features=out_f,
        in_features=in_f,
        fmt=fmt,
        act_fmt=act_fmt,
    )


def load_fp6_dense_checkpoint(
    model_path: str,
    *,
    device: torch.device | str = "cuda",
) -> dict[str, FP6DenseWeight]:
    """Load every FP6-quantized dense linear from a safetensors checkpoint.

    Keys are detected by the presence of a ``.weight_scale`` sibling next to a
    ``.weight`` tensor, so this is format-driven (works regardless of module
    naming). Returns ``{module_name: FP6DenseWeight}``.
    """
    from .model_fp6 import SafetensorsModel

    model = SafetensorsModel(model_path)
    quant_config = model.config.get("quantization_config")
    source_format = _source_format_from_config(quant_config)
    out: dict[str, FP6DenseWeight] = {}
    for key in model.keys():
        if not key.endswith(WEIGHT_SUFFIX):
            continue
        name = key[: -len(WEIGHT_SUFFIX)]
        if not model.has(name + WEIGHT_SCALE_SUFFIX):
            continue
        out[name] = load_fp6_dense_weight_from_safetensors(
            model.get_tensor, name, source_format=source_format, device=device
        )
    return out
