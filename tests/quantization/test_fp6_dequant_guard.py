"""Regression tests for FP6 dequantization geometry and working-set guards."""
from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch

import b12x._lib.fp6 as fp6_impl
import b12x.quantization.mxfp6.fp6_checkpoint as checkpoint
import b12x.quantization.mxfp6.fp6_safetensors_export as exporter
from b12x.quantization.mxfp6.model_fp6 import SafetensorsModel


def _artifact(rows: int = 2, k: int = 64):
    torch.manual_seed(0)
    weight = torch.randn(rows, k, dtype=torch.bfloat16) / 8
    return checkpoint.quantize_linear_to_fp6(
        weight,
        source_format="mxfp6_w6a8",
        use_gpu=False,
        block_scale_rule="mse",
    )


def _fail_if_expanded(*_args, **_kwargs):
    raise AssertionError("packed FP6 bytes expanded before validation")


def test_malformed_scale_rejected_before_expansion(monkeypatch) -> None:
    artifact = _artifact()
    malformed_scale = torch.zeros((2, 1), dtype=torch.uint8)
    monkeypatch.setattr(
        fp6_impl, "expand_mxfp6_packed_to_bytes", _fail_if_expanded
    )

    with pytest.raises(ValueError, match="block_scale shape"):
        checkpoint.dequantize_linear_from_fp6(
            artifact.weight,
            malformed_scale,
            fmt=artifact.fmt,
        )


def test_working_set_budget_rejected_before_expansion(monkeypatch) -> None:
    artifact = _artifact()
    monkeypatch.setattr(
        fp6_impl, "expand_mxfp6_packed_to_bytes", _fail_if_expanded
    )

    with pytest.raises(ValueError, match="exceeds the operator budget"):
        checkpoint.dequantize_linear_from_fp6(
            artifact.weight,
            artifact.weight_scale,
            fmt=artifact.fmt,
            max_working_bytes=1,
        )


def test_invalid_format_rejected_before_expansion(monkeypatch) -> None:
    artifact = _artifact()
    monkeypatch.setattr(
        fp6_impl, "expand_mxfp6_packed_to_bytes", _fail_if_expanded
    )

    with pytest.raises(ValueError, match="fmt must be"):
        checkpoint.dequantize_linear_from_fp6(
            artifact.weight,
            artifact.weight_scale,
            fmt="invalid",
        )


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_working_set_budget_rejected(budget) -> None:
    artifact = _artifact()
    expected = TypeError if isinstance(budget, (bool, float)) else ValueError

    with pytest.raises(expected, match="max_working_bytes"):
        checkpoint.dequantize_linear_from_fp6(
            artifact.weight,
            artifact.weight_scale,
            fmt=artifact.fmt,
            max_working_bytes=budget,
        )


def test_exact_working_set_budget_accepts_valid_producer_artifact() -> None:
    artifact = _artifact(rows=3, k=64)
    rows = int(artifact.weight.shape[0])
    k = int(artifact.weight.shape[1]) * 4 // 3
    budget = checkpoint._estimate_fp6_dequant_working_bytes(rows, k)

    decoded = checkpoint.dequantize_linear_from_fp6(
        artifact.weight,
        artifact.weight_scale,
        fmt=artifact.fmt,
        weight_scale_2=artifact.weight_scale_2,
        max_working_bytes=budget,
    )

    assert decoded.shape == (rows, k)
    assert bool(torch.isfinite(decoded).all())


def test_per_row_global_scale_remains_supported() -> None:
    artifact = _artifact(rows=2, k=64)
    unscaled = checkpoint.dequantize_linear_from_fp6(
        artifact.weight,
        artifact.weight_scale,
        fmt=artifact.fmt,
    )
    row_scales = torch.tensor([2.0, 4.0], dtype=torch.float32)

    scaled = checkpoint.dequantize_linear_from_fp6(
        artifact.weight,
        artifact.weight_scale,
        fmt=artifact.fmt,
        weight_scale_2=row_scales,
    )

    torch.testing.assert_close(scaled[0], unscaled[0] / 2)
    torch.testing.assert_close(scaled[1], unscaled[1] / 4)


def test_checkpoint_preflight_rejects_late_group_before_load_or_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_algo": checkpoint.QUANT_ALGO,
                    "weight_format": "e2m3",
                }
            }
        )
    )
    save_file(
        {
            "a.weight": torch.zeros((2, 48), dtype=torch.uint8),
            "a.weight_scale": torch.ones((2, 2), dtype=torch.uint8),
            "a.weight_scale_2": torch.ones(1),
            "a.input_scale": torch.ones(1),
            "b.weight": torch.zeros((2, 48), dtype=torch.uint8),
            "b.weight_scale": torch.ones((2, 1), dtype=torch.uint8),
            "b.weight_scale_2": torch.ones(1),
            "b.input_scale": torch.ones(1),
        },
        source / "model.safetensors",
    )

    def forbid_load(*_args, **_kwargs):
        raise AssertionError("preflight loaded tensor data")

    monkeypatch.setattr(SafetensorsModel, "get_tensor", forbid_load)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="b.weight_scale shape"):
        exporter.dequantize_fp6_checkpoint_to_bf16(
            source,
            output,
            device="cpu",
            verbose=False,
        )

    assert not output.exists()


def test_checkpoint_preflight_budget_rejects_before_load_or_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_algo": checkpoint.QUANT_ALGO,
                    "weight_format": "e2m3",
                }
            }
        )
    )
    save_file(
        {
            "a.weight": torch.zeros((2, 48), dtype=torch.uint8),
            "a.weight_scale": torch.ones((2, 2), dtype=torch.uint8),
            "a.weight_scale_2": torch.ones(1),
            "a.input_scale": torch.ones(1),
        },
        source / "model.safetensors",
    )

    def forbid_load(*_args, **_kwargs):
        raise AssertionError("preflight loaded tensor data")

    monkeypatch.setattr(SafetensorsModel, "get_tensor", forbid_load)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="max_dequant_working_bytes"):
        exporter.dequantize_fp6_checkpoint_to_bf16(
            source,
            output,
            device="cpu",
            max_dequant_working_bytes=1,
            verbose=False,
        )

    assert not output.exists()


def test_checkpoint_runtime_failure_removes_staged_shards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_algo": checkpoint.QUANT_ALGO,
                    "weight_format": "e2m3",
                }
            }
        )
    )
    save_file(
        {
            "a": torch.ones(4, dtype=torch.bfloat16),
            "b": torch.ones(4, dtype=torch.bfloat16),
            "c": torch.ones(4, dtype=torch.bfloat16),
        },
        source / "model.safetensors",
    )
    original_get_tensor = SafetensorsModel.get_tensor
    loads = 0

    def fail_third_load(self, key):
        nonlocal loads
        loads += 1
        if loads == 3:
            raise RuntimeError("injected read failure")
        return original_get_tensor(self, key)

    monkeypatch.setattr(SafetensorsModel, "get_tensor", fail_third_load)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="injected read failure"):
        exporter.dequantize_fp6_checkpoint_to_bf16(
            source,
            output,
            device="cpu",
            max_shard_bytes=12,
            verbose=False,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".output.staging-*")) == []


def test_checkpoint_refuses_existing_output_without_mutation(
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_algo": checkpoint.QUANT_ALGO,
                    "weight_format": "e2m3",
                }
            }
        )
    )
    save_file(
        {"a": torch.ones(1, dtype=torch.bfloat16)},
        source / "model.safetensors",
    )
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("keep")

    with pytest.raises(FileExistsError, match="refusing to replace existing"):
        exporter.dequantize_fp6_checkpoint_to_bf16(
            source,
            output,
            device="cpu",
            verbose=False,
        )

    assert sentinel.read_text() == "keep"


def test_orphan_fp6_sidecar_rejected_before_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_algo": checkpoint.QUANT_ALGO,
                    "weight_format": "e2m3",
                }
            }
        )
    )
    save_file(
        {
            "x.weight": torch.zeros((2, 48), dtype=torch.uint8),
            "x.weight_scale_2": torch.ones(1),
            "x.input_scale": torch.ones(1),
        },
        source / "model.safetensors",
    )
    monkeypatch.setattr(
        SafetensorsModel,
        "get_tensor",
        lambda *_args: pytest.fail("orphan group reached tensor load"),
    )
    with pytest.raises(ValueError, match="incomplete FP6 tensor group"):
        exporter.dequantize_fp6_checkpoint_to_bf16(
            source, tmp_path / "output", device="cpu", verbose=False
        )


def test_real_producer_artifact_dequantizes_successfully(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    artifact = _artifact(rows=2, k=64)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_algo": checkpoint.QUANT_ALGO,
                    "weight_format": artifact.fmt,
                }
            }
        )
    )
    save_file(
        {
            "linear.weight": artifact.weight,
            "linear.weight_scale": artifact.weight_scale,
            "linear.weight_scale_2": artifact.weight_scale_2,
            "linear.input_scale": artifact.input_scale,
        },
        source / "model.safetensors",
    )
    output = tmp_path / "output"
    exporter.dequantize_fp6_checkpoint_to_bf16(
        source, output, device="cpu", verbose=False
    )
    index = json.loads((output / "model.safetensors.index.json").read_text())
    shard = index["weight_map"]["linear.weight"]
    decoded = load_file(output / shard)["linear.weight"]
    assert decoded.shape == (2, 64)
    assert decoded.dtype == torch.bfloat16
    assert bool(torch.isfinite(decoded).all())


def test_writer_rejects_single_tensor_above_payload_before_copy(
    tmp_path: Path,
) -> None:
    writer = exporter._ShardWriter(
        tmp_path,
        max_shard_bytes=4,
        max_buffer_bytes=32,
    )
    tensor = torch.zeros(5, dtype=torch.uint8)
    with pytest.raises(ValueError, match="writer payload limit 4"):
        writer.add("oversized", tensor)


def test_zero_byte_tensor_metadata_is_charged_to_budget() -> None:
    keys = [f"copy.{index:05d}" for index in range(1024)]

    class MetadataOnlyModel:
        def keys(self):
            return keys

        def shape_of(self, _key):
            return (0,)

        def dtype_of(self, _key):
            return "U8"

    with pytest.raises(ValueError, match="metadata reserve"):
        exporter._preflight_fp6_dequant_checkpoint(
            MetadataOnlyModel(),
            set(),
            max_working_bytes=exporter._SERIALIZER_OVERHEAD_BYTES + 1024,
            max_shard_bytes=1024,
        )


def test_cli_forwards_operator_working_set_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import b12x.quantization.mxfp6 as package

    calls = []

    def fake_dequant(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            quantized_tensors=0,
            copied_tensors=0,
            shards=0,
            total_bytes=0,
            out_dir="out",
        )

    monkeypatch.setattr(package, "dequantize_fp6_checkpoint_to_bf16", fake_dequant)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dequantize_fp6_to_bf16.py",
            "--model",
            "model",
            "--out",
            "out",
            "--no-gpu",
            "--max-working-bytes",
            "1234567",
        ],
    )
    script = Path(__file__).parents[2] / "scripts" / "dequantize_fp6_to_bf16.py"
    runpy.run_path(str(script), run_name="__main__")
    assert calls[0][1]["max_dequant_working_bytes"] == 1234567
