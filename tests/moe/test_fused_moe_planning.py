from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from b12x.moe import fused_moe
import b12x.moe.fused_moe._impl as fused_moe_impl


def _weight_plan() -> fused_moe.WeightsPlan:
    return fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=160,
        hidden_size=6144,
        intermediate_size=512,
        w13_layout="w13",
    )


def _caps(*, block_size_m: int | None) -> fused_moe.Caps:
    return fused_moe.Caps(
        max_tokens=64,
        num_topk=8,
        route_num_experts=160,
        device="cpu",
        weight_plan=_weight_plan(),
        quant_mode="w4a16",
        w4a16_block_size_m=block_size_m,
    )


def _trellis_caps() -> fused_moe.Caps:
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="exl3_trellis_mcg",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=160,
        hidden_size=6144,
        intermediate_size=512,
        w13_layout="w13",
        trellis_bits=3,
        trellis_tile_config=(64, 256, 64, 256),
    )
    return fused_moe.Caps(
        max_tokens=3072,
        num_topk=8,
        route_num_experts=160,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
        w4a16_block_size_m=64,
    )


def _small_packed_caps() -> fused_moe.Caps:
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="compressed_tensors",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=16,
        hidden_size=128,
        intermediate_size=128,
        w13_layout="w13",
    )
    return fused_moe.Caps(
        max_tokens=4,
        num_topk=8,
        route_num_experts=16,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
    )


def _subset_router_caps() -> fused_moe.Caps:
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="compressed_tensors",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=160,
        hidden_size=128,
        intermediate_size=128,
        w13_layout="w13",
    )
    return fused_moe.Caps(
        max_tokens=8,
        num_topk=8,
        route_num_experts=16,
        device="cpu",
        weight_plan=weight_plan,
        quant_mode="w4a16",
    )


def test_required_nbytes_avoids_launch_prewarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    def fail_launch_prewarm(**_kwargs) -> None:
        raise AssertionError("launch prewarm called")

    monkeypatch.setattr(
        fused_moe_impl,
        "_plan_full_rotation_w4a16_launches",
        fail_launch_prewarm,
    )
    caps = _trellis_caps()

    required = fused_moe.required_nbytes(caps)

    assert 900 * (1 << 20) < required < 920 * (1 << 20)
    assert "required_nbytes" in fused_moe.META.entry_points
    with pytest.raises(TypeError, match="TPMoEScratchCaps"):
        fused_moe.required_nbytes(object())


def test_required_nbytes_matches_scratch_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)
    caps = _caps(block_size_m=8)

    plan = fused_moe.plan(caps)

    assert fused_moe.required_nbytes(caps) == plan.scratch_specs()[0].shape[0]


def test_small_packed_plan_covers_direct_topk_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    plan = fused_moe.plan(_small_packed_caps())
    specs = {spec.name: spec for spec in plan._core_workspace_plan.tensor_specs}

    assert specs["fc1_c_tmp"].shape == (131072,)
    assert specs["fc2_c_tmp"].shape == (65536,)


def test_non_trellis_core_sizes_routes_for_weight_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    plan = fused_moe.plan(_subset_router_caps())
    specs = {spec.name: spec for spec in plan._core_workspace_plan.tensor_specs}

    assert plan._core_workspace_plan.route_E == 160
    assert specs["packed_route_indices"].shape == (512,)
    assert specs["block_expert_ids"].shape == (64,)
    assert specs["expert_offsets"].shape == (161,)


def test_unpinned_small_capacity_matches_reachable_block_8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)

    automatic = fused_moe.required_nbytes(_caps(block_size_m=None))
    exact = fused_moe.required_nbytes(_caps(block_size_m=8))
    oversized = fused_moe.required_nbytes(_caps(block_size_m=64))

    assert automatic == exact
    # Reusing the disjoint FC1/FC2 split-K slot removes the smaller plane from
    # both plans; the oversized block choice must still waste material capacity.
    assert oversized - automatic > 32 * 1024 * 1024


def _fruit_qsrt_core_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> fused_moe_impl._TPCoreWorkspacePlan:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 170)
    return fused_moe_impl._plan_core_workspace(
        "w4a16",
        "w4a16",
        256,
        256,
        1024,
        256,
        8,
        torch.device("cpu"),
        torch.bfloat16,
        routed_rows=4096 * 8,
        max_rows=4096,
        activation="silu",
        source_format="qsrt_sqg_e4m3",
        w13_layout="w13",
        w4a16_weight_layout="trellis3_t256",
        w4a16_scale_format="e4m3_k32",
        route_num_experts=256,
        w4a16_block_size_m=16,
        trellis_bits=3,
        trellis_tile_config=(64, 256, 64, 256),
        qsrt_storage_format="qsrt_atoms_v1",
    )


def _linear_core_workspace_nbytes(
    plan: fused_moe_impl._TPCoreWorkspacePlan,
) -> int:
    nbytes = 0
    for spec in plan.tensor_specs:
        nbytes = fused_moe_impl.align_up(
            nbytes,
            max(16, fused_moe_impl._dtype_nbytes(spec.dtype)),
        )
        nbytes += fused_moe_impl._tensor_numel(
            spec.shape
        ) * fused_moe_impl._dtype_nbytes(spec.dtype)
    return nbytes


def _view_byte_range(tensor: torch.Tensor) -> tuple[int, int]:
    begin = tensor.data_ptr()
    return begin, begin + tensor.numel() * tensor.element_size()


def test_fruit_qsrt_reuse_groups_reduce_exact_required_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _fruit_qsrt_core_plan(monkeypatch)
    specs = {spec.name: spec for spec in plan.tensor_specs}
    required = fused_moe_impl._core_workspace_nbytes(plan)
    baseline = _linear_core_workspace_nbytes(plan)

    assert specs["fc1_c_tmp"].reuse_group == "w4a16_c_tmp"
    assert specs["fc2_c_tmp"].reuse_group == "w4a16_c_tmp"
    assert specs["fc1_c_tmp"].shape == (2_785_280,)
    assert specs["fc2_c_tmp"].shape == (2_785_280,)
    assert specs["fc1_c_tmp"].dtype == torch.float32
    assert specs["fc2_c_tmp"].dtype == torch.float32
    assert (
        fused_moe_impl._tensor_numel(specs["fc1_c_tmp"].shape)
        * fused_moe_impl._dtype_nbytes(specs["fc1_c_tmp"].dtype)
        == 11_141_120
    )

    post_fc1_group = "w4a16_full_rotation_post_fc1"
    assert specs["rotation_a_gate"].reuse_group == post_fc1_group
    assert specs["intermediate_cache2"].reuse_group == post_fc1_group
    assert specs["full_rotation_output"].reuse_group == post_fc1_group
    assert baseline - required == 44_695_552


def test_fruit_qsrt_reuse_views_alias_without_overlapping_other_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _fruit_qsrt_core_plan(monkeypatch)
    required = fused_moe_impl._core_workspace_nbytes(plan)
    storage = torch.empty((required,), dtype=torch.uint8)

    views = fused_moe_impl._map_core_workspace_views(
        plan,
        storage,
        do_init=False,
    )
    specs = {spec.name: spec for spec in plan.tensor_specs}
    fc1 = views["fc1_c_tmp"]
    fc2 = views["fc2_c_tmp"]

    assert fc1.data_ptr() == fc2.data_ptr()
    assert tuple(fc1.shape) == specs["fc1_c_tmp"].shape
    assert tuple(fc2.shape) == specs["fc2_c_tmp"].shape
    assert fc1.dtype == specs["fc1_c_tmp"].dtype
    assert fc2.dtype == specs["fc2_c_tmp"].dtype

    slot_begin = fc1.data_ptr()
    slot_nbytes = max(fc1.numel(), fc2.numel()) * fc1.element_size()
    slot_end = slot_begin + slot_nbytes
    for name, tensor in views.items():
        if name in {"fc1_c_tmp", "fc2_c_tmp"}:
            continue
        begin, end = _view_byte_range(tensor)
        assert end <= slot_begin or begin >= slot_end

    post_fc1_names = {
        "rotation_a_gate",
        "intermediate_cache2",
        "full_rotation_output",
    }
    post_fc1_views = [views[name] for name in post_fc1_names]
    assert len({tensor.data_ptr() for tensor in post_fc1_views}) == 1
    post_fc1_begin = post_fc1_views[0].data_ptr()
    post_fc1_nbytes = max(
        tensor.numel() * tensor.element_size() for tensor in post_fc1_views
    )
    post_fc1_end = post_fc1_begin + post_fc1_nbytes
    for name, tensor in views.items():
        if name in post_fc1_names:
            continue
        begin, end = _view_byte_range(tensor)
        assert end <= post_fc1_begin or begin >= post_fc1_end

    with pytest.raises(ValueError, match="requires .* but only .* available"):
        fused_moe_impl._map_core_workspace_views(
            plan,
            storage.narrow(0, 0, required - 1),
            do_init=False,
        )


def test_w4a16_reuse_slot_keeps_unequal_member_extents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fused_moe_impl, "get_num_sm", lambda _device: 188)
    core = fused_moe.plan(_small_packed_caps())._core_workspace_plan
    specs = {spec.name: spec for spec in core.tensor_specs}
    required = fused_moe_impl._core_workspace_nbytes(core)
    baseline = _linear_core_workspace_nbytes(core)
    storage = torch.empty((required,), dtype=torch.uint8)

    views = fused_moe_impl._map_core_workspace_views(
        core,
        storage,
        do_init=False,
    )

    assert specs["fc1_c_tmp"].shape == (131072,)
    assert specs["fc2_c_tmp"].shape == (65536,)
    assert tuple(views["fc1_c_tmp"].shape) == specs["fc1_c_tmp"].shape
    assert tuple(views["fc2_c_tmp"].shape) == specs["fc2_c_tmp"].shape
    assert views["fc1_c_tmp"].dtype == torch.float32
    assert views["fc2_c_tmp"].dtype == torch.float32
    assert views["fc1_c_tmp"].data_ptr() == views["fc2_c_tmp"].data_ptr()
    assert baseline - required == 65536 * fused_moe_impl._dtype_nbytes(
        torch.float32
    )


def test_non_w4a16_workspace_specs_do_not_reuse_arena_slots() -> None:
    plan = fused_moe_impl._plan_core_workspace(
        "micro",
        "nvfp4",
        8,
        8,
        128,
        64,
        2,
        torch.device("cpu"),
        torch.bfloat16,
        routed_rows=8,
        max_rows=8,
    )
    assert all(spec.reuse_group is None for spec in plan.tensor_specs)

    required = fused_moe_impl._core_workspace_nbytes(plan)
    views = fused_moe_impl._map_core_workspace_views(
        plan,
        torch.empty((required,), dtype=torch.uint8),
        do_init=False,
    )
    ranges = sorted(_view_byte_range(tensor) for tensor in views.values())

    assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))


def test_core_workspace_maps_mixed_dtype_empty_reuse_group() -> None:
    specs = (
        fused_moe_impl._TensorAllocSpec(
            "fp16", (4,), torch.float16, reuse_group="mixed"
        ),
        fused_moe_impl._TensorAllocSpec(
            "fp32", (4,), torch.float32, reuse_group="mixed"
        ),
    )
    plan = SimpleNamespace(tensor_specs=specs)
    required = fused_moe_impl._core_workspace_nbytes(plan)
    views = fused_moe_impl._map_core_workspace_views(
        plan,
        torch.empty((required,), dtype=torch.uint8),
        do_init=False,
    )

    assert required == 16
    assert views["fp16"].data_ptr() == views["fp32"].data_ptr()
    assert views["fp16"].dtype == torch.float16
    assert views["fp32"].dtype == torch.float32


@pytest.mark.parametrize(
    ("specs", "message"),
    (
        (
            (
                fused_moe_impl._TensorAllocSpec("dup", (1,), torch.float32),
                fused_moe_impl._TensorAllocSpec("dup", (2,), torch.float32),
            ),
            "duplicate tensor allocation name",
        ),
        (
            (
                fused_moe_impl._TensorAllocSpec(
                    "a", (1,), torch.float32, init="arange", reuse_group="slot"
                ),
                fused_moe_impl._TensorAllocSpec(
                    "b", (2,), torch.float16, init="arange", reuse_group="slot"
                ),
            ),
            "incompatible dtypes",
        ),
        (
            (
                fused_moe_impl._TensorAllocSpec(
                    "a", (1,), torch.float32, init="zeros", reuse_group="slot"
                ),
                fused_moe_impl._TensorAllocSpec(
                    "b", (2,), torch.float32, init="empty", reuse_group="slot"
                ),
            ),
            "incompatible init modes",
        ),
        (
            (
                fused_moe_impl._TensorAllocSpec(
                    "a", (1,), torch.float32, alignment=16, reuse_group="slot"
                ),
                fused_moe_impl._TensorAllocSpec(
                    "b", (2,), torch.float32, alignment=32, reuse_group="slot"
                ),
            ),
            "incompatible alignments",
        ),
    ),
)
def test_core_workspace_rejects_malformed_reuse_groups(
    specs: tuple[fused_moe_impl._TensorAllocSpec, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fused_moe_impl._core_workspace_nbytes(
            SimpleNamespace(tensor_specs=specs)
        )
