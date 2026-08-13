"""Focused regression tests for expert-ID range validation (issue #154).

Every public path that feeds ``topk_ids`` to a dynamic/micro MoE kernel
must reject IDs outside ``[0, num_experts)`` *before* any raw GPU write.
These tests cover:

* bind-time rejection (``fused_moe.bind`` / ``build_tp_moe_fp4_binding``)
* run-time rejection (``fused_moe.run`` after in-place ID mutation)
* boundary acceptance (IDs ``0`` and ``E-1`` remain valid)
* launch-sentinel confirmation that the kernel is never called for bad IDs
* CUDA-graph replay safety when the bound ``topk_ids`` buffer is mutated
  to invalid values between replays
* int32 and int64 ID dtypes, including int64 values that overflow int32
* caller-provided ``Routing`` in the sparse-MoE path
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from .._reference.helpers import make_tp_moe_fp4_binding, prepare_tp_moe_fp4_experts
from ..conftest import require_b12x


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_experts(device: torch.device, E: int = 4, K: int = 128, n: int = 64):
    """Create minimal NVFP4 experts for testing."""
    torch.manual_seed(999)
    a1_gscale = torch.ones(E, device=device)
    a2_gscale = torch.ones(E, device=device)
    w1_fp4 = torch.randint(0, 256, (E, 2 * n, K // 2), dtype=torch.uint8, device=device)
    w1_blockscale = (
        torch.rand(E, 2 * n, K // 16, device=device) * 0.25 + 0.03125
    ).to(torch.float8_e4m3fn)
    w1_alphas = (torch.rand(E, device=device) * 0.1 + 0.05).float()
    w2_fp4 = torch.randint(0, 256, (E, K, n // 2), dtype=torch.uint8, device=device)
    w2_blockscale = (
        torch.rand(E, K, n // 16, device=device) * 0.25 + 0.03125
    ).to(torch.float8_e4m3fn)
    w2_alphas = (torch.rand(E, device=device) * 0.1 + 0.05).float()
    a = torch.zeros(1, K, dtype=torch.bfloat16, device=device)
    return prepare_tp_moe_fp4_experts(
        a=a,
        a1_gscale=a1_gscale,
        w1_fp4=w1_fp4,
        w1_blockscale=w1_blockscale,
        w1_alphas=w1_alphas,
        a2_gscale=a2_gscale,
        w2_fp4=w2_fp4,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_alphas,
        activation="silu",
        quant_mode="nvfp4",
    )


def _make_valid_binding(device, E, m, topk, experts=None, ids_dtype=torch.int32):
    if experts is None:
        experts = _make_experts(device, E=E)
    torch.manual_seed(42)
    topk_ids = torch.randint(0, E, (m, topk), dtype=ids_dtype, device=device)
    topk_weights = torch.softmax(
        torch.randn(m, topk, device=device), dim=-1
    ).float()
    return make_tp_moe_fp4_binding(
        a=torch.randn(m, 128, device=device).to(torch.bfloat16),
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=torch.empty(m, 128, dtype=torch.bfloat16, device=device),
        quant_mode="nvfp4",
    )


# ---------------------------------------------------------------------------
# Bind-time rejection
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bad_id", [-1, 4, 1000000])
def test_bind_rejects_out_of_range_expert_ids(bad_id: int) -> None:
    """fused_moe.bind must raise ValueError for IDs outside [0, E)."""
    require_b12x()
    device = torch.device("cuda")
    E, m, topk = 4, 8, 2
    experts = _make_experts(device, E=E)
    topk_ids = torch.zeros(m, topk, dtype=torch.int32, device=device)
    topk_ids[0, 0] = bad_id
    topk_weights = torch.softmax(
        torch.randn(m, topk, device=device), dim=-1
    ).float()
    with pytest.raises(ValueError, match=r"outside \[0,"):
        make_tp_moe_fp4_binding(
            a=torch.randn(m, 128, device=device).to(torch.bfloat16),
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=torch.empty(m, 128, dtype=torch.bfloat16, device=device),
            quant_mode="nvfp4",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bind_rejects_int64_overflow_id() -> None:
    """An int64 ID that overflows int32 must be rejected."""
    require_b12x()
    device = torch.device("cuda")
    E, m, topk = 4, 8, 2
    experts = _make_experts(device, E=E)
    topk_ids = torch.zeros(m, topk, dtype=torch.int64, device=device)
    # 2**31 overflows int32 to a negative value; either way it is out of range.
    topk_ids[0, 0] = 2**31
    topk_weights = torch.softmax(
        torch.randn(m, topk, device=device), dim=-1
    ).float()
    with pytest.raises(ValueError, match=r"outside \[0,"):
        make_tp_moe_fp4_binding(
            a=torch.randn(m, 128, device=device).to(torch.bfloat16),
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=torch.empty(m, 128, dtype=torch.bfloat16, device=device),
            quant_mode="nvfp4",
        )


# ---------------------------------------------------------------------------
# Run-time rejection (mutate IDs after bind)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bad_id", [-1, 4, 1000000])
def test_run_rejects_mutated_out_of_range_ids(bad_id: int) -> None:
    """b12x_moe_fp4 must raise ValueError when IDs are mutated after bind."""
    require_b12x()
    from b12x.moe import fused_moe

    device = torch.device("cuda")
    E, m, topk = 4, 8, 2
    binding = _make_valid_binding(device, E, m, topk)
    # Mutate the bound topk_ids tensor in-place to an invalid value.
    binding.topk_ids[0, 0] = bad_id
    with pytest.raises(ValueError, match=r"outside \[0,"):
        fused_moe.run(binding=binding)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_run_rejects_int64_ids_after_bind() -> None:
    """Run-time validation must catch invalid int64 IDs."""
    require_b12x()
    from b12x.moe import fused_moe

    device = torch.device("cuda")
    E, m, topk = 4, 8, 2
    binding = _make_valid_binding(
        device, E, m, topk, ids_dtype=torch.int64
    )
    binding.topk_ids[0, 0] = -1
    with pytest.raises(ValueError, match=r"outside \[0,"):
        fused_moe.run(binding=binding)


# ---------------------------------------------------------------------------
# Launch sentinel: kernel must NOT be called for invalid IDs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bad_id", [-1, 4])
def test_kernel_not_launched_for_invalid_ids(bad_id: int) -> None:
    """The dynamic/micro kernel launch must never execute for bad IDs."""
    require_b12x()
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe import _impl

    device = torch.device("cuda")
    E, m, topk = 4, 8, 2
    binding = _make_valid_binding(device, E, m, topk)
    binding.topk_ids[0, 0] = bad_id

    launch_called = {"dynamic": False, "micro": False}

    original_dynamic = _impl._launch_dynamic
    original_micro = _impl._launch_micro

    def _sentinel_dynamic(*args, **kwargs):
        launch_called["dynamic"] = True
        return original_dynamic(*args, **kwargs)

    def _sentinel_micro(*args, **kwargs):
        launch_called["micro"] = True
        return original_micro(*args, **kwargs)

    with patch.object(_impl, "_launch_dynamic", _sentinel_dynamic), \
         patch.object(_impl, "_launch_micro", _sentinel_micro):
        with pytest.raises(ValueError):
            fused_moe.run(binding=binding)

    assert not launch_called["dynamic"], \
        "dynamic kernel was launched despite invalid expert IDs"
    assert not launch_called["micro"], \
        "micro kernel was launched despite invalid expert IDs"


# ---------------------------------------------------------------------------
# Boundary acceptance: IDs 0 and E-1 must remain valid
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_boundary_ids_zero_and_E_minus_one_accepted() -> None:
    """IDs 0 and E-1 are the valid boundaries and must not raise."""
    require_b12x()
    from b12x.moe import fused_moe

    device = torch.device("cuda")
    E, m, topk = 4, 8, 2
    experts = _make_experts(device, E=E)
    topk_ids = torch.zeros(m, topk, dtype=torch.int32, device=device)
    # Alternate between boundary IDs 0 and E-1.
    topk_ids[:, 0] = 0
    topk_ids[:, 1] = E - 1
    topk_weights = torch.softmax(
        torch.randn(m, topk, device=device), dim=-1
    ).float()
    binding = make_tp_moe_fp4_binding(
        a=torch.randn(m, 128, device=device).to(torch.bfloat16),
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=torch.empty(m, 128, dtype=torch.bfloat16, device=device),
        quant_mode="nvfp4",
    )
    # Must not raise.
    out = fused_moe.run(binding=binding)
    assert out is not None


# ---------------------------------------------------------------------------
# CUDA-graph replay safety
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_graph_replay_safe_with_mutated_invalid_ids() -> None:
    """Graph replay with mutated invalid IDs must not cause OOB access.

    The device-side sanitization in _flatten_and_validate_routing runs
    inside the captured graph, so replay with a mutated topk_ids buffer
    clamps invalid IDs to 0 and zeros their weights — memory-safe even
    though the host-side eager check cannot run during replay.
    """
    require_b12x()
    from b12x.moe import fused_moe

    device = torch.device("cuda")
    E, m, topk = 4, 8, 2
    binding = _make_valid_binding(device, E, m, topk)

    # Warm up to resolve kernels before capture.
    fused_moe.run(binding=binding)
    torch.cuda.synchronize()

    # Capture the run in a CUDA graph.
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
        fused_moe.run(binding=binding)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    # Mutate the bound topk_ids to contain invalid IDs (-1 and E).
    original_ids = binding.topk_ids.clone()
    binding.topk_ids[0, 0] = -1
    binding.topk_ids[0, 1] = E

    # Replay must not crash (no OOB access).
    graph.replay()
    torch.cuda.synchronize()

    # Restore valid IDs and verify the graph still produces correct output.
    binding.topk_ids.copy_(original_ids)
    graph.replay()
    torch.cuda.synchronize()

    # The output should be finite and non-trivial after restoring valid IDs.
    out = binding.output
    assert out.isfinite().all(), "output contains NaN/Inf after graph replay"
    assert out.abs().sum().item() > 0, "output is all zeros after valid replay"


# ---------------------------------------------------------------------------
# Sparse-MoE path: caller-provided Routing with invalid IDs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bad_id", [-1, 4])
def test_sparse_moe_rejects_invalid_routing(bad_id: int) -> None:
    """b12x_sparse_moe_fp4 must reject caller-provided Routing with bad IDs."""
    require_b12x()
    from b12x.moe.fused_moe._impl import (
        B12XTopKRouting,
        TPMoEWorkspacePool,
        b12x_sparse_moe_fp4,
        build_tp_moe_sparse_fp4_binding,
    )

    device = torch.device("cuda")
    E, m, topk = 4, 8, 2
    K = 128
    experts = _make_experts(device, E=E, K=K)
    a = torch.randn(m, K, device=device).to(torch.bfloat16)

    # Build a sparse binding with caller-provided invalid routing.
    topk_ids = torch.zeros(m, topk, dtype=torch.int32, device=device)
    topk_ids[0, 0] = bad_id
    topk_weights = torch.softmax(
        torch.randn(m, topk, device=device), dim=-1
    ).float()
    routing = B12XTopKRouting(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    sparse_binding = build_tp_moe_sparse_fp4_binding(
        scratch=TPMoEWorkspacePool(),
        hidden_states=a,
        experts=experts,
        routing=routing,
        output=torch.empty(m, K, dtype=torch.bfloat16, device=device),
        quant_mode="nvfp4",
    )
    with pytest.raises(ValueError, match=r"outside \[0,"):
        b12x_sparse_moe_fp4(binding=sparse_binding)
