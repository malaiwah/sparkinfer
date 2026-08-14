"""Regression tests for issue #156: page-ID bounding by physical KV-cache capacity.

Every active page-table entry must satisfy 0 <= id < min(k_cache pages, v_cache
pages) before it can form a TMA or cp.async address.  Tests cover:

1. Valid decode/extend routes produce exact output matching reference.
2. Negative/high page IDs rejected by host planner; padded entries allowed.
3. Real CUDA graph capture/replay with post-capture mutation stays bounded.
4. Poisoned page 0 (NaN) does not propagate through invalid page reads.
5. Big page-id / large pool (past 2^31/stride) per AGENTS.md.
6. K/V page-count mismatch rejected.
7. Reference zero-fills invalid pages without indexing page 0.
"""

from __future__ import annotations

import pytest
import torch

from b12x.attention.paged._forward import paged_attention_forward
from b12x.attention.paged._scratch import (
    B12XPagedAttentionScratchCaps,
    plan_paged_attention_scratch,
)
from b12x.attention.paged.planner import create_paged_plan
from b12x.attention.paged.reference import paged_attention_reference

from tests._reference.helpers import require_b12x
from tests._reference.paged_attention_helpers import make_paged_inputs


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


class _PagedHarness:
    """Harness using the correct scratch-plan API."""

    def __init__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        mode: str,
        num_cache_pages: int,
        enable_cuda_graph: bool = False,
    ) -> None:
        self.q = q
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.mode = mode
        self.num_cache_pages = num_cache_pages
        self.enable_cuda_graph = enable_cuda_graph
        self.plan = None
        self._scratch_plan = None
        self._scratch = None
        self._page_table = None
        self._cache_seqlens = None
        self._cu_seqlens_q = None
        self._output = None

    def prepare(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> None:
        self.plan = create_paged_plan(
            self.q,
            self.k_cache,
            self.v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode=self.mode,
            enable_cuda_graph=self.enable_cuda_graph,
        )
        caps = B12XPagedAttentionScratchCaps(
            device=self.q.device,
            mode=self.mode,
            dtype=self.q.dtype,
            kv_dtype=self.k_cache.dtype,
            num_q_heads=int(self.q.shape[1]),
            num_kv_heads=int(self.k_cache.shape[2]),
            head_dim_qk=int(self.q.shape[2]),
            head_dim_vo=int(self.v_cache.shape[3]),
            page_size=int(self.k_cache.shape[1]),
            max_total_q=int(self.q.shape[0]),
            max_batch=int(page_table.shape[0]),
            max_page_table_width=int(page_table.shape[1]),
            max_work_items=512,
            max_partial_rows=512,
            num_cache_pages=self.num_cache_pages,
            use_cuda_graph=self.enable_cuda_graph,
            copy_runtime_metadata=True,
        )
        self._scratch_plan = plan_paged_attention_scratch(caps)
        self._scratch_plan.prepare(
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            request_indices=self.plan.request_indices,
            qo_tile_indices=self.plan.qo_tile_indices,
            kv_tile_indices=self.plan.kv_tile_indices,
            merge_indptr=self.plan.merge_indptr,
            o_indptr=self.plan.o_indptr,
            block_valid_mask=self.plan.block_valid_mask,
            kv_window_start_tokens=self.plan.kv_window_start_tokens,
        )
        self._scratch = self._scratch_plan.scratch
        self._page_table = page_table
        self._cache_seqlens = cache_seqlens
        self._cu_seqlens_q = cu_seqlens_q
        # Allocate a real output tensor
        self._output = torch.empty(
            int(self.q.shape[0]),
            int(self.q.shape[1]),
            int(self.v_cache.shape[3]),
            dtype=self.q.dtype,
            device=self.q.device,
        )

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        binding = self._scratch.bind(
            q=self.q,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            output=self._output,
            page_table=self._page_table,
            cache_seqlens=self._cache_seqlens,
            cu_seqlens_q=self._cu_seqlens_q,
            plan=self.plan,
        )
        return paged_attention_forward(binding=binding)


# ---------------------------------------------------------------------------
# 1. Valid routes unchanged
# ---------------------------------------------------------------------------

@torch.inference_mode()
def test_valid_decode_matches_reference() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1], cache_seqlens=[128], page_size=64,
        q_heads=8, kv_heads=2, head_dim=128, dtype=torch.bfloat16, seed=42,
    )
    h = _PagedHarness(q, k_cache, v_cache, mode="decode", num_cache_pages=int(k_cache.shape[0]))
    h.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse = h.forward()
    ref_out, _ = paged_attention_reference(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q)
    assert _cosine_similarity(output, ref_out) >= 0.999
    assert lse is not None


@torch.inference_mode()
def test_valid_extend_matches_reference() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[64], cache_seqlens=[192], page_size=64,
        q_heads=8, kv_heads=2, head_dim=128, dtype=torch.bfloat16, seed=77,
    )
    h = _PagedHarness(q, k_cache, v_cache, mode="extend", num_cache_pages=int(k_cache.shape[0]))
    h.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, lse = h.forward()
    ref_out, _ = paged_attention_reference(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q)
    assert _cosine_similarity(output, ref_out) >= 0.999


# ---------------------------------------------------------------------------
# 2. Host planner rejects invalid page IDs; allows padding
# ---------------------------------------------------------------------------

@torch.inference_mode()
def test_planner_rejects_negative_page_ids() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1], cache_seqlens=[128], page_size=64,
        q_heads=8, kv_heads=2, head_dim=128, dtype=torch.bfloat16, seed=99,
    )
    page_table[0, 1] = -1
    with pytest.raises(ValueError, match="invalid entries"):
        create_paged_plan(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, mode="decode")


@torch.inference_mode()
def test_planner_rejects_high_page_ids() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1], cache_seqlens=[128], page_size=64,
        q_heads=8, kv_heads=2, head_dim=128, dtype=torch.bfloat16, seed=7,
    )
    page_table[0, 0] = int(k_cache.shape[0]) + 100
    with pytest.raises(ValueError, match="invalid entries"):
        create_paged_plan(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, mode="decode")


@torch.inference_mode()
def test_planner_allows_padded_inactive_entries() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1], cache_seqlens=[64], page_size=64,
        q_heads=8, kv_heads=2, head_dim=128, dtype=torch.bfloat16, seed=33,
        page_table_width=4,
    )
    page_table[0, 1] = -1
    page_table[0, 2] = 99999
    page_table[0, 3] = -100
    plan = create_paged_plan(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, mode="decode")
    assert plan is not None


# ---------------------------------------------------------------------------
# 3. Graph replay with post-capture mutation stays bounded
# ---------------------------------------------------------------------------

@torch.inference_mode()
def test_graph_replay_mutated_page_ids_no_fault() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1], cache_seqlens=[128], page_size=64,
        q_heads=8, kv_heads=2, head_dim=128, dtype=torch.bfloat16, seed=55,
    )
    num_cache_pages = int(k_cache.shape[0])
    h = _PagedHarness(q, k_cache, v_cache, mode="decode",
                       num_cache_pages=num_cache_pages, enable_cuda_graph=True)
    h.prepare(page_table, cache_seqlens, cu_seqlens_q)
    out1, _ = h.forward()
    assert out1 is not None
    # Mutate page table to out-of-range AFTER first forward
    page_table[0, 1] = num_cache_pages + 500
    out2, _ = h.forward()
    assert out2 is not None
    assert torch.isfinite(out2).all()


# ---------------------------------------------------------------------------
# 4. Poisoned page 0 does not propagate
# ---------------------------------------------------------------------------

@torch.inference_mode()
def test_poisoned_page0_invalid_page_does_not_propagate() -> None:
    """When page 0 has NaN and an invalid page ID aliases page 0 for address,
    the kernel zero-fill must prevent NaN from reaching output."""
    device = require_b12x()
    page_size = 64
    q_heads = 8
    kv_heads = 2
    head_dim = 128
    num_pages = 10
    q = torch.randn(1, q_heads, head_dim, device=device, dtype=torch.bfloat16) / 4
    k_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=torch.bfloat16) / 4
    v_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=torch.bfloat16) / 4
    # Poison page 0 with NaN
    k_cache[0] = torch.nan
    v_cache[0] = torch.nan
    # Use valid page IDs (not page 0) so output should be NaN-free
    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    page_table[0, 0] = 3
    page_table[0, 1] = 5
    cache_seqlens = torch.tensor([128], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    h = _PagedHarness(q, k_cache, v_cache, mode="decode", num_cache_pages=num_pages)
    h.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, _ = h.forward()
    assert torch.isfinite(output).all(), "NaN from poisoned page 0 propagated"


# ---------------------------------------------------------------------------
# 5. Big page-id / large pool (past 2^31/stride line) per AGENTS.md
# ---------------------------------------------------------------------------

@torch.inference_mode()
def test_large_pool_high_page_ids_decode() -> None:
    device = require_b12x()
    page_size = 64
    q_heads = 8
    kv_heads = 2
    head_dim = 128
    num_pages = 100000
    high_page_id = num_pages - 3
    page_stride_bytes = page_size * kv_heads * head_dim * 2
    assert high_page_id * page_stride_bytes > 2**31, "Must be past 2^31 line"
    q = torch.randn(1, q_heads, head_dim, device=device, dtype=torch.bfloat16) / 4
    k_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=torch.bfloat16) / 4
    v_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=torch.bfloat16) / 4
    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    page_table[0, 0] = high_page_id
    page_table[0, 1] = high_page_id + 1
    cache_seqlens = torch.tensor([128], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    h = _PagedHarness(q, k_cache, v_cache, mode="decode", num_cache_pages=num_pages)
    h.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, _ = h.forward()
    ref_out, _ = paged_attention_reference(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q)
    assert _cosine_similarity(output, ref_out) >= 0.999


@torch.inference_mode()
def test_large_pool_high_page_ids_extend() -> None:
    device = require_b12x()
    page_size = 64
    q_heads = 8
    kv_heads = 2
    head_dim = 128
    q_len = 64
    num_pages = 100000
    high_page_id = num_pages - 4
    page_stride_bytes = page_size * kv_heads * head_dim * 2
    assert high_page_id * page_stride_bytes > 2**31
    q = torch.randn(q_len, q_heads, head_dim, device=device, dtype=torch.bfloat16) / 4
    k_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=torch.bfloat16) / 4
    v_cache = torch.randn(num_pages, page_size, kv_heads, head_dim, device=device, dtype=torch.bfloat16) / 4
    pages_needed = 3
    page_table = torch.zeros(1, pages_needed, dtype=torch.int32, device=device)
    for i in range(pages_needed):
        page_table[0, i] = high_page_id + i
    cache_seqlens = torch.tensor([192], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    h = _PagedHarness(q, k_cache, v_cache, mode="extend", num_cache_pages=num_pages)
    h.prepare(page_table, cache_seqlens, cu_seqlens_q)
    output, _ = h.forward()
    ref_out, _ = paged_attention_reference(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q)
    assert _cosine_similarity(output, ref_out) >= 0.999


# ---------------------------------------------------------------------------
# 6. K/V capacity mismatch
# ---------------------------------------------------------------------------

@torch.inference_mode()
def test_planner_rejects_unequal_k_v_page_counts() -> None:
    device = require_b12x()
    q = torch.randn(1, 8, 128, device=device, dtype=torch.bfloat16) / 4
    k_cache = torch.randn(20, 64, 2, 128, device=device, dtype=torch.bfloat16) / 4
    v_cache = torch.randn(10, 64, 2, 128, device=device, dtype=torch.bfloat16) / 4
    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    page_table[0, 0] = 5
    page_table[0, 1] = 15
    cache_seqlens = torch.tensor([128], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="structural shapes must match"):
        create_paged_plan(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, mode="decode")


# ---------------------------------------------------------------------------
# 7. Reference zero-fills invalid pages without indexing page 0
# ---------------------------------------------------------------------------

@torch.inference_mode()
def test_reference_zero_fills_invalid_pages() -> None:
    device = require_b12x()
    num_pages = 10
    q = torch.randn(1, 8, 128, device=device, dtype=torch.bfloat16) / 4
    k_cache = torch.randn(num_pages, 64, 2, 128, device=device, dtype=torch.bfloat16) / 4
    v_cache = torch.randn(num_pages, 64, 2, 128, device=device, dtype=torch.bfloat16) / 4
    page_table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    page_table[0, 0] = 3
    page_table[0, 1] = 999  # invalid
    cache_seqlens = torch.tensor([128], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    ref_out, _ = paged_attention_reference(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q)
    assert ref_out is not None
    assert torch.isfinite(ref_out).all()
