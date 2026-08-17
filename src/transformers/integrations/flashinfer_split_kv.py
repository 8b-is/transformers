# Copyright 2026 The HuggingFace Team & 8b-is. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ultra-High-Speed Split-KV Flash-Decoding Attention for 32k+ Long-Context Inference.

Splits long KV-caches along the sequence dimension into independent partitions,
maximizing GPU SM occupancy on Hopper (H100/H200) and Blackwell (B200), followed
by online log-sum-exp reduction.
"""

from __future__ import annotations

import math
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton Kernels: Split-KV Stage 1 (Partial Attention) & Stage 2 (Log-Sum-Exp Reduction)
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _split_kv_stage1_kernel(
        Q_ptr, K_ptr, V_ptr,
        Mid_O_ptr, Mid_LSE_ptr,
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_os, stride_om, stride_od,
        stride_lb, stride_lh, stride_ls, stride_lm,
        seq_len,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Stage 1: Computes partial attention per KV-split block."""
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        split_idx = tl.program_id(2)

        # Offsets
        q_offset = batch_idx * stride_qb + head_idx * stride_qh
        offs_d = tl.arange(0, BLOCK_D)
        q = tl.load(Q_ptr + q_offset + offs_d * stride_qd)

        # Split range
        start_n = split_idx * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        # Load K, V for this split
        k_offset = batch_idx * stride_kb + head_idx * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_offset = batch_idx * stride_vb + head_idx * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

        k = tl.load(K_ptr + k_offset, mask=mask_n[:, None], other=0.0)
        v = tl.load(V_ptr + v_offset, mask=mask_n[:, None], other=0.0)

        # Dot product Q * K^T
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
        qk = tl.where(mask_n, qk, float("-inf"))

        # Local Softmax stats
        m_local = tl.max(qk, axis=0)
        p = tl.exp(qk - m_local)
        l_local = tl.sum(p, axis=0)
        p = tl.where(mask_n, p, 0.0)

        # Weighted value output
        acc = tl.sum(p[:, None] * v, axis=0)
        lse = m_local + tl.log(l_local)

        # Store intermediate split results
        mid_o_offset = (
            batch_idx * stride_ob
            + head_idx * stride_oh
            + split_idx * stride_os
            + offs_d * stride_od
        )
        tl.store(Mid_O_ptr + mid_o_offset, acc)

        mid_lse_offset = (
            batch_idx * stride_lb
            + head_idx * stride_lh
            + split_idx * stride_ls
        )
        tl.store(Mid_LSE_ptr + mid_lse_offset, lse)


    @triton.jit
    def _split_kv_stage2_kernel(
        Mid_O_ptr, Mid_LSE_ptr, Out_ptr,
        num_splits,
        stride_ob, stride_oh, stride_os, stride_om, stride_od,
        stride_lb, stride_lh, stride_ls, stride_lm,
        stride_final_b, stride_final_h, stride_final_m, stride_final_d,
        BLOCK_SPLITS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Stage 2: Online log-sum-exp reduction across all KV-splits."""
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        offs_s = tl.arange(0, BLOCK_SPLITS)
        offs_d = tl.arange(0, BLOCK_D)
        mask_s = offs_s < num_splits

        # Load all partial LSEs
        lse_offset = batch_idx * stride_lb + head_idx * stride_lh + offs_s * stride_ls
        lse = tl.load(Mid_LSE_ptr + lse_offset, mask=mask_s, other=float("-inf"))

        # Global max LSE
        max_lse = tl.max(lse, axis=0)
        weights = tl.exp(lse - max_lse)
        weights = tl.where(mask_s, weights, 0.0)
        sum_weights = tl.sum(weights, axis=0)

        # Load partial outputs and reduce
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for s in range(num_splits):
            w = tl.load(Mid_LSE_ptr + batch_idx * stride_lb + head_idx * stride_lh + s * stride_ls)
            w_norm = tl.exp(w - max_lse) / sum_weights

            mid_o_offset = batch_idx * stride_ob + head_idx * stride_oh + s * stride_os + offs_d * stride_od
            partial_o = tl.load(Mid_O_ptr + mid_o_offset)
            acc += partial_o * w_norm

        # Store final output
        final_offset = batch_idx * stride_final_b + head_idx * stride_final_h + offs_d * stride_final_d
        tl.store(Out_ptr + final_offset, acc)


# ---------------------------------------------------------------------------
# High-Level Vectorized Split-KV Decoding Attention
# ---------------------------------------------------------------------------

def split_kv_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float | None = None,
    split_size: int = 512,
) -> torch.Tensor:
    """
    High-Throughput Split-KV (Flash-Decoding) Attention for single-query decoding (Q_len = 1).
    
    Splits long sequence dimension (e.g. 32k - 128k) into parallel partitions to fully
    saturate GPU streaming multiprocessors (SMs), followed by numerically stable log-sum-exp reduction.
    
    Args:
        q: Query tensor of shape (batch_size, num_heads, 1, head_dim).
        k: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim).
        v: Value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim).
        sm_scale: Softmax scale factor (default: 1 / sqrt(head_dim)).
        split_size: Number of tokens per parallel KV split partition (default: 512).
    
    Returns:
        Output tensor of shape (batch_size, num_heads, 1, head_dim).
    """
    batch_size, num_heads, q_len, head_dim = q.shape
    seq_len = k.shape[-2]
    num_kv_heads = k.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # GQA / MQA Head Expansion if num_heads != num_kv_heads
    if num_heads != num_kv_heads:
        repeat_factor = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    # Fast-path for short sequences (<= split_size): standard fused SDPA
    if seq_len <= split_size or not q.is_cuda:
        # Vectorized split-reduction simulation / fallback
        num_splits = max(1, math.ceil(seq_len / split_size))
        
        if num_splits == 1:
            scores = torch.matmul(q, k.transpose(-1, -2)) * sm_scale
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        # Numerically stable multi-split log-sum-exp reduction
        partial_norm_outputs = []
        partial_lses = []

        for s in range(num_splits):
            s_start = s * split_size
            s_end = min(seq_len, s_start + split_size)
            k_chunk = k[:, :, s_start:s_end, :]
            v_chunk = v[:, :, s_start:s_end, :]

            chunk_scores = torch.matmul(q, k_chunk.transpose(-1, -2)) * sm_scale
            m_chunk = torch.max(chunk_scores, dim=-1, keepdim=True).values
            exp_scores = torch.exp(chunk_scores - m_chunk)
            l_chunk = torch.sum(exp_scores, dim=-1, keepdim=True)
            o_chunk = torch.matmul(exp_scores, v_chunk) / (l_chunk + 1e-8)

            lse_chunk = m_chunk + torch.log(l_chunk + 1e-8)
            partial_norm_outputs.append(o_chunk)
            partial_lses.append(lse_chunk)

        # Stage 2: Global Reduction via LSE Softmax Weights
        stacked_lse = torch.cat(partial_lses, dim=-1)  # (B, H, 1, num_splits)
        max_lse = torch.max(stacked_lse, dim=-1, keepdim=True).values
        weights = torch.softmax(stacked_lse - max_lse, dim=-1)

        final_out = torch.zeros_like(q)
        for s in range(num_splits):
            w_s = weights[:, :, :, s : s + 1]
            final_out = final_out + (partial_norm_outputs[s] * w_s)

        return final_out

    # Triton GPU Kernel execution on CUDA
    num_splits = math.ceil(seq_len / split_size)
    mid_o = torch.empty((batch_size, num_heads, num_splits, head_dim), dtype=q.dtype, device=q.device)
    mid_lse = torch.empty((batch_size, num_heads, num_splits), dtype=torch.float32, device=q.device)
    out = torch.empty((batch_size, num_heads, 1, head_dim), dtype=q.dtype, device=q.device)

    # Grid config
    grid_stage1 = (batch_size, num_heads, num_splits)
    _split_kv_stage1_kernel[grid_stage1](
        q, k, v,
        mid_o, mid_lse,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), 0, mid_o.stride(3),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), 0,
        seq_len,
        BLOCK_N=split_size,
        BLOCK_D=head_dim,
    )

    grid_stage2 = (batch_size, num_heads)
    _split_kv_stage2_kernel[grid_stage2](
        mid_o, mid_lse, out,
        num_splits,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), 0, mid_o.stride(3),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2), 0,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        BLOCK_D=head_dim,
    )

    return out
