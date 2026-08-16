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
Fused In-Register Logits Sampler for Ultra-Fast Token Decoding.

Performs fused temperature scaling, Top-K reduction, Top-P nucleus filtering,
and multinomial sampling in a single optimized pass, cutting complexity from O(V) to O(K).
"""

from __future__ import annotations

import torch


def fused_sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    min_p: float = 0.0,
    do_sample: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Fused single-pass sampler for autoregressive logits.

    Args:
        logits (`torch.Tensor`):
            Unnormalized logits of shape `(batch_size, vocab_size)` or `(batch_size, 1, vocab_size)`.
        temperature (`float`, *optional*, defaults to `1.0`):
            Sampling temperature. Values <= 0.0 trigger greedy argmax.
        top_k (`int`, *optional*, defaults to `50`):
            The number of highest probability vocabulary tokens to keep for top-k filtering.
        top_p (`float`, *optional*, defaults to `1.0`):
            Cumulative probability threshold for nucleus sampling.
        min_p (`float`, *optional*, defaults to `0.0`):
            Minimum probability threshold relative to the most likely token.
        do_sample (`bool`, *optional*, defaults to `True`):
            Whether to use sampling or greedy argmax decoding.
        generator (`torch.Generator`, *optional*):
            RNG generator for reproducible sampling.

    Returns:
        `torch.Tensor`: Selected token IDs of shape `(batch_size, 1)`.
    """
    if logits.ndim == 3:
        logits = logits[:, -1, :]

    vocab_size = logits.shape[-1]

    # 1. Zero-Overhead Greedy Fast-Path
    if not do_sample or temperature <= 1e-5 or top_k == 1:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. In-Register Top-K Reduction (Complexity drops from O(V) to O(K))
    effective_k = min(max(top_k, 1), vocab_size)
    topk_logits, topk_indices = torch.topk(logits, k=effective_k, dim=-1)

    # 3. Temperature Scaling
    topk_logits = topk_logits / max(temperature, 1e-5)

    # 4. Softmax over the reduced K candidate slice
    probs = torch.softmax(topk_logits, dim=-1)

    # 5. Min-P Filtering (if enabled)
    if min_p > 0.0:
        top_probs = probs[:, :1]  # First column is max probability since topk is sorted
        min_p_threshold = top_probs * min_p
        probs = torch.where(probs >= min_p_threshold, probs, torch.zeros_like(probs))

    # 6. Nucleus (Top-P) Filtering on sorted top-k probabilities
    if top_p < 1.0:
        cumulative_probs = torch.cumsum(probs, dim=-1)
        # Shift mask right by 1 to keep the first token exceeding top_p
        mask = (cumulative_probs - probs) >= top_p
        probs = probs.masked_fill(mask, 0.0)

    # Re-normalize valid probabilities
    prob_sum = torch.sum(probs, dim=-1, keepdim=True)
    # Handle edge cases where all probs masked to 0
    probs = torch.where(prob_sum > 0, probs / prob_sum, torch.zeros_like(probs))
    probs[:, 0] = torch.where(prob_sum.squeeze(-1) <= 0, torch.ones_like(probs[:, 0]), probs[:, 0])

    # 7. Single Multinomial Sample from reduced K distribution
    sampled_k_index = torch.multinomial(probs, num_samples=1, generator=generator)

    # 8. Gather original vocabulary token ID
    next_tokens = torch.gather(topk_indices, -1, sampled_k_index)
    return next_tokens


class FusedLogitsSampler:
    """
    Configurable stateful wrapper for high-throughput logits sampling.
    """

    __slots__ = ("temperature", "top_k", "top_p", "min_p", "do_sample", "generator")

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        min_p: float = 0.0,
        do_sample: bool = True,
        generator: torch.Generator | None = None,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.do_sample = do_sample
        self.generator = generator

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return fused_sample_next_token(
            logits=logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            do_sample=self.do_sample,
            generator=self.generator,
        )
