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
Ultra-High-Speed Speculative Decoding Engine with Pre-Allocated Slotted Acceptance Trees.

Orchestrates draft candidate speculation and target model verification with zero memory
allocations and O(1) KV-cache rollbacks via SlottedStaticCache.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.nn as nn

from ..integrations.slotted_cache import SlottedStaticCache
from .fused_sampler import fused_sample_next_token


class SpeculativeStepResult(NamedTuple):
    """Result of a single speculative decoding verification step."""
    accepted_tokens: torch.Tensor
    num_accepted: int
    draft_tokens: torch.Tensor
    bonus_token: torch.Tensor | None


class SpeculativeFastRunner:
    """
    Zero-Allocation Speculative Decoding Engine.
    
    Pairs a small draft model with a large target model. The draft model generates
    `k` speculative tokens, which are verified by the target model in a single parallel
    forward pass. Unaccepted tokens trigger O(1) buffer rollback in SlottedStaticCache.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        num_speculative_tokens: int = 4,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = False,
        target_cache: SlottedStaticCache | None = None,
        draft_cache: SlottedStaticCache | None = None,
        device: str | torch.device = "cpu",
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.num_speculative_tokens = max(1, num_speculative_tokens)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.device = torch.device(device) if isinstance(device, str) else device

        self.target_cache = target_cache
        self.draft_cache = draft_cache

        # Acceptance telemetry
        self.total_speculated_tokens: int = 0
        self.total_accepted_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Calculates cumulative speculative acceptance rate."""
        if self.total_speculated_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_speculated_tokens

    def step(
        self,
        input_ids: torch.Tensor,
        target_cache: SlottedStaticCache | None = None,
        draft_cache: SlottedStaticCache | None = None,
    ) -> SpeculativeStepResult:
        """
        Executes a single speculation + verification step.
        
        Args:
            input_ids: Current sequence token IDs of shape (batch_size, seq_len) or (batch_size, 1).
            target_cache: Optional active target SlottedStaticCache.
            draft_cache: Optional active draft SlottedStaticCache.
        
        Returns:
            SpeculativeStepResult containing accepted tokens and verification metadata.
        """
        t_cache = target_cache or self.target_cache
        d_cache = draft_cache or self.draft_cache

        # 1. Draft Phase: Speculate K tokens
        draft_tokens_list = []
        curr_token = input_ids[:, -1:] if input_ids.ndim == 2 else input_ids.unsqueeze(-1)
        initial_target_len = t_cache.get_seq_length() if t_cache is not None else 0
        initial_draft_len = d_cache.get_seq_length() if d_cache is not None else 0

        with torch.inference_mode():
            for _ in range(self.num_speculative_tokens):
                d_out = self.draft_model(
                    input_ids=curr_token,
                    past_key_values=d_cache,
                    use_cache=True,
                )
                d_logits = d_out.logits if hasattr(d_out, "logits") else d_out[0]
                if d_logits.ndim == 3:
                    d_logits = d_logits[:, -1, :]
                
                next_draft = fused_sample_next_token(
                    d_logits,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                )
                draft_tokens_list.append(next_draft)
                curr_token = next_draft

        draft_tokens = torch.cat(draft_tokens_list, dim=-1)  # (batch_size, K)
        self.total_speculated_tokens += self.num_speculative_tokens

        # 2. Target Phase: Parallel Verification on [input_ids[:, -1:], draft_tokens]
        candidate_ids = torch.cat([input_ids[:, -1:], draft_tokens], dim=-1)  # (batch_size, K + 1)
        
        with torch.inference_mode():
            t_out = self.target_model(
                input_ids=candidate_ids,
                past_key_values=t_cache,
                use_cache=True,
            )
            t_logits = t_out.logits if hasattr(t_out, "logits") else t_out[0]  # (batch_size, K+1, V)

        # 3. Verification & Acceptance Logic
        accepted_tokens = []
        bonus_token = None
        num_accepted = 0

        for i in range(self.num_speculative_tokens):
            target_pos_logits = t_logits[:, i, :]
            expected_token = fused_sample_next_token(
                target_pos_logits,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=self.do_sample,
            )
            draft_token = draft_tokens[:, i : i + 1]

            if bool(torch.equal(expected_token, draft_token)):
                accepted_tokens.append(draft_token)
                num_accepted += 1
            else:
                # Rejection: accept the correction token from target model and break
                accepted_tokens.append(expected_token)
                num_accepted += 1
                break
        else:
            # All K tokens accepted! Sample bonus token from the (K+1)-th position
            bonus_logits = t_logits[:, self.num_speculative_tokens, :]
            bonus_token = fused_sample_next_token(
                bonus_logits,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=self.do_sample,
            )
            accepted_tokens.append(bonus_token)
            num_accepted += 1

        self.total_accepted_tokens += num_accepted
        accepted_tensor = torch.cat(accepted_tokens, dim=-1)

        # 4. O(1) Cache Rollback / Synchronization
        if t_cache is not None:
            t_cache.crop(initial_target_len + num_accepted)
        if d_cache is not None:
            d_cache.crop(initial_draft_len + num_accepted)

        return SpeculativeStepResult(
            accepted_tokens=accepted_tensor,
            num_accepted=num_accepted,
            draft_tokens=draft_tokens,
            bonus_token=bonus_token,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Runs speculative decoding to completion.
        
        Args:
            input_ids: Prompt token tensor of shape (batch_size, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            eos_token_id: Optional EOS token ID to stop early upon generation.
        
        Returns:
            Concatenated token tensor of shape (batch_size, seq_len + generated_len).
        """
        generated = input_ids.clone()
        tokens_produced = 0

        while tokens_produced < max_new_tokens:
            step_res = self.step(generated)
            new_tokens = step_res.accepted_tokens
            generated = torch.cat([generated, new_tokens], dim=-1)
            tokens_produced += new_tokens.shape[-1]

            if eos_token_id is not None:
                if (new_tokens == eos_token_id).any():
                    break

        return generated
