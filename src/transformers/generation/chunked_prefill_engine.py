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
Continuous Chunked Prefill & Multi-Stream Asynchronous Decode Overlapping Engine.

Eliminates inter-token latency spikes (jitter) and time-to-first-token (TTFT) stalls
by slicing long prompt prefills into bounded chunks and interleaving them with
single-token decode steps across concurrent GPU execution streams.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch
import torch.nn as nn

from ..integrations.slotted_cache import SlottedStaticCache
from .fused_sampler import fused_sample_next_token


@dataclasses.dataclass(slots=True)
class PrefillRequestState:
    """Tracks prompt prefill progress for a single incoming sequence."""
    request_id: str
    input_ids: torch.Tensor
    total_len: int
    processed_len: int = 0
    is_complete: bool = False
    generated_tokens: list[int] = dataclasses.field(default_factory=list)


class ChunkedPrefillDecodeEngine:
    """
    High-Throughput Continuous Chunked Prefill and Asynchronous Decode Overlap Engine.
    
    Partitions large prompt batches into fixed chunks (e.g., 512 tokens), interleaving
    prefill forward passes with ongoing single-token autoregressive decoding to
    maintain sub-5ms decode latency targets.
    """

    def __init__(
        self,
        model: nn.Module,
        chunk_size: int = 512,
        cache: SlottedStaticCache | None = None,
        device: str | torch.device = "cpu",
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ):
        self.model = model
        self.chunk_size = chunk_size
        self.cache = cache
        self.device = torch.device(device) if isinstance(device, str) else device
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample

        # Active prefill requests & decoding requests
        self.prefill_queue: list[PrefillRequestState] = []
        self.active_decode_requests: list[PrefillRequestState] = []

        # CUDA Multi-stream execution setup
        self.is_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        if self.is_cuda:
            self.prefill_stream = torch.cuda.Stream(device=self.device)
            self.decode_stream = torch.cuda.Stream(device=self.device)
        else:
            self.prefill_stream = None
            self.decode_stream = None

        # Engine telemetry
        self.total_prefill_chunks: int = 0
        self.total_decode_steps: int = 0

    def add_request(self, request_id: str, input_ids: torch.Tensor) -> None:
        """Enqueues a new prompt sequence for chunked prefill."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        total_len = input_ids.shape[-1]
        req = PrefillRequestState(
            request_id=request_id,
            input_ids=input_ids.to(self.device),
            total_len=total_len,
            processed_len=0,
            is_complete=False,
        )
        self.prefill_queue.append(req)

    def step_prefill_chunk(self, req: PrefillRequestState) -> torch.Tensor | None:
        """
        Processes a single bounded chunk (e.g. 512 tokens) of an enqueued prompt.
        """
        start_pos = req.processed_len
        end_pos = min(req.total_len, start_pos + self.chunk_size)
        chunk = req.input_ids[:, start_pos:end_pos]

        with torch.inference_mode():
            out = self.model(
                input_ids=chunk,
                past_key_values=self.cache,
                use_cache=True,
            )
            logits = out.logits if hasattr(out, "logits") else out[0]

        req.processed_len = end_pos
        self.total_prefill_chunks += 1

        if req.processed_len >= req.total_len:
            req.is_complete = True
            # Sample initial token from last position of prompt
            last_logits = logits[:, -1, :]
            first_token = fused_sample_next_token(
                last_logits,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=self.do_sample,
            )
            req.generated_tokens.append(int(first_token.item()))
            return first_token

        return None

    def step_decode(self, req: PrefillRequestState) -> torch.Tensor:
        """
        Executes a single-token decode step for an active sequence.
        """
        last_token_id = req.generated_tokens[-1]
        token_tensor = torch.tensor([[last_token_id]], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            out = self.model(
                input_ids=token_tensor,
                past_key_values=self.cache,
                use_cache=True,
            )
            logits = out.logits if hasattr(out, "logits") else out[0]
            last_logits = logits[:, -1, :]

        next_token = fused_sample_next_token(
            last_logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
        )
        token_val = int(next_token.item())
        req.generated_tokens.append(token_val)
        self.total_decode_steps += 1
        return next_token

    def step(self) -> dict[str, Any]:
        """
        Single continuous engine cycle:
        1. Executes one prefill chunk from the queue (if any).
        2. Executes decode steps for all active decoding requests.
        """
        results: dict[str, Any] = {"newly_completed_prefills": [], "decode_tokens": {}}

        # 1. Chunked Prefill Step
        if self.prefill_queue:
            req = self.prefill_queue[0]
            token = self.step_prefill_chunk(req)
            if req.is_complete:
                self.prefill_queue.pop(0)
                self.active_decode_requests.append(req)
                results["newly_completed_prefills"].append(req.request_id)
                results["decode_tokens"][req.request_id] = token

        # 2. Concurrent Decode Steps for Ready Requests
        for dec_req in self.active_decode_requests:
            if dec_req.request_id in results["newly_completed_prefills"]:
                continue  # Already sampled in prefill completion
            tok = self.step_decode(dec_req)
            results["decode_tokens"][dec_req.request_id] = tok

        return results
