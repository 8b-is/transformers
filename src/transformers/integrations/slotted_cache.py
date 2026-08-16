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
Ultra-High-Speed Slotted Static KV-Cache for Zero-Allocation Autoregressive Decoding.

Pre-allocates contiguous (batch_size, num_heads, max_cache_len, head_dim) buffers
on the target accelerator to eliminate dynamic CUDA memory allocations and VRAM fragmentation.
"""

from __future__ import annotations

from typing import Any

import torch

from ..cache_utils import Cache, CacheLayerMixin


class SlottedStaticLayer(CacheLayerMixin):
    """
    Individual static cache layer with pre-allocated contiguous key/value tensors.
    """

    is_compileable = True
    is_sliding = False

    def __init__(
        self,
        batch_size: int,
        num_key_value_heads: int,
        max_cache_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_heads = num_key_value_heads
        self.max_cache_len = max_cache_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        self.keys = torch.zeros(
            (batch_size, num_key_value_heads, max_cache_len, head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (batch_size, num_key_value_heads, max_cache_len, head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.cumulative_length = 0
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = key_states.shape[-2]
        cache_pos = None

        if cache_kwargs is not None and "cache_position" in cache_kwargs:
            cache_pos = cache_kwargs["cache_position"]

        if cache_pos is not None:
            self.keys[:, :, cache_pos, :] = key_states
            self.values[:, :, cache_pos, :] = value_states
            cur_end = int(cache_pos[-1].item()) + 1 if hasattr(cache_pos[-1], "item") else int(cache_pos[-1]) + 1
            self.cumulative_length = max(self.cumulative_length, cur_end)
        else:
            start_pos = self.cumulative_length
            end_pos = start_pos + seq_len
            if end_pos > self.max_cache_len:
                raise ValueError(
                    f"SlottedStaticLayer capacity exceeded: needed {end_pos} tokens, but max_cache_len is {self.max_cache_len}."
                )
            self.keys[:, :, start_pos:end_pos, :].copy_(key_states)
            self.values[:, :, start_pos:end_pos, :].copy_(value_states)
            self.cumulative_length = end_pos

        active_len = self.cumulative_length
        return (
            self.keys[:, :, :active_len, :],
            self.values[:, :, :active_len, :],
        )

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_length(self) -> int:
        return self.max_cache_len

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.cumulative_length, 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.is_initialized = True

    def reset(self) -> None:
        self.cumulative_length = 0

    def crop(self, max_length: int) -> None:
        self.cumulative_length = min(self.cumulative_length, max_length)


class SlottedStaticCache(Cache):
    """
    Zero-allocation static key-value cache designed for high-throughput LLM inference and CUDA Graphs.
    Pre-allocates contiguous memory buffers once and uses in-place slice copies (`copy_()`) during decoding.
    """

    def __init__(
        self,
        batch_size: int,
        num_key_value_heads: int,
        max_cache_len: int,
        head_dim: int,
        num_layers: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ):
        target_device = torch.device(device) if isinstance(device, str) else device
        layers = [
            SlottedStaticLayer(
                batch_size=batch_size,
                num_key_value_heads=num_key_value_heads,
                max_cache_len=max_cache_len,
                head_dim=head_dim,
                dtype=dtype,
                device=target_device,
            )
            for _ in range(num_layers)
        ]
        super().__init__(layers=layers)
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = target_device

    @property
    def key_cache(self) -> list[torch.Tensor]:
        return [layer.keys for layer in self.layers]

    @property
    def value_cache(self) -> list[torch.Tensor]:
        return [layer.values for layer in self.layers]

    @classmethod
    def from_config(
        cls,
        config: Any,
        batch_size: int = 1,
        max_cache_len: int = 2048,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> SlottedStaticCache:
        """Convenience constructor from PretrainedConfig."""
        text_config = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
        num_layers = getattr(text_config, "num_hidden_layers", getattr(text_config, "n_layer", 32))
        num_kv_heads = getattr(
            text_config,
            "num_key_value_heads",
            getattr(text_config, "num_attention_heads", getattr(text_config, "n_head", 32)),
        )
        hidden_size = getattr(text_config, "hidden_size", getattr(text_config, "n_embd", 4096))
        num_heads = getattr(text_config, "num_attention_heads", getattr(text_config, "n_head", 32))
        head_dim = getattr(text_config, "head_dim", hidden_size // num_heads)

        return cls(
            batch_size=batch_size,
            num_key_value_heads=num_kv_heads,
            max_cache_len=max_cache_len,
            head_dim=head_dim,
            num_layers=num_layers,
            dtype=dtype,
            device=device,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs=cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def crop(self, max_length: int) -> None:
        for layer in self.layers:
            layer.crop(max_length)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(
            (
                layer.keys[:, :, : layer.cumulative_length, :].clone(),
                layer.values[:, :, : layer.cumulative_length, :].clone(),
            )
            for layer in self.layers
        )
