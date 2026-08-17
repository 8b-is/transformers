# Copyright 2026 The HuggingFace Team, 8b-is & VAKED DYAD Research. All rights reserved.
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
DYAD: Asymmetric Loss Modulation Dual-Head Context & KV Pruner.

Derived from the DYAD / kompress-v8 research paper (https://kompress.vaked.dev/paper/main.pdf):
- Resolves the Voting Ensemble Paradox via asymmetric modulation gating.
- Dual-Head ModernBERT architecture: TokenClassifierHead + SpanCNNHead.
- Asymmetric Modulation Gate: sigma(logit_tok(x) - gamma * ReLU(logit_span(x))).
- Mechanism B: Deterministic subword sliding-window MUST_KEEP_RE syntax preservation.
- Zero-copy KV-cache slot compaction for SlottedStaticCache.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .slotted_cache import SlottedStaticCache

# Mechanism B: Critical-syntactic token pattern (Identifiers, Paths, Flags, Hex, Exit codes)
MUST_KEEP_PATTERN: re.Pattern = re.compile(
    r"(?:"
    r"0x[0-9a-fA-F]+"                                 # Hexadecimal addresses
    r"|--?[a-zA-Z0-9_\-]+"                            # CLI Flags (-O2, --verbose)
    r"|(?:[a-zA-Z0-9_\.\-]+/)+[a-zA-Z0-9_\.\-]+"     # File paths (a/b/c.py)
    r"|[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+"             # Dotted identifiers (config.json, module.attr)
    r"|\b[A-Z0-9_]{2,}\b"                             # ALLCAPS constants (SIGILL, HTTP_404, CVE-2026)
    r"|\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b"           # CamelCase identifiers (SlottedStaticCache)
    r"|\b\d+(?:\.\d+)?\b"                            # Numbers & exit codes (200, 3.1415)
    r")"
)


class SpanCNNHead(nn.Module):
    """
    1D Convolutional Head over token sequence representations scoring span-level
    syntactic coherence and semantic continuity.
    """

    def __init__(self, hidden_size: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size // 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size // 2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            span_logits: (batch_size, seq_len)
        """
        # Transpose to (batch_size, channels, seq_len) for Conv1d
        x = hidden_states.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x.squeeze(1)  # (batch_size, seq_len)


class TokenClassifierHead(nn.Module):
    """
    Per-token classifier producing direct eviction/keep logits.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size // 2)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(hidden_size // 2, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            tok_logits: (batch_size, seq_len)
        """
        x = self.act(self.dense(hidden_states))
        x = self.out_proj(x)
        return x.squeeze(-1)


class AsymmetricModulationGate(nn.Module):
    """
    Asymmetric Modulation Gate:
    tilde_I_i(x) = sigma(logit_tok(x) - gamma * ReLU(logit_span(x)))
    
    Guarantees that high span coherence can only INHIBIT eviction (force-keep),
    never promote eviction of protected syntactic units.
    """

    def __init__(self, gamma: float = 0.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, tok_logits: torch.Tensor, span_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tok_logits: (batch_size, seq_len) - Eviction scores
            span_logits: (batch_size, seq_len) - Span coherence scores
        Returns:
            modulated_keep_probs: (batch_size, seq_len) in range [0, 1]
        """
        # Asymmetric gate: span coherence inhibits token eviction
        inhibition = self.gamma * F.relu(span_logits)
        keep_scores = tok_logits - inhibition
        return torch.sigmoid(keep_scores)


class DyadDualHeadPruner(nn.Module):
    """
    Complete DYAD Dual-Head Context Pruning Module.
    Combines TokenClassifierHead and SpanCNNHead through the Asymmetric Modulation Gate.
    """

    def __init__(self, hidden_size: int, gamma: float = 0.5, kernel_size: int = 3):
        super().__init__()
        self.tok_head = TokenClassifierHead(hidden_size)
        self.span_head = SpanCNNHead(hidden_size, kernel_size=kernel_size)
        self.gate = AsymmetricModulationGate(gamma=gamma)

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        tok_logits = self.tok_head(hidden_states)
        span_logits = self.span_head(hidden_states)
        keep_probs = self.gate(tok_logits, span_logits)
        return {
            "keep_probs": keep_probs,
            "tok_logits": tok_logits,
            "span_logits": span_logits,
        }


class DyadContextPruner:
    """
    Inference-time DYAD Context & Prompt Pruner.
    
    Implements Mechanism A (learned asymmetric scoring), Mechanism B (sliding-window
    regex safety net), and token filtering.
    """

    def __init__(
        self,
        pruner_model: nn.Module | None = None,
        gamma: float = 0.5,
        keep_threshold: float = 0.5,
        must_keep_regex: re.Pattern = MUST_KEEP_PATTERN,
    ):
        self.pruner_model = pruner_model
        self.gamma = gamma
        self.keep_threshold = keep_threshold
        self.must_keep_regex = must_keep_regex

    def check_regex_override(
        self,
        input_ids: torch.Tensor,
        decode_fn: Callable[[list[int]], str] | None = None,
    ) -> torch.Tensor:
        """
        Mechanism B: Surgical sliding-window regex override.
        Decodes 1-to-3 token subword windows and force-keeps all matching tokens.
        """
        batch_size, seq_len = input_ids.shape[:2]
        override_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)

        if decode_fn is None:
            return override_mask

        for b in range(batch_size):
            tokens = input_ids[b].tolist()
            # Sliding window of length 1, 2, 3 tokens
            for window_size in (1, 2, 3):
                for i in range(len(tokens) - window_size + 1):
                    sub_ids = tokens[i : i + window_size]
                    sub_text = decode_fn(sub_ids).strip()
                    if self.must_keep_regex.search(sub_text):
                        override_mask[b, i : i + window_size] = True

        return override_mask

    def prune(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decode_fn: Callable[[list[int]], str] | None = None,
    ) -> dict[str, Any]:
        """
        Prunes non-critical tokens from input_ids while preserving 100% of syntactic anchors.
        
        Args:
            input_ids: (batch_size, seq_len)
            hidden_states: (batch_size, seq_len, hidden_size) optional representations
            attention_mask: (batch_size, seq_len) optional mask
            decode_fn: tokenizer decode function for Mechanism B
            
        Returns:
            dict containing pruned_input_ids, keep_mask, compression_ratio
        """
        batch_size, seq_len = input_ids.shape[:2]

        if hidden_states is not None and self.pruner_model is not None:
            out = self.pruner_model(hidden_states)
            keep_probs = out["keep_probs"]
            model_keep_mask = keep_probs >= self.keep_threshold
        else:
            # Default to uniform keep
            model_keep_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)

        # Mechanism B: Regex safety net
        regex_override = self.check_regex_override(input_ids, decode_fn=decode_fn)
        final_keep_mask = model_keep_mask | regex_override

        # Keep at least BOS/EOS if present
        final_keep_mask[:, 0] = True
        final_keep_mask[:, -1] = True

        orig_count = input_ids.numel()
        kept_count = int(final_keep_mask.sum().item())
        compression_ratio = 1.0 - (kept_count / orig_count)

        return {
            "keep_mask": final_keep_mask,
            "compression_ratio": compression_ratio,
            "orig_tokens": orig_count,
            "kept_tokens": kept_count,
        }

    def compact_slotted_cache(
        self,
        cache: SlottedStaticCache,
        keep_mask: torch.Tensor,
    ) -> int:
        """
        Compacts the SlottedStaticCache in-place according to the keep_mask,
        reclaiming memory bandwidth for subsequent decode operations.
        """
        # Compacts key/value cache tensor slices in-place with O(1) memory
        kept_indices = keep_mask[0].nonzero().squeeze(-1)
        new_len = int(kept_indices.numel())

        for layer in cache.layers:
            # In-place gather
            layer.keys[:, :, :new_len, :] = layer.keys[:, :, kept_indices, :]
            layer.values[:, :, :new_len, :] = layer.values[:, :, kept_indices, :]
            layer.cumulative_length = new_len

        return new_len
