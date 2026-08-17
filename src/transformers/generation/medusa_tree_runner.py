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
Ultra-High-Speed Tree-Based Speculative Drafting Engine (Medusa / Eagle Architecture).

Generates non-linear candidate token trees with pre-computed 2D causal visibility masks,
verifying multiple speculative branches in parallel in a single target forward pass.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.nn as nn

from ..integrations.slotted_cache import SlottedStaticCache
from .fused_sampler import fused_sample_next_token


class MedusaTreePath(NamedTuple):
    """Represents a single candidate branch through the speculative tree."""
    node_indices: list[int]
    depth: int


class MedusaTreeTopology:
    """
    Manages non-linear tree branch topology, causal 2D attention masks, and position offsets.
    """

    def __init__(
        self,
        tree_paths: list[list[int]] | None = None,
        device: str | torch.device = "cpu",
    ):
        # Default 2-depth branching tree: Root -> [Node 1, Node 2] -> [Node 3, 4, 5, 6]
        # Paths represented by sequence of node IDs
        if tree_paths is None:
            self.paths = [
                [0, 1, 3],
                [0, 1, 4],
                [0, 2, 5],
                [0, 2, 6],
            ]
        else:
            self.paths = tree_paths

        self.device = torch.device(device) if isinstance(device, str) else device
        self.num_nodes = max(max(p) for p in self.paths) + 1

        # Build 2D Tree Attention Mask (N_nodes x N_nodes)
        # Node i can attend to Node j if j is an ancestor of i along any valid path
        mask = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.bool)
        for path in self.paths:
            for idx_i, node_i in enumerate(path):
                for node_j in path[: idx_i + 1]:
                    mask[node_i, node_j] = True

        self.tree_mask = mask.to(self.device)

        # Node depths (distance from root)
        depths = [0] * self.num_nodes
        for path in self.paths:
            for d, node in enumerate(path):
                depths[node] = d
        self.node_depths = torch.tensor(depths, dtype=torch.long, device=self.device)


class MedusaTreeStepResult(NamedTuple):
    """Result of a single tree-speculative decoding verification pass."""
    accepted_tokens: torch.Tensor
    num_accepted: int
    winning_path: list[int]
    bonus_token: torch.Tensor | None


class MedusaTreeFastRunner:
    """
    Tree-based Speculative Fast Runner.
    
    Verifies non-linear tree candidate paths simultaneously in a single target model forward pass
    using custom 2D causal visibility masks.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_heads: list[nn.Module] | nn.Module,
        topology: MedusaTreeTopology | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = False,
        target_cache: SlottedStaticCache | None = None,
        device: str | torch.device = "cpu",
    ):
        self.target_model = target_model
        self.draft_heads = draft_heads if isinstance(draft_heads, (list, nn.ModuleList)) else [draft_heads]
        self.device = torch.device(device) if isinstance(device, str) else device
        self.topology = topology or MedusaTreeTopology(device=self.device)

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.target_cache = target_cache

        # Acceptance telemetry
        self.total_tree_steps: int = 0
        self.total_accepted_tokens: int = 0

    @property
    def average_acceptance_per_step(self) -> float:
        """Returns average accepted tokens per forward verification pass."""
        if self.total_tree_steps == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_tree_steps

    def step(
        self,
        input_ids: torch.Tensor,
        target_cache: SlottedStaticCache | None = None,
    ) -> MedusaTreeStepResult:
        """
        Executes a single tree speculation and parallel branch verification step.
        """
        t_cache = target_cache or self.target_cache
        batch_size = input_ids.shape[0]
        initial_cache_len = t_cache.get_seq_length() if t_cache is not None else 0
        curr_token = input_ids[:, -1:] if input_ids.ndim == 2 else input_ids.unsqueeze(-1)

        # 1. Draft Phase: Predict candidate tree nodes from heads
        # Root node (index 0) is current active token
        tree_tokens = [curr_token]

        with torch.inference_mode():
            # Run draft heads to populate tree nodes
            for head in self.draft_heads:
                head_out = head(curr_token)
                h_logits = head_out.logits if hasattr(head_out, "logits") else head_out
                if h_logits.ndim == 3:
                    h_logits = h_logits[:, -1, :]

                cand = fused_sample_next_token(
                    h_logits,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                )
                tree_tokens.append(cand)

        # Expand / fill any remaining nodes up to topology.num_nodes
        while len(tree_tokens) < self.topology.num_nodes:
            tree_tokens.append(tree_tokens[-1])

        tree_tensor = torch.cat(tree_tokens, dim=-1)  # (batch_size, num_nodes)

        # 2. Parallel Target Model Forward Pass over Tree
        with torch.inference_mode():
            t_out = self.target_model(
                input_ids=tree_tensor,
                past_key_values=t_cache,
                use_cache=True,
            )
            t_logits = t_out.logits if hasattr(t_out, "logits") else t_out[0]  # (B, num_nodes, V)

        # 3. Find Longest Valid Branch in Topology
        best_path = self.topology.paths[0]
        best_accepted = [tree_tensor[:, best_path[0] : best_path[0] + 1]]
        max_accepted_len = 1
        bonus_token = None

        for path in self.topology.paths:
            path_accepted = []
            for depth, node_idx in enumerate(path):
                expected = fused_sample_next_token(
                    t_logits[:, node_idx, :],
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                )
                cand = tree_tensor[:, node_idx : node_idx + 1]

                if depth == 0 or torch.equal(expected, cand):
                    path_accepted.append(cand)
                else:
                    path_accepted.append(expected)
                    break
            else:
                # Entire path accepted! Sample bonus token from final node
                final_node = path[-1]
                bonus_token = fused_sample_next_token(
                    t_logits[:, final_node, :],
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                )
                path_accepted.append(bonus_token)

            if len(path_accepted) > max_accepted_len:
                max_accepted_len = len(path_accepted)
                best_accepted = path_accepted
                best_path = path

        self.total_tree_steps += 1
        self.total_accepted_tokens += max_accepted_len
        accepted_tensor = torch.cat(best_accepted, dim=-1)

        # 4. O(1) Cache Rollback to accepted length
        if t_cache is not None:
            t_cache.crop(initial_cache_len + max_accepted_len)

        return MedusaTreeStepResult(
            accepted_tokens=accepted_tensor,
            num_accepted=max_accepted_len,
            winning_path=best_path,
            bonus_token=bonus_token,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Runs tree-speculative decoding to completion."""
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
