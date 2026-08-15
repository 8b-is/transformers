# Copyright 2026 The HuggingFace Team / 8b.IS. All rights reserved.
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

import unittest
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class SpecialTokensPreservationTest(unittest.TestCase):
    def test_additional_special_tokens_preservation_when_extra_empty(self):
        """Fixes Issue #47838: additional_special_tokens should not be dropped when extra_special_tokens is empty."""
        init_kwargs = {
            "additional_special_tokens": ["<|media_start|>", "<|media_content|>", "<|media_end|>"],
            "extra_special_tokens": {},
        }
        # Simulate V5 conversion in _from_pretrained
        add_toks = init_kwargs.pop("additional_special_tokens")
        if "extra_special_tokens" in init_kwargs and init_kwargs["extra_special_tokens"]:
            existing = init_kwargs["extra_special_tokens"]
            if isinstance(existing, list) and isinstance(add_toks, (list, tuple)):
                init_kwargs["extra_special_tokens"] = existing + list(add_toks)
            elif isinstance(existing, dict) and isinstance(add_toks, dict):
                existing.update(add_toks)
        else:
            init_kwargs["extra_special_tokens"] = add_toks

        self.assertIn("extra_special_tokens", init_kwargs)
        self.assertEqual(
            init_kwargs["extra_special_tokens"],
            ["<|media_start|>", "<|media_content|>", "<|media_end|>"],
        )

    def test_additional_special_tokens_merging_when_extra_present(self):
        """Fixes Issue #47838: additional_special_tokens and extra_special_tokens merge safely."""
        init_kwargs = {
            "additional_special_tokens": ["<|token_b|>"],
            "extra_special_tokens": ["<|token_a|>"],
        }
        add_toks = init_kwargs.pop("additional_special_tokens")
        if "extra_special_tokens" in init_kwargs and init_kwargs["extra_special_tokens"]:
            existing = init_kwargs["extra_special_tokens"]
            if isinstance(existing, list) and isinstance(add_toks, (list, tuple)):
                init_kwargs["extra_special_tokens"] = existing + list(add_toks)
            elif isinstance(existing, dict) and isinstance(add_toks, dict):
                existing.update(add_toks)
        else:
            init_kwargs["extra_special_tokens"] = add_toks

        self.assertEqual(init_kwargs["extra_special_tokens"], ["<|token_a|>", "<|token_b|>"])
