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

import unittest
import torch
import re

from transformers.integrations.dyad_compressor import (
    SpanCNNHead,
    TokenClassifierHead,
    AsymmetricModulationGate,
    DyadDualHeadPruner,
    DyadContextPruner,
    MUST_KEEP_PATTERN,
)


class DyadCompressorTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 16
        self.hidden_size = 64
        torch.manual_seed(42)

    def test_token_classifier_head_forward(self):
        head = TokenClassifierHead(hidden_size=self.hidden_size)
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        tok_logits = head(hidden_states)

        self.assertEqual(tok_logits.shape, (self.batch_size, self.seq_len))
        self.assertFalse(torch.isnan(tok_logits).any())

    def test_span_cnn_head_forward(self):
        head = SpanCNNHead(hidden_size=self.hidden_size, kernel_size=3)
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        span_logits = head(hidden_states)

        self.assertEqual(span_logits.shape, (self.batch_size, self.seq_len))
        self.assertFalse(torch.isnan(span_logits).any())

    def test_asymmetric_modulation_gate(self):
        gate = AsymmetricModulationGate(gamma=0.5)
        tok_logits = torch.tensor([[2.0, -1.0, 3.0]])
        span_logits = torch.tensor([[4.0, 2.0, -5.0]])

        # High span coherence (4.0) should inhibit eviction score (2.0 - 0.5*4 = 0.0 -> sig=0.5)
        # Negative span (-5.0) has ReLU=0, no inhibition (3.0 - 0 = 3.0 -> sig~0.952)
        probs = gate(tok_logits, span_logits)
        self.assertEqual(probs.shape, (1, 3))
        self.assertAlmostEqual(probs[0, 0].item(), 0.5, places=3)
        self.assertAlmostEqual(probs[0, 2].item(), torch.sigmoid(torch.tensor(3.0)).item(), places=3)

    def test_dyad_dual_head_pruner_module(self):
        pruner = DyadDualHeadPruner(hidden_size=self.hidden_size, gamma=0.5)
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        out = pruner(hidden_states)

        self.assertIn("keep_probs", out)
        self.assertIn("tok_logits", out)
        self.assertIn("span_logits", out)
        self.assertEqual(out["keep_probs"].shape, (self.batch_size, self.seq_len))
        self.assertTrue((out["keep_probs"] >= 0.0).all() and (out["keep_probs"] <= 1.0).all())

    def test_mechanism_b_regex_preservation(self):
        pruner = DyadContextPruner()
        
        # Test identifiers that MUST be preserved by regex
        test_strings = [
            ("0xbaab7dbf64751104133af04abc7d9979f0fda3b059a322a8333f533d3f32bf7f", True),
            ("--max-tokens", True),
            ("src/transformers/models/gpt2.py", True),
            ("config.json", True),
            ("SIGILL", True),
            ("SlottedStaticCache", True),
            ("432", True),
            ("the", False),
            ("is", False),
        ]

        for text, expected in test_strings:
            is_match = bool(MUST_KEEP_PATTERN.search(text))
            self.assertEqual(is_match, expected, f"Regex failed on token '{text}': got {is_match}, expected {expected}")

    def test_dyad_context_pruner_prune(self):
        pruner_model = DyadDualHeadPruner(hidden_size=self.hidden_size, gamma=0.5)
        pruner = DyadContextPruner(pruner_model=pruner_model, keep_threshold=0.5)

        input_ids = torch.randint(100, 1000, (1, 10))
        hidden_states = torch.randn(1, 10, self.hidden_size)

        # Mock decode function returning words
        vocab = {
            input_ids[0, 1].item(): "0x1234abcd",
            input_ids[0, 2].item(): "the",
            input_ids[0, 3].item(): "SlottedCache",
            input_ids[0, 4].item(): "random",
        }
        decode_fn = lambda ids: " ".join([vocab.get(i, "token") for i in ids])

        result = pruner.prune(input_ids, hidden_states=hidden_states, decode_fn=decode_fn)

        self.assertIn("keep_mask", result)
        self.assertIn("compression_ratio", result)
        self.assertTrue(result["keep_mask"][0, 0])  # BOS kept
        self.assertTrue(result["keep_mask"][0, -1])  # EOS kept
        self.assertTrue(result["keep_mask"][0, 1])  # Hex address preserved by Mechanism B
        self.assertTrue(result["keep_mask"][0, 3])  # CamelCase preserved by Mechanism B


if __name__ == "__main__":
    unittest.main()
