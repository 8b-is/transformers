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

import torch

from transformers.utils.pydantic_schemas import (
    UltraFastCausalLMOutput,
    UltraGenerationConfigSchema,
    UltraQuantizationConfigSchema,
)


class PydanticDataclassHybridTest(unittest.TestCase):
    def test_pydantic_generation_config_validation(self):
        """Verify Pydantic v2 validation for generation parameters."""
        schema = UltraGenerationConfigSchema(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_new_tokens=256,
            do_sample=True,
        )
        self.assertEqual(schema.temperature, 0.7)
        self.assertEqual(schema.max_new_tokens, 256)

        cfg_dict = schema.to_dict()
        self.assertIn("temperature", cfg_dict)
        self.assertIn("max_new_tokens", cfg_dict)

    def test_pydantic_quantization_config_validation(self):
        """Verify Pydantic v2 validation for quantization configs."""
        q_schema = UltraQuantizationConfigSchema(
            quant_method="metal",
            bits=4,
            group_size=64,
        )
        self.assertEqual(q_schema.quant_method, "metal")
        self.assertEqual(q_schema.bits, 4)

    def test_fast_slotted_causal_lm_output_pattern_matching(self):
        """Verify zero-overhead slotted dataclass and match/case structural extraction."""
        logits = torch.randn(1, 4, 32000)
        loss = torch.tensor(1.234)

        output = UltraFastCausalLMOutput(loss=loss, logits=logits)
        self.assertTrue(hasattr(output, "__slots__"))
        self.assertFalse(hasattr(output, "__dict__"))  # Zero __dict__ overhead

        # Test Structural Pattern Matching
        match output:
            case UltraFastCausalLMOutput(loss=l, logits=_) if l is not None:
                matched = True
                loss_val = float(l)
            case _:
                matched = False
                loss_val = None

        self.assertTrue(matched)
        self.assertAlmostEqual(loss_val, 1.234, places=3)
