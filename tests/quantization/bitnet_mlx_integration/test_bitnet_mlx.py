# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from transformers.utils import is_torch_available


if is_torch_available():
    import torch.nn as nn

    from transformers import BitNetMlxQuantConfig
    from transformers.integrations.bitnet_mlx import BitNetTernaryLinear, replace_with_bitnet_mlx_linear


class BitNetMlxTest(unittest.TestCase):
    def test_bitnet_ternary_linear_packing(self):
        if not is_torch_available():
            return

        in_features = 32
        out_features = 64

        # Initialize a float linear layer
        linear = nn.Linear(in_features, out_features, bias=False)
        # Create weights that are purely -1, 0, 1
        with torch.no_grad():
            linear.weight.data = torch.randint(-1, 2, (out_features, in_features)).float()

        # Instantiate BitNetTernaryLinear
        bitnet_linear = BitNetTernaryLinear(in_features, out_features, bias=False)

        # Load state dict directly (the hook should pack it)
        bitnet_linear.load_state_dict(linear.state_dict())

        # Forward pass on both
        x = torch.randn(2, in_features)

        with torch.no_grad():
            # Calculate what the quantized float weight should be
            scale = linear.weight.data.abs().mean(dim=-1, keepdim=True).clamp_(min=1e-5)
            quantized_weight = torch.round(linear.weight.data / scale).clamp_(-1, 1) * scale
            out_float = torch.nn.functional.linear(x, quantized_weight)
            out_packed = bitnet_linear(x)

        # The outputs should be identical
        torch.testing.assert_close(out_float, out_packed)

    def test_replace_with_bitnet_mlx_linear(self):
        if not is_torch_available():
            return

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 16)
                self.fc2 = nn.Linear(16, 16)

        model = DummyModel()
        config = BitNetMlxQuantConfig(modules_to_not_convert=["fc2"])

        model = replace_with_bitnet_mlx_linear(model, quantization_config=config, modules_to_not_convert=["fc2"])

        self.assertTrue(isinstance(model.fc1, BitNetTernaryLinear))
        self.assertTrue(isinstance(model.fc2, nn.Linear))
