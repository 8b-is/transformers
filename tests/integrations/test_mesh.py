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

    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.integrations.mesh import MeshRouterWrapper


class DummyExpertConfig(PretrainedConfig):
    model_type = "dummy"

    def __init__(self, vocab_size=100, hidden_size=16, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class DummyExpert(PreTrainedModel):
    config_class = DummyExpertConfig

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids=None, **kwargs):
        x = self.embedding(input_ids)
        logits = self.fc(x)

        class Output:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values

        return Output(logits=logits, past_key_values=None)


class MeshRouterWrapperTest(unittest.TestCase):
    def test_mesh_router_wrapper(self):
        if not is_torch_available():
            return

        config = DummyExpertConfig()
        config.n_experts = 2

        expert1 = DummyExpert(config)
        expert2 = DummyExpert(config)

        mesh = MeshRouterWrapper(config, [expert1, expert2])

        input_ids = torch.randint(0, config.vocab_size, (2, 5))

        outputs = mesh(input_ids=input_ids)

        self.assertEqual(outputs.logits.shape, (2, 5, config.vocab_size))

        # Test gate explicitly
        gate_probs = mesh._gate(input_ids)
        self.assertEqual(gate_probs.shape, (2, 5, 2))

        # Probabilities should sum to 1
        torch.testing.assert_close(gate_probs.sum(dim=-1), torch.ones_like(gate_probs.sum(dim=-1)))
