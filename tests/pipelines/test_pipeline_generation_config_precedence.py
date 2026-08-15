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
from unittest.mock import MagicMock

from transformers.generation.configuration_utils import GenerationConfig
from transformers.pipelines.base import Pipeline


class MockPipeline(Pipeline):
    _default_generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
    )

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, {}, {}

    def preprocess(self, input_):
        return input_

    def _forward(self, model_inputs):
        return model_inputs

    def postprocess(self, model_outputs):
        return model_outputs


class PipelineGenerationConfigPrecedenceTest(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.can_generate.return_value = True
        self.mock_model.config.prefix = None
        self.mock_model.config.task_specific_params = None
        self.mock_model.device = "cpu"
        self.mock_model.hf_device_map = {"": "cpu"}

        # Implementation of _prepare_generation_config simulating GenerativePreTrainedModel
        def _mock_prepare(generation_config=None, **kwargs):
            if generation_config is None:
                generation_config = GenerationConfig()
            import copy
            cfg = copy.deepcopy(generation_config)
            unused_kwargs = cfg.update(**kwargs)
            return cfg, unused_kwargs

        self.mock_model._prepare_generation_config = _mock_prepare

    def test_model_generation_config_takes_precedence_over_pipeline_defaults(self):
        """Fixes Issue #47752: Model's explicit generation config values must override pipeline defaults."""
        self.mock_model.generation_config = GenerationConfig(
            max_new_tokens=500,
            do_sample=True,
            temperature=0.2,
        )

        pipe = MockPipeline(
            model=self.mock_model,
            tokenizer=MagicMock(),
            task="text-generation",
        )

        # 1. Model's explicit max_new_tokens=500 is preserved (not overwritten by pipeline's default 256)
        self.assertEqual(pipe.generation_config.max_new_tokens, 500)
        # 2. Model's explicit temperature=0.2 is preserved (not overwritten by pipeline's default 0.7)
        self.assertEqual(pipe.generation_config.temperature, 0.2)
        # 3. Pipeline default for unset property (do_sample=True) is applied as fallback
        self.assertTrue(pipe.generation_config.do_sample)

    def test_pipeline_kwargs_take_highest_precedence(self):
        """kwargs passed directly to pipeline() override both model and pipeline defaults."""
        self.mock_model.generation_config = GenerationConfig(
            max_new_tokens=500,
            do_sample=True,
            temperature=0.2,
        )

        pipe = MockPipeline(
            model=self.mock_model,
            tokenizer=MagicMock(),
            task="text-generation",
            max_new_tokens=100,
        )

        self.assertEqual(pipe.generation_config.max_new_tokens, 100)
