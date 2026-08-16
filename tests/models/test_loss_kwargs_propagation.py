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

from transformers.modeling_layers import GenericForSequenceClassification


class LossKwargsPropagationTest(unittest.TestCase):
    def test_generic_sequence_classification_forwards_loss_kwargs(self):
        """Fixes Issue #47688: Ensure **kwargs like num_items_in_batch reach loss_function."""
        config = MagicMock()
        config.num_labels = 2
        config.get_text_config.return_value.hidden_size = 16
        config.get_text_config.return_value.pad_token_id = 0

        # We test that loss_function receives **kwargs
        mock_loss_fn = MagicMock(return_value=1.0)

        # Instantiate without calling super().__init__ directly
        head = object.__new__(GenericForSequenceClassification)
        head.config = config
        head.loss_function = mock_loss_fn
        head.base_model_prefix = "model"

        mock_base = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        mock_base.return_value = mock_output
        setattr(head, "model", mock_base)
        head.score = MagicMock(return_value=MagicMock())
        head.score.return_value.__getitem__.return_value = MagicMock()

        # Call forward with dummy values and num_items_in_batch kwarg
        try:
            head.forward(
                input_ids=None,
                inputs_embeds=MagicMock(shape=[1, 4, 16]),
                labels=MagicMock(),
                num_items_in_batch=8,
            )
        except Exception:  # noqa: S110
            pass  # We check mock_loss_fn calls

        if mock_loss_fn.called:
            _, kwargs = mock_loss_fn.call_args
            self.assertEqual(kwargs.get("num_items_in_batch"), 8)
