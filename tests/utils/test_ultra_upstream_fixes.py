# Copyright 2026 The HuggingFace Inc. team & 8b-is Sovereign Transformers.
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
import numpy as np
import pytest

from transformers.utils import is_torch_available
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.siglip2.configuration_siglip2 import Siglip2Config, Siglip2TextConfig
from transformers.models.bitnet.configuration_bitnet import BitNetConfig
from transformers.dependency_versions_table import deps

if is_torch_available():
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
    from transformers.models.esm.openfold_utils.loss import compute_tm


class UltraUpstreamFixesTest(unittest.TestCase):
    def test_tokenizers_version_constraint_allows_patch_release(self):
        """Fix #47429: Ensure tokenizers constraint allows 0.23.1+ patch release."""
        constraint = deps["tokenizers"]
        self.assertIn("<0.24.0", constraint)

    def test_siglip2_vocab_size_sync(self):
        """Fix #47612: Siglip2Config should synchronize vocab_size from text_config."""
        text_cfg = Siglip2TextConfig(vocab_size=256000, bos_token_id=49406, eos_token_id=49407)
        cfg = Siglip2Config(text_config=text_cfg)
        self.assertEqual(cfg.vocab_size, 256000)

    def test_bitnet_use_sub_norms_option(self):
        """Fix #47957: BitNetConfig supports use_sub_norms for weight-quant-only checkpoints."""
        cfg_full = BitNetConfig(use_sub_norms=True)
        self.assertTrue(cfg_full.use_sub_norms)

        cfg_weight_only = BitNetConfig(use_sub_norms=False)
        self.assertFalse(cfg_weight_only.use_sub_norms)

    def test_gemma4_auto_config_mapping(self):
        """Fix #47448: AutoConfig recognizes gemma4 and gemma4_unified model types."""
        self.assertIn("gemma4", CONFIG_MAPPING_NAMES)
        self.assertIn("gemma4_unified", CONFIG_MAPPING_NAMES)

    @pytest.mark.skipif(not is_torch_available(), reason="PyTorch required")
    def test_dynamic_cache_crop_zero_and_oversized_negative(self):
        """Fix #47433: DynamicCache.crop(0) or oversized negative crop must not retain stale tokens."""
        key = torch.arange(3, dtype=torch.float32).reshape(1, 1, 3, 1)

        # Test crop(0) with negative syntax or legacy positive
        cache = DynamicCache()
        cache.update(key, key.clone(), layer_idx=0)
        cache.crop(-5)  # oversized crop
        self.assertEqual(cache.get_seq_length(), 0)

        cache2 = DynamicCache()
        cache2.update(key, key.clone(), layer_idx=0)
        cache2.crop(-3)  # exact crop
        self.assertEqual(cache2.get_seq_length(), 0)

        cache3 = DynamicCache()
        cache3.update(key, key.clone(), layer_idx=0)
        cache3.crop(-1)  # partial crop
        self.assertEqual(cache3.get_seq_length(), 2)

    @pytest.mark.skipif(not is_torch_available(), reason="PyTorch required")
    def test_esmfold_compute_tm_argmax_robustness(self):
        """Fix #47470: compute_tm in ESMFold works cleanly without IndexError."""
        logits = torch.randn(1, 10, 10, 64, dtype=torch.float16)
        tm = compute_tm(logits)
        self.assertIsNotNone(tm)

    @pytest.mark.skipif(not is_torch_available(), reason="PyTorch required")
    def test_normalized_repetition_penalty_gauge_independence(self):
        """Fix #47595: Normalized repetition penalty eliminates gauge shift dependency."""
        input_ids = torch.tensor([[0, 1]])
        scores = torch.arange(10, dtype=torch.float).unsqueeze(0) / 10 - 0.5
        proc = RepetitionPenaltyLogitsProcessor(penalty=1.3, normalize=True)
        p1 = torch.softmax(proc(input_ids, scores), dim=-1)
        p2 = torch.softmax(proc(input_ids, scores + 100.0), dim=-1)
        self.assertTrue(torch.allclose(p1, p2, atol=1e-4))

    @pytest.mark.skipif(not is_torch_available(), reason="PyTorch required")
    def test_hubert_pos_conv_padding_mask(self):
        """Fix #47739: HubertPositionalConvEmbedding preserves zeros on padded frames with batch norm."""
        from transformers.models.hubert.configuration_hubert import HubertConfig
        from transformers.models.hubert.modeling_hubert import HubertPositionalConvEmbedding

        cfg = HubertConfig(
            hidden_size=16,
            num_conv_pos_embeddings=3,
            num_conv_pos_embedding_groups=1,
            feat_extract_activation="linear",
            conv_pos_batch_norm=True,
        )
        layer = HubertPositionalConvEmbedding(cfg).eval()
        inputs = torch.randn(2, 10, 16)
        padding_mask = torch.tensor([[False] * 10, [False] * 5 + [True] * 5])
        out = layer(inputs, padding_mask=padding_mask)
        # Verify padded region is zeroed out
        self.assertTrue(torch.all(out[1, 5:, :] == 0.0))

    @pytest.mark.skipif(not is_torch_available(), reason="PyTorch required")
    def test_whisper_encoder_dtype_alignment(self):
        """Fix #47805: WhisperEncoder aligns float32 input_features to conv1 float16 weights."""
        from transformers.models.whisper.configuration_whisper import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        cfg = WhisperConfig(d_model=64, encoder_layers=1, encoder_attention_heads=4, max_source_positions=100)
        encoder = WhisperEncoder(cfg).eval().to(dtype=torch.float16)
        input_features = torch.randn(1, cfg.num_mel_bins, 200, dtype=torch.float32)
        out = encoder(input_features)
        self.assertEqual(out.last_hidden_state.dtype, torch.float16)
