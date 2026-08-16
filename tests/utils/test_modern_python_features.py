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

import concurrent.futures
import unittest

from transformers.integrations.mem8_wave import CognitiveBand, MEM8MemoryStore, MEM8Wave


class ModernPythonFeaturesTest(unittest.TestCase):
    def test_strenum_and_slots(self):
        """Test StrEnum and __slots__ for memory efficiency and speed."""
        self.assertEqual(CognitiveBand.GAMMA, "gamma")
        self.assertEqual(CognitiveBand.BETA, "beta")
        self.assertEqual(CognitiveBand.ALPHA, "alpha")
        self.assertEqual(CognitiveBand.THETA, "theta")

        wave = MEM8Wave(text="def compute_attention(q, k, v): return q @ k.T", kind="code")
        self.assertEqual(wave.band, CognitiveBand.BETA)
        self.assertTrue(hasattr(wave, "__slots__"))
        self.assertFalse(hasattr(wave, "__dict__"))  # Strict memory optimization via __slots__

    def test_pattern_matching_bands(self):
        """Test match/case classification across all cognitive bands."""
        w_math = MEM8Wave(text=r"\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}")
        self.assertEqual(w_math.band, CognitiveBand.GAMMA)

        w_code = MEM8Wave(text="class TransformerBlock(nn.Module): pass")
        self.assertEqual(w_code.band, CognitiveBand.BETA)

        w_logic = MEM8Wave(text="Because the premises hold, therefore the conclusion is sound.")
        self.assertEqual(w_logic.band, CognitiveBand.ALPHA)

        w_text = MEM8Wave(text="The quick brown fox jumps over the lazy dog.")
        self.assertEqual(w_text.band, CognitiveBand.THETA)

    def test_matmul_operator_dunder(self):
        """Test the @ matrix multiplication operator for wave interference."""
        w1 = MEM8Wave(text="def attention(q, k): pass")
        w2 = MEM8Wave(text="fn attention(q: Tensor, k: Tensor) -> Tensor")

        # Using @ operator
        score = w1 @ w2
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_free_threading_nogil_concurrency(self):
        """Test multi-threaded recording and recall under Python 3.13 free-threading."""
        store = MEM8MemoryStore()

        def worker(idx):
            store.record(f"def kernel_{idx}(x): return x * {idx}", kind="code")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(worker, range(50)))

        self.assertEqual(len(store), 50)
        self.assertEqual(len(list(store)), 50)

        recalled = store.recall("def kernel_eval(x):", top_k=5, threshold=0.05)
        self.assertLessEqual(len(recalled), 5)
        self.assertGreater(len(recalled), 0)

    def test_data_collator_pattern_matching(self):
        """Test DataCollatorMixin structural pattern matching on return_tensors."""
        from transformers.data.data_collator import DataCollatorMixin

        class MockCollator(DataCollatorMixin):
            return_tensors = "pt"

            def torch_call(self, features):
                return {"format": "torch", "count": len(features)}

            def numpy_call(self, features):
                return {"format": "numpy", "count": len(features)}

        collator = MockCollator()
        self.assertEqual(collator([1, 2, 3])["format"], "torch")
        self.assertEqual(collator([1, 2, 3], return_tensors="np")["format"], "numpy")
        with self.assertRaises(ValueError):
            collator([1, 2, 3], return_tensors="unsupported_fw")

    def test_strtobool_match_case(self):
        """Test modernized strtobool boolean parser with match/case."""
        from transformers.utils.generic import strtobool

        self.assertEqual(strtobool(True), 1)
        self.assertEqual(strtobool(False), 0)
        self.assertEqual(strtobool("true"), 1)
        self.assertEqual(strtobool("yes"), 1)
        self.assertEqual(strtobool("1"), 1)
        self.assertEqual(strtobool("on"), 1)
        self.assertEqual(strtobool("false"), 0)
        self.assertEqual(strtobool("no"), 0)
        self.assertEqual(strtobool("0"), 0)
        self.assertEqual(strtobool("off"), 0)
        with self.assertRaises(ValueError):
            strtobool("invalid_bool_string")
