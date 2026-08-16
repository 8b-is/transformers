import unittest

import numpy as np
import torch

from transformers.integrations.bitnet_mlx import (
    BitNetTernaryLinear,
    quantize_ternary_numpy,
    quantize_ternary_torch,
    unpack_ternary_numpy,
    unpack_ternary_torch,
)
from transformers.integrations.mem8_wave import MEM8MemoryStore


class SovereignIntegrationsTest(unittest.TestCase):
    def test_bitnet_mlx_quantize_unpack_roundtrip(self):
        w = np.array(
            [
                [-0.8, -0.4, 0.0, 0.4, 0.9, -0.1, 0.0, 0.2, -0.9, 0.8, 0.0, -0.3, 0.5, -0.5, 0.1, -0.2],
                [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, 0.0, 0.0, -0.1, -0.2, 0.3, 0.4],
            ],
            dtype=np.float32,
        )

        packed, scales = quantize_ternary_numpy(w)
        self.assertEqual(packed.shape, (2, 1))
        self.assertEqual(scales.shape, (2, 1))

        unpacked = unpack_ternary_numpy(packed, 16)
        self.assertEqual(unpacked.shape, (2, 16))
        self.assertTrue(np.all(np.isin(unpacked, [-1, 0, 1])))

    def test_bitnet_torch_quantize_unpack_roundtrip(self):
        w = torch.tensor(
            [
                [-0.8, -0.4, 0.0, 0.4, 0.9, -0.1, 0.0, 0.2, -0.9, 0.8, 0.0, -0.3, 0.5, -0.5, 0.1, -0.2],
                [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, 0.0, 0.0, -0.1, -0.2, 0.3, 0.4],
            ],
            dtype=torch.float32,
        )

        packed, scales = quantize_ternary_torch(w)
        self.assertEqual(packed.shape, (2, 1))
        self.assertEqual(scales.shape, (2, 1))

        unpacked = unpack_ternary_torch(packed, 16)
        self.assertEqual(unpacked.shape, (2, 16))
        self.assertTrue(torch.all(torch.isin(unpacked, torch.tensor([-1.0, 0.0, 1.0]))))

    def test_bitnet_ternary_linear_forward(self):
        layer = BitNetTernaryLinear(in_features=32, out_features=16)
        w_float = torch.randn(16, 32)
        layer.pack_from_float(w_float)

        x = torch.randn(2, 4, 32)
        out = layer(x)
        self.assertEqual(out.shape, (2, 4, 16))
        self.assertFalse(torch.isnan(out).any())

    def test_mem8_wave_interference_recall(self):
        store = MEM8MemoryStore()
        store.record(
            "def calculate_fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)", kind="assistant"
        )
        store.record("Theorem: The Riemann Zeta function has zeros on the critical line", kind="assistant")
        store.record("Today is a sunny morning in Budapest", kind="user")

        # Query for code should match fibonacci wave constructively
        results = store.recall("fn compute_primes(limit: usize) -> Vec<usize>", top_k=1)
        self.assertGreater(len(results), 0)
        top_match, score = results[0]
        self.assertIn("fibonacci", top_match.text)
        self.assertGreater(score, 0.15)


if __name__ == "__main__":
    unittest.main()
