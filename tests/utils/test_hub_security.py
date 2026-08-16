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

import json
import os
import tempfile
import unittest

from transformers.utils.hub import get_checkpoint_shard_files


class HubSecurityTest(unittest.TestCase):
    def test_checkpoint_shard_files_path_traversal_blocked(self):
        """Fixes Issue #47176: Reject shard filenames containing path traversal or absolute paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = os.path.join(tmp_dir, "model.safetensors.index.json")

            # Malicious index pointing outside directory
            malicious_index = {
                "metadata": {"total_size": 1024},
                "weight_map": {
                    "weight_a": "../secret_file.safetensors",
                    "weight_b": "model-00001-of-00002.safetensors",
                },
            }
            with open(index_path, "w") as f:
                json.dump(malicious_index, f)

            with self.assertRaises(ValueError) as ctx:
                get_checkpoint_shard_files(tmp_dir, index_path)

            self.assertIn("Unsafe shard filename", str(ctx.exception))

    def test_checkpoint_shard_files_absolute_path_blocked(self):
        """Fixes Issue #47176: Reject absolute shard filenames in checkpoint index."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = os.path.join(tmp_dir, "model.safetensors.index.json")

            malicious_index = {
                "metadata": {"total_size": 1024},
                "weight_map": {
                    "weight_a": "/etc/passwd",
                },
            }
            with open(index_path, "w") as f:
                json.dump(malicious_index, f)

            with self.assertRaises(ValueError) as ctx:
                get_checkpoint_shard_files(tmp_dir, index_path)

            self.assertIn("Unsafe shard filename", str(ctx.exception))

    def test_checkpoint_shard_files_safe_allowed(self):
        """Valid local relative shard filenames are allowed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = os.path.join(tmp_dir, "model.safetensors.index.json")
            safe_index = {
                "metadata": {"total_size": 1024},
                "weight_map": {
                    "weight_a": "model-00001-of-00002.safetensors",
                    "weight_b": "model-00002-of-00002.safetensors",
                },
            }
            with open(index_path, "w") as f:
                json.dump(safe_index, f)

            shards, metadata = get_checkpoint_shard_files(tmp_dir, index_path)
            self.assertEqual(len(shards), 2)
            self.assertTrue(all(s.startswith(tmp_dir) for s in shards))
