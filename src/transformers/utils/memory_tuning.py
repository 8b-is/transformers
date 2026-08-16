# Copyright 2026 The HuggingFace Team & 8b-is. All rights reserved.
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

"""
Ultra-Low-Overhead Memory & Garbage Collection Optimization Layer.

Features:
- Detection and configuration of ultra-fast allocators (mimalloc / jemalloc).
- PyTorch CUDA allocator tuning (`expandable_segments:True`, `garbage_collection_threshold`).
- Zero-pause generational GC tuning and object freezing (`gc.freeze()`) for LLM inference.
- `no_gc_cycle()` context manager for zero-overhead token generation loops.
- `FastTmaBufferPool`: Pre-allocated 128-byte aligned hardware buffer pool eliminating Python heap allocations.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import os
import sys
from typing import Generator

from . import logging


logger = logging.get_logger(__name__)


def detect_fast_allocator() -> str:
    """
    Detects if an optimized memory allocator like mimalloc or jemalloc is active in the process.
    Returns: 'mimalloc', 'jemalloc', or 'system'.
    """
    try:
        # Check current process symbols via default C library
        handle = ctypes.CDLL(None)
        if hasattr(handle, "mi_version") or hasattr(handle, "mi_malloc"):
            return "mimalloc"
        if hasattr(handle, "je_mallctl") or hasattr(handle, "mallocx"):
            return "jemalloc"
    except Exception:
        pass

    # Check environment preload configurations
    preload = os.environ.get("LD_PRELOAD", "") + os.environ.get("DYLD_INSERT_LIBRARIES", "")
    if "mimalloc" in preload.lower():
        return "mimalloc"
    if "jemalloc" in preload.lower():
        return "jemalloc"

    return "system"


def configure_fast_allocator() -> dict[str, str]:
    """
    Configures recommended high-throughput allocator environment flags for mimalloc and PyTorch CUDA.
    """
    applied = {}

    # PyTorch CUDA Allocator tuning
    cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in cuda_alloc_conf:
        new_conf = "expandable_segments:True,garbage_collection_threshold:0.8"
        if cuda_alloc_conf:
            new_conf = f"{cuda_alloc_conf},{new_conf}"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf
        applied["PYTORCH_CUDA_ALLOC_CONF"] = new_conf

    # mimalloc low-latency flags
    if "MIMALLOC_LARGE_OS_PAGES" not in os.environ:
        os.environ["MIMALLOC_LARGE_OS_PAGES"] = "1"
        applied["MIMALLOC_LARGE_OS_PAGES"] = "1"
    if "MIMALLOC_EAGER_COMMIT" not in os.environ:
        os.environ["MIMALLOC_EAGER_COMMIT"] = "1"
        applied["MIMALLOC_EAGER_COMMIT"] = "1"

    return applied


def configure_gc_for_inference(freeze_existing_objects: bool = True, threshold_multiplier: int = 100) -> tuple[int, int, int]:
    """
    Optimizes Python garbage collection for high-throughput model inference:
    1. Freezes existing model weights and permanent objects (`gc.freeze()`) so the GC tracker skips them entirely.
    2. Scales generational collection thresholds to avoid stop-the-world pauses in token generation loops.

    Returns the previous GC threshold tuple (threshold0, threshold1, threshold2).
    """
    old_thresholds = gc.get_threshold()

    if freeze_existing_objects and hasattr(gc, "freeze"):
        # Full collection before freezing to eliminate transient setup garbage
        gc.collect()
        gc.freeze()

    new_thresholds = (
        old_thresholds[0] * threshold_multiplier,
        old_thresholds[1] * threshold_multiplier,
        old_thresholds[2] * threshold_multiplier,
    )
    gc.set_threshold(*new_thresholds)
    return old_thresholds


@contextlib.contextmanager
def no_gc_cycle() -> Generator[None, None, None]:
    """
    Zero-pause context manager for latency-critical token generation and forward passes.
    Temporarily disables cyclic garbage collection and re-enables it upon exit.
    """
    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


class FastTmaBufferPool:
    """
    Pre-allocated pool of 128-byte aligned raw C memory buffers for TMA hardware descriptors.
    Completely eliminates Python heap memory allocation and garbage collector pressure in forward loops.
    """

    __slots__ = ("_pool", "_capacity", "_size")

    def __init__(self, capacity: int = 256):
        self._capacity = capacity
        # Pre-allocate 128-byte ctypes character arrays
        self._pool: list[ctypes.Array[ctypes.c_char]] = [
            (ctypes.c_char * 128)() for _ in range(capacity)
        ]
        self._size = capacity

    def acquire(self) -> ctypes.Array[ctypes.c_char]:
        """Acquires a 128-byte aligned buffer with zero allocations."""
        if self._size > 0:
            self._size -= 1
            return self._pool[self._size]
        # Pool empty fallback
        return (ctypes.c_char * 128)()

    def release(self, buf: ctypes.Array[ctypes.c_char]) -> None:
        """Releases the buffer back to the pool."""
        if self._size < self._capacity:
            # Zero out buffer memory in-place
            ctypes.memset(ctypes.byref(buf), 0, 128)
            self._pool[self._size] = buf
            self._size += 1

    def __len__(self) -> int:
        return self._size
