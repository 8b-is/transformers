"""
MEM8 Wave Interference Memory Engine for Transformers
=====================================================
Reused from 8b-is/hf-mac (Swift 6) & entheai (Rust).
Wave-based associative recall without neural network re-evaluation.

Keep the past RAW; search the raw space; compress LAST.

Frequency Bands:
  - Math / Formal: Gamma band (500–800 Hz)
  - Code / Structure: Beta band (300–500 Hz)
  - Reasoning / Logic: Alpha band (100–300 Hz)
  - General Text: Theta band (0–100 Hz)
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from enum import StrEnum
from typing import Self


class CognitiveBand(StrEnum):
    """Cognitive frequency bands for wave memory resonance."""

    GAMMA = "gamma"  # 500–800 Hz: Math, formal proofs, symbolic equations
    BETA = "beta"  # 300–500 Hz: Code, syntax, structure, schemas
    ALPHA = "alpha"  # 100–300 Hz: Logic, reasoning, causal chains
    THETA = "theta"  # 0–100 Hz: General text, narrative, dialogue


class MEM8Wave:
    """A memory encoded as an oscillatory wave — frequency, amplitude, phase.

    Optimized with `__slots__` for zero-overhead memory footprint and fast attribute dispatch.
    """

    __slots__ = ("text", "kind", "timestamp", "frequency", "amplitude", "phase", "band")

    def __init__(self, text: str, kind: str = "user", timestamp: float | None = None):
        self.text = text
        self.kind = kind
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.band, self.frequency = self._compute_band_and_frequency(text)
        self.amplitude = max(0.5, min(1.0, math.log1p(len(text.split())) / 4.0))
        self.phase = (self.timestamp % 86400.0) / 86400.0 * (2.0 * math.pi)

    def _compute_band_and_frequency(self, text: str) -> tuple[CognitiveBand, float]:
        # Domain classification using match/case pattern matching
        lower = text.lower()

        is_code = any(w in lower for w in ["def ", "class ", "fn ", "struct ", "impl ", "import ", "const ", "let "])
        is_math = any(w in lower for w in ["\\frac", "\\int", "theorem", "lemma", "sigma", "delta", "alpha", "greek"])
        is_logic = any(w in lower for w in ["because", "therefore", "implies", "proof", "logic", "deduce", "axiom"])

        match (is_code, is_math, is_logic):
            case (_, True, _):
                band = CognitiveBand.GAMMA
                base = 650.0
            case (True, _, _):
                band = CognitiveBand.BETA
                base = 350.0
            case (_, _, True):
                band = CognitiveBand.ALPHA
                base = 200.0
            case _:
                band = CognitiveBand.THETA
                base = 50.0

        # Add deterministic jitter from sha256 hash
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        jitter = (int(h[:4], 16) % 100) - 50.0
        frequency = max(10.0, min(990.0, base + jitter))
        return band, frequency

    def interference(self, other: Self) -> float:
        """Calculate wave interference magnitude with another wave."""
        df = self.frequency - other.frequency
        freq_factor = math.exp(-(df * df) / (200.0 * 200.0))
        amp_product = self.amplitude * other.amplitude
        phase_align = math.cos(self.phase - other.phase)
        return amp_product * freq_factor * (phase_align * 0.5 + 0.5)

    def __matmul__(self, other: Self) -> float:
        """Overload matrix multiplication operator `@` for direct wave interference: `w1 @ w2`."""
        return self.interference(other)

    def __repr__(self) -> str:
        snippet = self.text[:30] + "..." if len(self.text) > 30 else self.text
        return f"<MEM8Wave band={self.band.value} f={self.frequency:.1f}Hz A={self.amplitude:.2f} text={snippet!r}>"


class MEM8MemoryStore:
    """Zero-overhead wave interference memory store for transformers context augmentation.

    Equipped with `__slots__` and thread-safe lock for Python 3.13 free-threaded (nogil) execution.
    """

    __slots__ = ("waves", "_lock")

    def __init__(self):
        self.waves: list[MEM8Wave] = []
        self._lock = threading.Lock()

    def record(self, text: str, kind: str = "user") -> MEM8Wave:
        wave = MEM8Wave(text=text, kind=kind)
        with self._lock:
            self.waves.append(wave)
        return wave

    def recall(self, query: str, top_k: int = 3, threshold: float = 0.15) -> list[tuple[MEM8Wave, float]]:
        with self._lock:
            if not self.waves:
                return []
            current_waves = list(self.waves)

        q_wave = MEM8Wave(text=query, kind="query")
        scored = [(w, q_wave @ w) for w in current_waves]
        scored = [item for item in scored if item[1] >= threshold]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        with self._lock:
            return len(self.waves)

    def __getitem__(self, index: int) -> MEM8Wave:
        with self._lock:
            return self.waves[index]

    def __iter__(self):
        with self._lock:
            return iter(list(self.waves))

    def __repr__(self) -> str:
        return f"<MEM8MemoryStore entries={len(self)}>"
