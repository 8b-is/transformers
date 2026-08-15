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

import math
import hashlib
import time
from typing import List, Optional, Tuple, Dict, Any


class MEM8Wave:
    """A memory encoded as an oscillatory wave — frequency, amplitude, phase."""
    def __init__(self, text: str, kind: str = "user", timestamp: Optional[float] = None):
        self.text = text
        self.kind = kind
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.frequency = self._compute_frequency(text)
        self.amplitude = max(0.5, min(1.0, math.log1p(len(text.split())) / 4.0))
        self.phase = (self.timestamp % 86400.0) / 86400.0 * (2.0 * math.pi)

    def _compute_frequency(self, text: str) -> float:
        # Domain keyword classification
        lower = text.lower()
        if any(w in lower for w in ["def ", "class ", "fn ", "struct ", "impl ", "import ", "const ", "let "]):
            base = 350.0 # Beta band (Code)
        elif any(w in lower for w in ["\\frac", "\\int", "theorem", "lemma", "sigma", "delta", "alpha", "greek"]):
            base = 650.0 # Gamma band (Math)
        elif any(w in lower for w in ["because", "therefore", "implies", "proof", "logic", "deduce", "axiom"]):
            base = 200.0 # Alpha band (Reasoning)
        else:
            base = 50.0  # Theta band (General)
            
        # Add deterministic jitter from sha256 hash
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        jitter = (int(h[:4], 16) % 100) - 50.0
        return max(10.0, min(990.0, base + jitter))

    def interference(self, other: "MEM8Wave") -> float:
        """Calculate wave interference magnitude with another wave."""
        df = self.frequency - other.frequency
        freq_factor = math.exp(-(df * df) / (200.0 * 200.0))
        amp_product = self.amplitude * other.amplitude
        phase_align = math.cos(self.phase - other.phase)
        return amp_product * freq_factor * (phase_align * 0.5 + 0.5)


class MEM8MemoryStore:
    """Zero-overhead wave interference memory store for transformers context augmentation."""
    def __init__(self):
        self.waves: List[MEM8Wave] = []

    def record(self, text: str, kind: str = "user") -> MEM8Wave:
        wave = MEM8Wave(text=text, kind=kind)
        self.waves.append(wave)
        return wave

    def recall(self, query: str, top_k: int = 3, threshold: float = 0.15) -> List[Tuple[MEM8Wave, float]]:
        if not self.waves:
            return []
        q_wave = MEM8Wave(text=query, kind="query")
        scored = [(w, q_wave.interference(w)) for w in self.waves]
        scored = [item for item in scored if item[1] >= threshold]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
