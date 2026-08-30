"""Keyword-based deterministic extractor for the synthetic benchmark.

Its vocabulary is deliberately matched to a subset of the simulation's stylized
phrasing. It is a reproducible reference, not evidence of performance on organic
notes. Unmatched text falls back to the unadjusted prior (phi=1.0).
"""

import numpy as np

from .base import AbstractExtractor, ExtractionResult

# Signal vocabularies per scenario
_SIGNALS = {
    "s1": {
        "negative": ["burn", "burnt", "water damage", "corrosion", "bent pin",
                     "crack", "thermal", "swollen", "leak", "smoke", "melted",
                     "oxidiz", "short circuit", "capacitor"],
        "positive": ["routine decommission", "no corrosion", "seated properly",
                     "clean", "functional", "passed diagnostics", "normal wear",
                     "no damage", "all slots populated"],
        "ambiguous": ["fan loud", "intermittent", "occasional", "minor scratch",
                      "dust", "noisy"],
    },
    "s2": {
        "negative": ["corroded", "cracked", "failed", "inoperative", "seized",
                     "delaminated", "fatigue", "leak", "worn beyond", "beyond repair"],
        "positive": ["serviceable", "repaired", "within limits", "no defect found",
                     "complies", "overhauled", "within tolerance"],
        "ambiguous": ["oil", "vibration", "noise", "discoloration", "trending"],
    },
    "s3": {
        "negative": ["dead", "won't turn on", "exploded", "swollen battery",
                     "screen shattered", "water damage", "smoke", "fire"],
        "positive": ["wrong item", "changed mind", "unopened", "like new",
                     "works fine", "cosmetic only", "never used"],
        "ambiguous": ["stopped working", "slow", "glitchy", "sometimes",
                      "not happy", "disappointed"],
    },
}


class KeywordExtractor(AbstractExtractor):
    """Deterministic keyword-and-pattern classifier."""

    def __init__(self, scenario: str, drop_families: tuple[str, ...] = ()):
        """Build the reader, optionally blanking whole signal families.

        ``drop_families`` supports the held-out-vocabulary audit: a blanked
        family emulates a condition family the dictionary was never written for,
        so its notes match nothing and fall back to the uninformative prior.
        """
        if scenario not in _SIGNALS:
            raise ValueError(f"Unknown scenario: {scenario}")
        unknown = sorted(set(drop_families) - set(_SIGNALS[scenario]))
        if unknown:
            raise ValueError(f"Unknown signal families: {unknown}")
        self.scenario = scenario
        self.drop_families = tuple(drop_families)
        self.signals = {
            name: ([] if name in drop_families else list(phrases))
            for name, phrases in _SIGNALS[scenario].items()
        }

    def extract(self, text: str, rng: np.random.Generator | None,
                asset: dict | None = None) -> ExtractionResult:
        text_lower = text.lower()
        # Negation-aware: "no corrosion" should not trigger "corrosion"
        used_fallback = False
        neg = [s for s in self.signals["negative"]
               if s in text_lower and f"no {s}" not in text_lower]
        pos = [s for s in self.signals["positive"] if s in text_lower]
        amb = [s for s in self.signals["ambiguous"] if s in text_lower]

        # Fixed values make repeated extraction of the same note deterministic.
        if neg and not pos:
            phi = np.clip(0.675 ** len(set(neg)), 0.01, 0.40)
            sigma = min(0.95, 0.85 + 0.025 * (len(set(neg)) - 1))
        elif pos and not neg:
            phi = 0.925
            sigma = min(0.95, 0.85 + 0.025 * len(set(pos)))
        elif neg and pos:
            phi = 0.50
            sigma = 0.45
        elif amb:
            phi = 0.60
            sigma = 0.35
        else:
            # No signals: fallback to the unadjusted prior.
            phi = 1.0
            sigma = 0.275
            used_fallback = True

        # Scenario-specific signal-quality penalties.
        if self.scenario == "s2" and not neg and not pos:
            sigma *= 0.6  # standardized codes reduce quality
        if self.scenario == "s3":
            sigma *= 0.5  # colloquial language has weak yield correlation

        return ExtractionResult(
            phi=float(np.clip(phi, 0.01, 1.0)),
            sigma=float(np.clip(sigma, 0.05, 0.99)),
            used_fallback=used_fallback,
        )
