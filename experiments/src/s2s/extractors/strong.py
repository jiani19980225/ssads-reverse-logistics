"""Full-vocabulary phrase-matching extractor: a signal-quality reference.

This is not an LLM or an oracle. It is a deterministic phrase matcher with full
template-vocabulary coverage for each scenario. At runtime it reads the same
noisy note as the KeywordExtractor and cannot access the latent condition.
It emits ONE discrete phi per recognized condition. The note generation noise
(p_omit=0.15, p_mislabel=0.25) keeps any note-reader below r=1.0.

It is a deterministic comparison point, not an oracle or an assumed ceiling.
"""

import numpy as np

from .base import AbstractExtractor, ExtractionResult

# Distinctive phrases per (scenario, condition). Chosen to avoid cross-condition
# collisions (e.g. "no corrosion" in a clean note must NOT match "corrosion on").
_PATTERNS = {
    "s1": {
        "clean": ["routine decommission", "seated properly", "no corrosion",
                  "standard lifecycle", "clean internals", "passed diagnostics",
                  "scheduled refresh", "normal wear", "end of lease", "no issues noted"],
        "mixed": ["fan loud", "minor dust", "intermittent error", "minor scratch",
                  "thermal throttling", "psu fan noisy", "cable wear", "oxidiz"],
        "damaged": ["psu failure", "burn mark", "smells burnt", "water damage",
                    "short circuit", "thermal damage", "swollen capacitor",
                    "bent pcie", "bent pin", "board flexion", "smoke damage",
                    "melted plastic", "do not power on"],
        "ambiguous": ["no notes from previous", "status unknown", "needs verification",
                      "asset tag mismatch", "condition not assessed"],
    },
    "s2": {
        "serviceable": ["component serviceable", "within limits", "no defect found",
                        "within tolerance", "complies with ad", "returned to serviceable"],
        "worn": ["worn beyond limits", "cable worn", "rollers worn", "requires overhaul"],
        "inoperative": ["inoperative", "fails self-test", "intermittent"],
        "corroded": ["corroded", "corrosion", "pitting", "exfoliation"],
        "cracked": ["cracked", "crack found", "fatigue crack", "bulkhead crack"],
        "failed": ["failed", "metal contamination", "seized"],
    },
    "s3": {
        "functional_return": ["wrong item", "changed mind", "never opened",
                              "unused", "sealed", "like new", "don't need it"],
        "cosmetic": ["cosmetic only", "small scratch", "minor dent", "dead pixel"],
        "degraded": ["battery drains", "slow performance", "charging port loose",
                     "speaker crackles", "factory reset"],
        "dead": ["won't turn on", "stopped working", "screen went black",
                 "sometimes it works", "dead."],
        "hazard": ["swollen battery", "bulging", "smoke came out", "very hot",
                   "burn mark"],
    },
}

# Recoverability map (declared before extraction): condition -> phi point estimate.
_PHI = {
    "s1": {"clean": 0.92, "mixed": 0.60, "damaged": 0.18, "ambiguous": 0.50},
    "s2": {"serviceable": 0.85, "worn": 0.55, "inoperative": 0.45,
           "corroded": 0.35, "cracked": 0.25, "failed": 0.15},
    "s3": {"functional_return": 0.95, "cosmetic": 0.82, "degraded": 0.52,
           "dead": 0.20, "hazard": 0.06},
}


class StrongExtractor(AbstractExtractor):
    """Full-vocabulary phrase matcher: classify the described condition, map to phi.
    Reads the same noisy condition-anchored note as KeywordExtractor; not an oracle."""

    def __init__(self, scenario: str):
        if scenario not in _PATTERNS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario = scenario
        self.patterns = _PATTERNS[scenario]
        self.phi_map = _PHI[scenario]

    def extract(self, text: str, rng: np.random.Generator | None,
                asset: dict | None = None) -> ExtractionResult:
        t = text.lower()
        # Score each condition by number of distinctive phrases present.
        scores = {cond: sum(p in t for p in phrases)
                  for cond, phrases in self.patterns.items()}
        best = max(scores, key=lambda condition: scores[condition])
        if scores[best] == 0:
            # Unmatched note: no condition signal recovered, so use the fallback prior.
            return ExtractionResult(phi=1.0, sigma=0.20, used_fallback=True)
        return ExtractionResult(phi=float(self.phi_map[best]), sigma=0.90)
