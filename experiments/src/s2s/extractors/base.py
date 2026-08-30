"""Abstract extractor interface."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExtractionResult:
    phi: float            # context factor in (0, 1]
    sigma: float          # note-informativeness score in [0, 1]
    used_fallback: bool = False  # True iff no signal fired and phi was set to the
                                 # uninformative prior (1.0). Lets callers tell a
                                 # genuine high-condition read (phi -> 0.999) apart
                                 # from a no-signal fallback (both have phi ~ 1.0).

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("phi", self.phi, 0.0, 1.0),
            ("sigma", self.sigma, 0.0, 1.0),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if name == "phi" and not lower < float(value) <= upper:
                raise ValueError("phi must be in (0, 1]")
            if name == "sigma" and not lower <= float(value) <= upper:
                raise ValueError("sigma must be in [0, 1]")
        if not isinstance(self.used_fallback, bool):
            raise TypeError("used_fallback must be boolean")
        object.__setattr__(self, "phi", float(self.phi))
        object.__setattr__(self, "sigma", float(self.sigma))


class AbstractExtractor(ABC):
    @abstractmethod
    def extract(self, text: str, rng: np.random.Generator | None,
                asset: dict | None = None) -> ExtractionResult:
        """Extract phi and sigma from asset text.

        `asset` is the full asset dict, passed so extractors that use structured
        features (e.g. the structured gradient-boosting baseline) can access them.
        Text-only extractors ignore it.
        """
        ...


class NullExtractor(AbstractExtractor):
    """Returns phi=1.0, sigma=0.0 (no semantic information)."""

    def extract(self, text: str, rng: np.random.Generator | None,
                asset: dict | None = None) -> ExtractionResult:
        return ExtractionResult(phi=1.0, sigma=0.0, used_fallback=True)
