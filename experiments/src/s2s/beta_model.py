"""Ground-truth stochastic yield model used for realized-value evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

_EPS = 1e-3
ArrayLike: TypeAlias = float | np.ndarray


@dataclass(frozen=True)
class BetaParams:
    alpha: ArrayLike
    beta: ArrayLike

    @property
    def mean(self) -> ArrayLike:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> ArrayLike:
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))


def sample_yield(params: BetaParams, rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """Draw yields while preserving every parameter dimension.

    A single draw removes only the leading draw dimension. In particular, one
    component remains a one-element vector rather than collapsing to a scalar.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    alpha, beta = np.broadcast_arrays(
        np.asarray(params.alpha, dtype=float), np.asarray(params.beta, dtype=float)
    )
    if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(beta)):
        raise ValueError("Beta parameters must be finite")
    if np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError("Beta parameters must be positive")
    samples = rng.beta(alpha, beta, size=(n, *alpha.shape))
    return samples[0] if n == 1 else samples


def ground_truth_params(true_yield: ArrayLike, concentration: float = 20.0) -> BetaParams:
    """Create the evaluation distribution around a synthetic true yield."""
    if (
        isinstance(concentration, bool)
        or not isinstance(concentration, (int, float))
    ):
        raise TypeError("concentration must be numeric")
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration must be finite and positive")
    try:
        raw_yield = np.asarray(true_yield)
    except (TypeError, ValueError) as exc:
        raise TypeError("true_yield must be numeric") from exc
    if raw_yield.dtype.kind not in "iuf":
        raise TypeError("true_yield must be numeric")
    y = raw_yield.astype(float, copy=False)
    if not np.all(np.isfinite(y)):
        raise ValueError("true_yield must be finite")
    if np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("true_yield must be in [0, 1]")
    y = np.clip(y, _EPS, 1.0 - _EPS)
    return BetaParams(alpha=y * concentration, beta=(1.0 - y) * concentration)
