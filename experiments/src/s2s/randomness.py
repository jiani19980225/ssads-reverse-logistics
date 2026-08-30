"""Named random-number streams and seed-spec parsing for reproducible runs."""

from __future__ import annotations

import re

import numpy as np

STREAM_OFFSETS = {
    "generation": 1,
    "extraction": 2,
    "allocation": 3,
    "inspection": 4,
    "outcome": 5,
    "note_sensitivity": 6,
}

_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")


def rng_for(seed: int, stream: str) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if stream not in STREAM_OFFSETS:
        raise ValueError(f"Unknown random-number stream: {stream}")
    return np.random.default_rng(seed * 1000 + STREAM_OFFSETS[stream])


def parse_seed_spec(spec: str) -> list[int]:
    """Parse an inclusive range such as ``0-29`` or a comma-separated list."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Seed specification must be non-empty")
    spec = spec.strip()
    match = _RANGE_RE.fullmatch(spec)
    if match:
        lo, hi = (int(value) for value in match.groups())
        if lo > hi:
            raise ValueError("Seed range start cannot exceed its end")
        return list(range(lo, hi + 1))
    try:
        seeds = [int(value.strip()) for value in spec.split(",")]
    except ValueError as exc:
        raise ValueError(f"Invalid seed specification: {spec!r}") from exc
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be nonnegative")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed specification contains duplicates")
    return seeds
