"""Common asset generation: shared schema for all scenarios.

Each scenario has its own generator (s1, s2, s3) but they all produce
the same asset dict schema:
    {
        "asset_id": int,
        "asset_type": str,
        "age_bracket": int (0=young, 1=old),
        "components": {name: count},
        "text": str,
        "true_condition": str,
        "observed_condition": str,
        "true_yield_factor": float in (0, 1),
    }
"""

from __future__ import annotations

import numpy as np

from .s1_generator import generate_s1_assets
from .s2_loader import generate_s2_assets
from .s3_loader import generate_s3_assets

_GENERATORS = {
    "s1_it_infrastructure": generate_s1_assets,
    "s2_aviation_mro": generate_s2_assets,
    "s3_consumer_electronics": generate_s3_assets,
}


def generate_assets(
    config: dict,
    rng: np.random.Generator,
    note_rng: np.random.Generator | None = None,
) -> list[dict]:
    """Dispatch to a scenario generator, optionally isolating note randomness."""
    name = config["name"]
    gen_fn = _GENERATORS.get(name)
    if gen_fn is None:
        raise ValueError(f"No generator for scenario '{name}'. "
                        f"Available: {list(_GENERATORS.keys())}")
    return gen_fn(config, rng, note_rng)
