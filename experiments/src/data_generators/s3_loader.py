"""S3 Loader: Consumer Electronics asset generation (SYNTHETIC data only).

All assets and review texts generated here are synthetic. The condition
distribution, yield ranges, refurbishment values, and narrative vocabulary are
stylized benchmark assumptions; no customer record is used as simulation input.

The latent condition drives true yield. The review is generated from a separately
corrupted observation of that condition (omission or an adjacent-severity
perturbation attempt), and the extractor sees only the resulting text at runtime.
"""

from __future__ import annotations

import numpy as np

from .noise import observed_condition, resolve_noise

_TEMPLATES = {
    "functional_return": [
        "Wrong item shipped. Unopened box. Like new condition.",
        "Changed mind. Works fine, just don't need it anymore.",
        "Bought two by accident. This one never opened.",
        "Gift recipient already had one. Unused, sealed.",
    ],
    "cosmetic": [
        "Cosmetic only - small scratch on back. Fully functional.",
        "Minor dent on corner from shipping. Everything works.",
        "Screen has one dead pixel in corner. Otherwise perfect.",
    ],
    "degraded": [
        "Battery drains fast. Otherwise works. {age} months old.",
        "Slow performance after update. Might need factory reset.",
        "Charging port loose. Have to hold cable at angle.",
        "Speaker crackles at high volume. Rest is fine.",
    ],
    "dead": [
        "Dead. Won't turn on at all. Tried everything.",
        "It just stopped working after a week.",
        "Screen went black randomly. No response to any button.",
        "Sometimes it works sometimes it doesn't. Mostly doesn't now.",
    ],
    "hazard": [
        "Swollen battery. Phone is bulging. Scared to use it.",
        "Smoke came out when charging. Stopped using immediately.",
        "Got very hot while charging. Burn mark on table.",
    ],
    # Uninformative reviews used only for the "omission" noise mode (signal lost).
    # Deliberately free of any keyword-vocabulary signal.
    "uninformative": [
        "Returned item. No reason provided.",
        "Customer return processed. No description given.",
        "Item returned within the window. Condition not specified.",
    ],
}

# Severity-ordered (best -> worst recovery) for mislabel noise.
_SEVERITY_ORDER = ["functional_return", "cosmetic", "degraded", "dead", "hazard"]


def generate_s3_assets(
    config: dict,
    rng: np.random.Generator,
    note_rng: np.random.Generator | None = None,
) -> list[dict]:
    paired_noise = note_rng is not None
    note_rng = rng if note_rng is None else note_rng
    n = config["n_assets"]
    types = config["asset_types"]
    type_weights = np.array([t["weight"] for t in types])
    type_weights /= type_weights.sum()

    note_dist = config["note_distribution"]
    yield_ranges = config["condition_yield_ranges"]
    conditions = list(note_dist)
    cond_probs = np.array([note_dist[c] for c in conditions])
    cond_probs /= cond_probs.sum()

    p_omit, p_mislabel = resolve_noise(config)

    assets = []
    for i in range(n):
        atype = types[rng.choice(len(types), p=type_weights)]
        age_bracket = 0  # consumer electronics are typically young

        # Ground-truth condition -> true_yield (independent of the review text).
        condition = conditions[rng.choice(len(conditions), p=cond_probs)]
        yf_lo, yf_hi = yield_ranges[condition]
        true_yield = float(rng.uniform(yf_lo, yf_hi))

        # Review text generated from a NOISY observation of the condition.
        observed = observed_condition(
            condition,
            _SEVERITY_ORDER,
            note_rng,
            p_omit,
            p_mislabel,
            omit_label="uninformative",
            fixed_draw_count=paired_noise,
        )
        templates = _TEMPLATES[observed]
        note = templates[note_rng.integers(0, len(templates))]
        note = note.format(age=int(note_rng.integers(1, 24)))

        assets.append({
            "asset_id": i,
            "asset_type": atype["name"],
            "age_bracket": age_bracket,
            "components": dict(atype["components"]),
            "text": note,
            "true_condition": condition,
            "observed_condition": observed,
            "true_yield_factor": true_yield,
        })

    return assets
