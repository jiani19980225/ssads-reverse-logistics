"""S1 Generator: IT Infrastructure asset generation.

Generates 500 synthetic technician notes with controlled signal distribution.
The vocabulary and condition mix are stylized benchmark assumptions; they are
not estimated from an operational ITAD corpus.

The latent condition drives ``true_yield_factor``. The note is generated from a
separately corrupted observation of that condition (omission or an adjacent-severity
perturbation attempt). The extractor sees only this noisy text at runtime. See
``noise.observed_condition`` and ``run_calibration.py``.
"""

from __future__ import annotations

import numpy as np

from .noise import observed_condition, resolve_noise

_TEMPLATES = {
    "clean": [
        "Routine decommission. All components seated properly. No corrosion. {age}yr service.",
        "Standard lifecycle replacement. Clean internals, no damage. Passed diagnostics.",
        "Scheduled refresh. Normal wear only. All slots populated, no bent pins.",
        "End of lease return. Functional, clean. No issues noted during final check.",
    ],
    "mixed": [
        "Fan loud on startup but functional. Minor dust buildup. {age}yr old.",
        "One DIMM slot shows intermittent errors. Rest of system clean.",
        "Minor scratch on chassis. Occasional thermal throttling under load.",
        "PSU fan noisy. Otherwise functional. Some cable wear near power connector.",
        "Slight oxidization on rear I/O panel. All ports tested functional.",
    ],
    "damaged": [
        "PSU failure. Visible burn marks on mainboard near power connector J12. CPU smells burnt.",
        "Water damage near DIMM slots A1-A4. Corrosion on contacts. Short circuit suspected.",
        "Thermal damage to GPU. Swollen capacitors on VRM. Bent PCIe slot.",
        "Multiple bent pins on CPU socket. Board flexion damage from shipping.",
        "Smoke damage. Melted plastic near power supply. Do not power on.",
    ],
    "ambiguous": [
        "Decommissioned. No notes from previous tech.",
        "Pulled from rack B7. Status unknown. Needs verification.",
        "Asset tag mismatch. Physical condition not assessed.",
    ],
}

# Severity-ordered conditions used for "mislabel" noise (ambiguous is excluded
# because it carries no severity information). Omission maps to "ambiguous".
_SEVERITY_ORDER = ["clean", "mixed", "damaged"]


def generate_s1_assets(
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
    conditions = list(note_dist.keys())
    cond_probs = np.array([note_dist[c] for c in conditions])
    cond_probs /= cond_probs.sum()

    p_omit, p_mislabel = resolve_noise(config)

    assets = []
    for i in range(n):
        atype = types[rng.choice(len(types), p=type_weights)]
        age_bracket = int(rng.choice([0, 1], p=[0.6, 0.4]))
        age_years = int(
            note_rng.integers(0, 3) if age_bracket == 0 else note_rng.integers(3, 6)
        )

        # Latent condition -> synthetic true-yield factor.
        condition = conditions[rng.choice(len(conditions), p=cond_probs)]
        yf_lo, yf_hi = yield_ranges[condition]
        true_yield = float(rng.uniform(yf_lo, yf_hi))

        # The technician NOTE is generated from a NOISY observation of the
        # condition. The extractor reads only this corrupted text at runtime.
        observed = observed_condition(
            condition,
            _SEVERITY_ORDER,
            note_rng,
            p_omit,
            p_mislabel,
            omit_label="ambiguous",
            fixed_draw_count=paired_noise,
        )
        templates = _TEMPLATES[observed]
        note = templates[note_rng.integers(0, len(templates))]
        note = note.format(age=age_years)
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
