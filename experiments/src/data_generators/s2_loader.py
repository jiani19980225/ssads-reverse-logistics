"""S2 Loader: Aviation MRO asset generation (SYNTHETIC data only).

All assets and notes generated here are synthetic. The condition distribution,
yield ranges, and narrative vocabulary are stylized assumptions inspired by
Service Difficulty Report terminology; no FAA record is used as simulation input.

The latent condition drives true yield. The SDR-style note is generated from a
separately corrupted observation of that condition (omission or an adjacent-severity
perturbation attempt), and the extractor sees only the resulting text at runtime.
"""

from __future__ import annotations

import numpy as np

from .noise import observed_condition, resolve_noise

_SDR_TEMPLATES = {
    "corroded": [
        "Corroded skin panel at STA {sta}. Fastener holes show pitting.",
        "Corrosion found on wing spar cap during C-check. Beyond blend limits.",
        "Floor beam corroded at lavatory area. Exfoliation type corrosion.",
    ],
    "cracked": [
        "Cracked window frame at station {sta}. Fatigue crack 2.5 inches.",
        "Crack found in engine mount during borescope. Exceeds serviceable limits.",
        "Pressure bulkhead crack detected during NDI. Requires structural repair.",
    ],
    "inoperative": [
        "Nav unit inoperative. No output on test bench. Suspected circuit board failure.",
        "Comm radio intermittent. Fails self-test on channel 3.",
    ],
    "failed": [
        "Turbine blade failed. Metal contamination in oil filter at {hours} hrs.",
        "Bearing seized during ground run. Engine removed for shop visit.",
    ],
    "worn": [
        "Brake assembly worn beyond limits at {hours} landings. Requires overhaul.",
        "Flight control cable worn. Strand breakage exceeds allowable per AMM.",
        "Seat track rollers worn. Binding noted during adjustment.",
    ],
    "serviceable": [
        "Component serviceable. Within limits per AMM. No defect found.",
        "Overhauled per SB-{sta}. All measurements within tolerance.",
        "Routine removal at {hours} flight hours. Complies with AD requirements.",
        "Repaired per SRM. Returned to serviceable condition.",
    ],
    # Uninformative notes used only for the "omission" noise mode (signal lost).
    # Deliberately free of any keyword-vocabulary signal.
    "uninformative": [
        "Removed per maintenance schedule. No further detail recorded.",
        "Component removed; disposition pending. Status not documented at STA {sta}.",
        "Logged removal at {hours} hours. Paperwork incomplete at time of entry.",
    ],
}

# Severity-ordered (best -> worst recovery) for mislabel noise.
_SEVERITY_ORDER = ["serviceable", "worn", "inoperative", "corroded", "cracked", "failed"]


def generate_s2_assets(
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
        age_bracket = int(rng.choice([0, 1], p=[0.5, 0.5]))

        # The latent condition sets the synthetic true-yield factor.
        condition = conditions[rng.choice(len(conditions), p=cond_probs)]
        yf_lo, yf_hi = yield_ranges[condition]
        true_yield = float(rng.uniform(yf_lo, yf_hi))

        # SDR note generated from a NOISY observation of the condition.
        observed = observed_condition(
            condition,
            _SEVERITY_ORDER,
            note_rng,
            p_omit,
            p_mislabel,
            omit_label="uninformative",
            fixed_draw_count=paired_noise,
        )
        templates = _SDR_TEMPLATES[observed]
        note = templates[note_rng.integers(0, len(templates))]
        note = note.format(sta=int(note_rng.integers(100, 999)),
                          hours=int(note_rng.integers(5000, 25000)))

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
