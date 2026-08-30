"""Noisy note-observation model shared by S1/S2/S3.

The latent condition drives synthetic yield. A separately corrupted observation
of that same latent condition drives the note. This avoids direct label access at
runtime while preserving an intentionally simulated text-to-condition signal.
"""

from __future__ import annotations

import numpy as np

# Default corruption levels applied to every scenario unless overridden via
# config["note_noise"].
DEFAULT_NOTE_NOISE = {
    "p_omit": 0.15,      # observer omits detail -> vague/uninformative note
    "p_mislabel": 0.25,  # perceived severity off by one adjacent category
}


def observed_condition(
    true_condition: str,
    severity_order: list,
    rng: np.random.Generator,
    p_omit: float,
    p_mislabel: float,
    omit_label: str,
    fixed_draw_count: bool = False,
) -> str:
    """Return the condition the note describes: a noisy view of ground truth.

    Three realistic corruption modes:
      - omission : note becomes uninformative (`omit_label`), losing the signal
      - mislabel : perceived severity drifts to an adjacent category
      - faithful : note correctly reflects the true condition

    Mislabeling is implemented as an attempted one-step severity perturbation.
    At either endpoint, an outward perturbation is clipped and leaves the label
    unchanged. Thus ``p_mislabel`` is the perturbation-attempt probability, not
    the realized changed-label rate.

    ``fixed_draw_count`` always consumes the direction draw, preserving common
    random numbers when corruption probabilities are varied in sensitivity runs.
    """
    if not severity_order or len(set(severity_order)) != len(severity_order):
        raise ValueError("severity_order must contain unique labels")
    if not 0.0 <= p_omit <= 1.0 or not 0.0 <= p_mislabel <= 1.0:
        raise ValueError("noise probabilities must be in [0, 1]")
    if p_omit + p_mislabel > 1.0:
        raise ValueError("noise probabilities cannot sum above 1")
    roll = rng.random()
    direction_roll = rng.random() if fixed_draw_count else None
    if roll < p_omit:
        return omit_label
    if roll < p_omit + p_mislabel and true_condition in severity_order:
        idx = severity_order.index(true_condition)
        if direction_roll is None:
            direction_roll = rng.random()
        direction = -1 if direction_roll < 0.5 else 1
        observed_idx = int(np.clip(idx + direction, 0, len(severity_order) - 1))
        return severity_order[observed_idx]
    return true_condition


def resolve_noise(config: dict) -> tuple:
    """(p_omit, p_mislabel) from config['note_noise'] merged over defaults."""
    noise = {**DEFAULT_NOTE_NOISE, **config.get("note_noise", {})}
    p_omit = float(noise["p_omit"])
    p_mislabel = float(noise["p_mislabel"])
    if not 0.0 <= p_omit <= 1.0 or not 0.0 <= p_mislabel <= 1.0:
        raise ValueError("note_noise probabilities must be in [0, 1]")
    if p_omit + p_mislabel > 1.0:
        raise ValueError("note_noise probabilities cannot sum above 1")
    return p_omit, p_mislabel
