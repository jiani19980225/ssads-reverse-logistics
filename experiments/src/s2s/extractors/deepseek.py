"""DeepSeek extractor: the LLM implementation of the extractor interface.

This is the concrete, runnable backing for the paper's "extractor-agnostic"
claim: it satisfies the same AbstractExtractor contract as the keyword and
full-vocabulary (phrase-matcher) extractors, returns a context factor phi in
(0,1] and signal-quality score sigma in [0,1], and plugs into the pipeline with no changes
to the decision engine. DeepSeek exposes an OpenAI-compatible endpoint, so this
uses the `openai` package pointed at DeepSeek's base URL; no separate SDK.

OPTIONAL LIVE ACCESS:
  - Requires: pip install openai, and a DEEPSEEK_API_KEY environment variable.
  - A response cache (dict mapping note -> (phi, sigma)) gives reproducible,
    zero-API-cost re-runs. The committed caches cover all unique notes generated
    by evaluation seeds 0--29, including full end-to-end pipeline runs.
  - New notes require live access, a currently supported explicit model ID when
    the historical `deepseek-chat` identifier is unavailable, and are not
    bit-reproducible across model versions. The original API date, cost, latency,
    and parse-failure log were not retained; see outputs/llm_cache/PROVENANCE.md.

Usage:
    python scripts/run_calibration.py --seeds 0-29 --llm --llm-sample 150
"""

from __future__ import annotations

import json
import os

import numpy as np

from .base import AbstractExtractor, ExtractionResult

_DEFAULT_MODEL = "deepseek-chat"
_BASE_URL = "https://api.deepseek.com"

# Per-scenario recoverability framing. Declared BEFORE extraction (the same
# protocol the deterministic extractors follow): the model reads only the note
# text and maps the described condition to a yield context factor phi in (0,1].
_GUIDANCE = {
    "s1": (
        "You assess decommissioned IT hardware (servers, drives, memory, GPUs) "
        "from a technician's note. Map the described physical condition to a yield "
        "context factor phi in (0,1]: ~0.9-1.0 for clean/routine units with no damage; "
        "~0.5-0.7 for mixed signals (minor wear, intermittent faults); ~0.05-0.3 for "
        "clear damage (burn marks, water damage, corrosion, bent pins, swollen caps); "
        "~0.5 when the note is uninformative. phi must never exceed 1.0."
    ),
    "s2": (
        "You assess aviation components from an FAA Service Difficulty Report-style "
        "maintenance note. Map the described condition to a yield context factor phi "
        "in (0,1]: ~0.85 serviceable; ~0.55 worn; ~0.45 inoperative; ~0.35 corroded; "
        "~0.25 cracked; ~0.15 failed; ~0.5 when uninformative. A logged repair does not "
        "guarantee the root cause was resolved. phi must never exceed 1.0."
    ),
    "s3": (
        "You assess consumer-electronics returns from a customer description. Map the "
        "described condition to a yield context factor phi in (0,1]: ~0.9-0.99 functional/"
        "unopened returns; ~0.8 cosmetic-only damage; ~0.5 degraded (battery, charging, "
        "performance); ~0.2 dead; ~0.06 safety hazard (swollen battery, smoke); ~0.5 when "
        "uninformative. phi must never exceed 1.0."
    ),
}

# This prompt is frozen because it produced the committed score caches. Its use of
# "confidence" means confidence that the note is informative, i.e. signal quality;
# it is not predictive confidence or a calibrated probability.
_SYSTEM_TEMPLATE = (
    "{guidance}\n\n"
    "Also return sigma in [0,1], your confidence that the note contains enough "
    "information to assess condition: high (~0.85) for explicit, descriptive notes; "
    "low (~0.2) for vague, uninformative, or missing detail. Confidence and condition "
    "are independent: you can be highly confident a unit is damaged (low phi, high sigma). "
    "Respond only with the structured fields."
)


class DeepSeekExtractor(AbstractExtractor):
    """DeepSeek extractor via OpenAI-compatible API.

    Args:
        scenario:       "s1", "s2", or "s3".
        model:          DeepSeek model ID (default is the historical cache request
                        identifier deepseek-chat; override it for a current live API).
        client:         injectable OpenAI client for offline tests.
        response_cache: dict mapping note text -> (phi, sigma); populated
                        on every live call and persisted by the calibration
                        script so re-runs are free.
        allow_live:     whether a cache miss may call the live API. Reproduction
                        scripts disable this unless the user explicitly opts in.
    """

    def __init__(self, scenario: str, model: str = _DEFAULT_MODEL,
                 client=None, response_cache: dict | None = None, *,
                 allow_live: bool = True):
        if scenario not in _GUIDANCE:
            raise ValueError(f"Unknown scenario: {scenario}")
        if not isinstance(allow_live, bool):
            raise TypeError("allow_live must be a bool")
        self.scenario = scenario
        self.model = model
        self._client = client
        self._cache = response_cache if response_cache is not None else {}
        self._allow_live = allow_live
        self._system = _SYSTEM_TEMPLATE.format(guidance=_GUIDANCE[scenario])

    def _client_or_build(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "DeepSeekExtractor needs the openai package. "
                "Install: pip install openai"
            ) from e
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError(
                "DeepSeekExtractor needs DEEPSEEK_API_KEY in the environment."
            )
        self._client = OpenAI(api_key=key, base_url=_BASE_URL)
        return self._client

    @staticmethod
    def _result(phi, sigma) -> ExtractionResult:
        return ExtractionResult(phi=phi, sigma=sigma)

    def extract(self, text: str, rng: np.random.Generator | None,
                asset: dict | None = None) -> ExtractionResult:
        if text in self._cache:
            phi, sigma = self._cache[text]
            return self._result(phi, sigma)

        if not self._allow_live:
            raise RuntimeError(
                "DeepSeek cache miss in cache-only mode. Reproduce with the "
                "committed seeds/cache, or explicitly enable live API access."
            )

        client = self._client_or_build()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=0,          # low-variance request; the cache fixes reruns
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system + (
                    "\n\nReturn ONLY a JSON object with keys: "
                    "\"phi\" (number 0-1), \"sigma\" (number 0-1), "
                    "\"condition\" (string)."
                )},
                {"role": "user", "content": f"Note:\n{text}"},
            ],
        )
        try:
            data = json.loads(resp.choices[0].message.content)
            phi = data["phi"]
            sigma = data["sigma"]
            result = self._result(phi, sigma)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("DeepSeek response does not contain valid phi and sigma fields") from exc
        self._cache[text] = (phi, sigma)
        return result
