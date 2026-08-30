"""Baseline definitions sharing the common pipeline."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from ..extractors.base import AbstractExtractor, ExtractionResult, NullExtractor
from ..metrics import RunMetrics
from ..pipeline import run_pipeline

_STRUCTURED_TRAIN_SEED = 99999
_STRUCTURED_TRAIN_N = 4000
_STRUCTURED_MODEL_CACHE: dict[str, tuple] = {}
_COMBINED_MODEL_CACHE: dict[str, tuple] = {}


class RandomExtractor(AbstractExtractor):
    """Random scores used only by the random-routing comparator."""

    def extract(self, text: str, rng: np.random.Generator | None,
                asset: dict | None = None) -> ExtractionResult:
        if rng is None:
            raise ValueError("RandomExtractor requires a random-number generator")
        return ExtractionResult(phi=rng.uniform(0.3, 1.0), sigma=rng.uniform(0.0, 1.0))


def _structured_schema(config: dict) -> tuple[list[str], list[str]]:
    asset_types = [entry["name"] for entry in config["asset_types"]]
    components = sorted({
        name
        for entry in config["asset_types"]
        for name in entry["components"]
    })
    return asset_types, components


def structured_features(asset: dict, schema: tuple[list[str], list[str]]) -> np.ndarray:
    """Encode age, asset type, and BOM composition without using text."""
    asset_types, components = schema
    if "age_bracket" not in asset or "asset_type" not in asset or "components" not in asset:
        raise ValueError("Structured features require age_bracket, asset_type, and components")
    return np.array(
        [float(asset["age_bracket"])]
        + [float(asset["asset_type"] == name) for name in asset_types]
        + [float(asset["components"].get(name, 0)) for name in components],
        dtype=float,
    )


def _training_cache_key(config: dict, model_kind: str) -> str:
    relevant = {
        "name": config["name"],
        "asset_types": config["asset_types"],
        "note_distribution": config["note_distribution"],
        "condition_yield_ranges": config["condition_yield_ranges"],
        "realized_yield_concentration": config["realized_yield_concentration"],
        "note_noise": config.get("note_noise", {"p_omit": 0.15, "p_mislabel": 0.25}),
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return f"{model_kind}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _new_regressor() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=_STRUCTURED_TRAIN_SEED,
        loss="squared_error",
    )


def train_structured_model(config: dict) -> tuple[GradientBoostingRegressor, tuple]:
    """Fit a real gradient-boosting baseline on an independent population."""
    key = _training_cache_key(config, "structured")
    if key in _STRUCTURED_MODEL_CACHE:
        return _STRUCTURED_MODEL_CACHE[key]

    from ...data_generators.common import generate_assets

    train_config = copy.deepcopy(config)
    train_config["n_assets"] = _STRUCTURED_TRAIN_N
    assets = generate_assets(train_config, np.random.default_rng(_STRUCTURED_TRAIN_SEED))
    schema = _structured_schema(config)
    x_train = np.vstack([structured_features(asset, schema) for asset in assets])
    y_train = np.array([asset["true_yield_factor"] for asset in assets], dtype=float)
    model = _new_regressor()
    model.fit(x_train, y_train)
    _STRUCTURED_MODEL_CACHE[key] = (model, schema)
    return model, schema


class StructuredOnlyExtractor(AbstractExtractor):
    """Condition estimate from age, asset type, and BOM composition."""

    def __init__(self, trained: tuple):
        self.model, self.schema = trained

    def extract(self, text: str, rng: np.random.Generator | None,
                asset: dict | None = None) -> ExtractionResult:
        if asset is None:
            raise ValueError("StructuredOnlyExtractor requires asset attributes")
        phi = float(self.model.predict(structured_features(asset, self.schema)[None, :])[0])
        return ExtractionResult(phi=float(np.clip(phi, 0.01, 1.0)), sigma=0.5)


def train_combined_model(config: dict) -> tuple[GradientBoostingRegressor, tuple]:
    """Fit structured plus deterministic keyword features on an independent population."""
    key = _training_cache_key(config, "combined_keyword")
    if key in _COMBINED_MODEL_CACHE:
        return _COMBINED_MODEL_CACHE[key]

    from ...data_generators.common import generate_assets
    from ..extractors.keyword import KeywordExtractor

    train_config = copy.deepcopy(config)
    train_config["n_assets"] = _STRUCTURED_TRAIN_N
    assets = generate_assets(train_config, np.random.default_rng(_STRUCTURED_TRAIN_SEED))
    schema = _structured_schema(config)
    keyword = KeywordExtractor(config["name"].split("_")[0])
    rows = []
    for asset in assets:
        text_result = keyword.extract(asset["text"], None, asset=asset)
        rows.append(np.concatenate([
            structured_features(asset, schema),
            np.array([text_result.phi, text_result.sigma, float(text_result.used_fallback)]),
        ]))
    x_train = np.vstack(rows)
    y_train = np.array([asset["true_yield_factor"] for asset in assets], dtype=float)
    model = _new_regressor()
    model.fit(x_train, y_train)
    _COMBINED_MODEL_CACHE[key] = (model, schema)
    return model, schema


class CombinedExtractor(AbstractExtractor):
    """Condition estimate from structured attributes plus keyword text features."""

    def __init__(self, trained: tuple, scenario: str):
        from ..extractors.keyword import KeywordExtractor

        self.model, self.schema = trained
        self.keyword = KeywordExtractor(scenario)

    def extract(self, text: str, rng: np.random.Generator | None,
                asset: dict | None = None) -> ExtractionResult:
        if asset is None:
            raise ValueError("CombinedExtractor requires asset attributes")
        text_result = self.keyword.extract(text, rng, asset=asset)
        features = np.concatenate([
            structured_features(asset, self.schema),
            np.array([text_result.phi, text_result.sigma, float(text_result.used_fallback)]),
        ])
        phi = float(self.model.predict(features[None, :])[0])
        return ExtractionResult(
            phi=float(np.clip(phi, 0.01, 1.0)),
            sigma=text_result.sigma,
            used_fallback=text_result.used_fallback,
        )


def run_baseline(
    baseline: str,
    config: dict,
    keyword_extractor: AbstractExtractor,
    seed: int,
    capacity_fraction: float = 1.0,
) -> RunMetrics:
    """Run one named method through the shared operational timeline."""
    if baseline == "random":
        return run_pipeline(
            config,
            RandomExtractor(),
            seed,
            inspection_mode="none",
            capacity_fraction=capacity_fraction,
            allocator="random",
        )
    if baseline == "rule_based":
        return run_pipeline(
            config,
            NullExtractor(),
            seed,
            inspection_mode="none",
            capacity_fraction=capacity_fraction,
            allocator="fifo",
        )
    if baseline == "structured":
        return run_pipeline(
            config,
            StructuredOnlyExtractor(train_structured_model(config)),
            seed,
            inspection_mode="full",
            capacity_fraction=capacity_fraction,
            allocator="greedy",
        )
    if baseline == "combined":
        scenario = config["name"].split("_")[0]
        return run_pipeline(
            config,
            CombinedExtractor(train_combined_model(config), scenario),
            seed,
            inspection_mode="policy",
            capacity_fraction=capacity_fraction,
            allocator="greedy",
        )
    if baseline == "oracle":
        return run_pipeline(
            config,
            NullExtractor(),
            seed,
            inspection_mode="oracle",
            capacity_fraction=capacity_fraction,
            allocator="greedy",
        )
    if baseline == "no_signal":
        return run_pipeline(
            config,
            NullExtractor(),
            seed,
            inspection_mode="none",
            capacity_fraction=capacity_fraction,
            allocator="greedy",
        )
    if baseline == "semantic_only":
        return run_pipeline(
            config,
            keyword_extractor,
            seed,
            inspection_mode="policy",
            capacity_fraction=capacity_fraction,
            allocator="threshold",
        )
    if baseline == "ours":
        return run_pipeline(
            config,
            keyword_extractor,
            seed,
            inspection_mode="policy",
            capacity_fraction=capacity_fraction,
            allocator="greedy",
        )
    raise ValueError(f"Unknown baseline: {baseline}")
