from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def load_reward_config() -> dict[str, Any]:
    config_name = "supplymind_v2_rewards.yaml"
    env_path = os.environ.get("SUPPLYMIND_REWARD_CONFIG")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))

    package_path = Path(__file__).resolve()
    candidates.extend(
        [
            package_path.parents[2] / "configs" / config_name,
            package_path.parents[3] / "configs" / config_name,
            Path.cwd() / "configs" / config_name,
        ]
    )

    for path in candidates:
        if path.exists():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"Reward config is not a mapping: {path}")
            return loaded

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {config_name}; searched: {searched}")


def price_band(sku: str) -> dict[str, float]:
    return load_reward_config()["price_bands"][sku]


def cfg(section: str, key: str) -> float:
    return float(load_reward_config()[section][key])
