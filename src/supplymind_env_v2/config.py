from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def load_reward_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "configs" / "supplymind_v2_rewards.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def price_band(sku: str) -> dict[str, float]:
    return load_reward_config()["price_bands"][sku]


def cfg(section: str, key: str) -> float:
    return float(load_reward_config()[section][key])
