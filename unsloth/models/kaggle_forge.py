"""[KAGGLE FORGE] Opt-in helpers for constrained Kaggle environments."""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional


def _is_env_flag_enabled(name: str, env: Optional[Mapping[str, str]] = None) -> bool:
    """Return True if an environment flag is set to 1."""
    env_map = os.environ if env is None else env
    return env_map.get(name, "0") == "1"


def _env_str(name: str, env: Optional[Mapping[str, str]] = None) -> str:
    env_map = os.environ if env is None else env
    return env_map.get(name, "").strip()


def _env_float(name: str, env: Optional[Mapping[str, str]] = None) -> Optional[float]:
    value = _env_str(name, env=env)
    if value == "":
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def kaggle_timebomb_hours(env: Optional[Mapping[str, str]] = None) -> Optional[float]:
    """[KAGGLE FORGE] Return graceful-stop wallclock limit in hours.

    Enabled via: UNSLOTH_KAGGLE_TIMEBOMB=<hours>
    """
    return _env_float("UNSLOTH_KAGGLE_TIMEBOMB", env=env)


def kaggle_auto_vram_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """[KAGGLE FORGE] Return True when OOM fallback sweeper is enabled."""
    return _is_env_flag_enabled("UNSLOTH_AUTO_VRAM", env=env)


def kaggle_ghost_cache_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """[KAGGLE FORGE] Return True when dataset ghost cache is enabled."""
    return _is_env_flag_enabled("UNSLOTH_GHOST_CACHE", env=env)


def kaggle_ghost_cache_dir(env: Optional[Mapping[str, str]] = None) -> str:
    """[KAGGLE FORGE] Resolve ghost cache root directory."""
    custom_dir = _env_str("UNSLOTH_GHOST_CACHE_DIR", env=env)
    if custom_dir:
        return custom_dir
    return "/kaggle/working/.ghost_cache"


def build_kaggle_t4x2_device_map(
    num_hidden_layers: int,
    layer_prefix: str = "model.layers",
) -> Dict[str, str]:
    """[KAGGLE FORGE] Build a deterministic 2-GPU layer split device map."""
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be a positive integer")

    midpoint = num_hidden_layers // 2
    device_map: Dict[str, str] = {
        "model.embed_tokens": "cuda:0",
        "model.norm": "cuda:1",
        "lm_head": "cuda:1",
    }
    for layer_idx in range(num_hidden_layers):
        device_map[f"{layer_prefix}.{layer_idx}"] = (
            "cuda:0" if layer_idx < midpoint else "cuda:1"
        )
    return device_map
