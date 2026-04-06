"""[KAGGLE FORGE] Opt-in helpers for constrained Kaggle environments."""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional

import torch


def _is_env_flag_enabled(name: str, env: Optional[Mapping[str, str]] = None) -> bool:
    """Return True if an environment flag is set to 1."""
    env_map = os.environ if env is None else env
    return env_map.get(name, "0") == "1"


def should_force_gemma4_float32(
    model_types_all: str,
    requested_dtype: torch.dtype,
    supports_bfloat16: bool,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """[KAGGLE FORGE] Decide if Gemma4 should keep legacy float32 fallback."""
    if "gemma4" not in model_types_all:
        return False
    if requested_dtype != torch.float16:
        return False
    if supports_bfloat16:
        return False
    return not _is_env_flag_enabled("UNSLOTH_FORCE_FP16", env=env)


def should_preserve_gemma4_fp16(
    model_types_all: str,
    requested_dtype: torch.dtype,
    supports_bfloat16: bool,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """[KAGGLE FORGE] Decide if Gemma4 should remain in float16 on T4-class GPUs."""
    if "gemma4" not in model_types_all:
        return False
    if requested_dtype != torch.float16:
        return False
    if supports_bfloat16:
        return False
    return _is_env_flag_enabled("UNSLOTH_FORCE_FP16", env=env)


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
