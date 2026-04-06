"""[KAGGLE FORGE] Runtime monkey patches for opt-in Kaggle behavior."""

from __future__ import annotations

import inspect
import logging
import os
from functools import wraps
from typing import Optional

import torch
from transformers import AutoConfig

from .kaggle_sync import attach_kaggle_cloud_sync_if_needed
from .models._utils import is_bfloat16_supported
from .models.kaggle_forge import (
    build_kaggle_t4x2_device_map,
    should_force_gemma4_float32,
    should_preserve_gemma4_fp16,
)

logger = logging.getLogger(__name__)


def _resolve_dtype(dtype_value):
    if isinstance(dtype_value, str) and hasattr(torch, dtype_value):
        return getattr(torch, dtype_value)
    return dtype_value


def _effective_requested_dtype(dtype_value):
    dtype_value = _resolve_dtype(dtype_value)
    if dtype_value is None:
        return torch.float16 if not is_bfloat16_supported() else torch.bfloat16
    return dtype_value


def _num_hidden_layers_from_config(model_config) -> Optional[int]:
    if model_config is None:
        return None
    candidates = [
        getattr(model_config, "num_hidden_layers", None),
        getattr(model_config, "n_layer", None),
    ]
    text_config = getattr(model_config, "text_config", None)
    if text_config is not None:
        candidates.extend(
            [
                getattr(text_config, "num_hidden_layers", None),
                getattr(text_config, "n_layer", None),
            ]
        )
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    return None


def _is_gemma4_model(model_name, token=None, revision=None, trust_remote_code=False):
    if isinstance(model_name, str):
        lower_name = model_name.lower()
        if "gemma-4" in lower_name or "gemma4" in lower_name:
            return True

    try:
        config = AutoConfig.from_pretrained(
            model_name,
            token=token,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        return getattr(config, "model_type", "") == "gemma4"
    except Exception:
        return False


def _apply_kaggle_router_if_needed(bound_arguments):
    if os.environ.get("UNSLOTH_KAGGLE_MULTI_GPU", "0") != "1":
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        logger.warning(
            "[KAGGLE FORGE] UNSLOTH_KAGGLE_MULTI_GPU=1 requested, but 2 CUDA GPUs were not detected."
        )
        return

    model_name = bound_arguments.get("model_name")
    token = bound_arguments.get("token")
    revision = bound_arguments.get("revision")
    trust_remote_code = bound_arguments.get("trust_remote_code", False)

    try:
        model_config = AutoConfig.from_pretrained(
            model_name,
            token=token,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        logger.warning(
            "[KAGGLE FORGE] Failed to inspect model config for hard routing: %s", exc
        )
        return

    num_hidden_layers = _num_hidden_layers_from_config(model_config)
    if num_hidden_layers is None:
        logger.warning(
            "[KAGGLE FORGE] Could not determine num_hidden_layers. Keeping original device_map."
        )
        return

    logger.info("[KAGGLE FORGE] Initiating T4x2 Hard-Routing...")
    bound_arguments["device_map"] = build_kaggle_t4x2_device_map(
        num_hidden_layers=num_hidden_layers,
    )


def _apply_gemma4_dtype_override_if_needed(bound_arguments):
    requested_dtype = _effective_requested_dtype(bound_arguments.get("dtype"))
    supports_bfloat16 = is_bfloat16_supported()

    if requested_dtype != torch.float16:
        return

    model_name = bound_arguments.get("model_name")
    token = bound_arguments.get("token")
    revision = bound_arguments.get("revision")
    trust_remote_code = bound_arguments.get("trust_remote_code", False)
    is_gemma4 = _is_gemma4_model(
        model_name=model_name,
        token=token,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if not is_gemma4:
        return

    model_types_all = "gemma4," if is_gemma4 else ""
    if should_preserve_gemma4_fp16(
        model_types_all=model_types_all,
        requested_dtype=requested_dtype,
        supports_bfloat16=supports_bfloat16,
    ):
        logger.warning(
            "[KAGGLE FORGE] ⚠️ UNSLOTH_FORCE_FP16 ACTIVE: Bypassing float32 upcast."
        )
        logger.warning(
            "[KAGGLE FORGE] Model will load in FP16. Risk of NaN gradients acknowledged."
        )
        bound_arguments["dtype"] = torch.float16
        return

    if should_force_gemma4_float32(
        model_types_all=model_types_all,
        requested_dtype=requested_dtype,
        supports_bfloat16=supports_bfloat16,
    ):
        logger.warning(
            "Using float16 precision for gemma4 won't work! Using float32. "
            "Set UNSLOTH_FORCE_FP16=1 to override."
        )
        bound_arguments["dtype"] = torch.float32


def _wrap_from_pretrained(original_function):
    signature = inspect.signature(original_function)

    @wraps(original_function)
    def wrapped(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        # [KAGGLE FORGE] Opt-in dtype override and hard router.
        _apply_gemma4_dtype_override_if_needed(bound.arguments)
        _apply_kaggle_router_if_needed(bound.arguments)

        return original_function(*bound.args, **bound.kwargs)

    wrapped._kaggle_forge_wrapped = True
    return wrapped


def _patch_loader_entrypoints():
    from .models.loader import FastLanguageModel, FastModel

    if not getattr(FastLanguageModel.from_pretrained, "_kaggle_forge_wrapped", False):
        FastLanguageModel.from_pretrained = staticmethod(
            _wrap_from_pretrained(FastLanguageModel.from_pretrained)
        )

    if not getattr(FastModel.from_pretrained, "_kaggle_forge_wrapped", False):
        FastModel.from_pretrained = staticmethod(
            _wrap_from_pretrained(FastModel.from_pretrained)
        )


def _patch_sft_trainer_cloud_sync():
    try:
        from trl import SFTTrainer
    except Exception:
        return

    if getattr(SFTTrainer, "_kaggle_forge_cloud_sync_wrapped", False):
        return

    original_init = SFTTrainer.__init__

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # [KAGGLE FORGE] Auto-register cloud sync callback when requested.
        attach_kaggle_cloud_sync_if_needed(self)

    SFTTrainer.__init__ = wrapped_init
    SFTTrainer._kaggle_forge_cloud_sync_wrapped = True


def apply_kaggle_forge_patches():
    """Apply all [KAGGLE FORGE] runtime patches once."""
    if globals().get("_KAGGLE_FORGE_PATCHED", False):
        return

    _patch_loader_entrypoints()
    _patch_sft_trainer_cloud_sync()

    globals()["_KAGGLE_FORGE_PATCHED"] = True
