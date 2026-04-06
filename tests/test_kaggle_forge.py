import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


def _load_kaggle_forge_module():
    module_path = Path(__file__).resolve().parents[1] / "unsloth/models/kaggle_forge.py"
    spec = importlib.util.spec_from_file_location("kaggle_forge", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


kaggle_forge = _load_kaggle_forge_module()


def test_force_float32_without_override_for_gemma4_float16():
    with patch.dict(os.environ, {"UNSLOTH_FORCE_FP16": "0"}, clear=False):
        assert kaggle_forge.should_force_gemma4_float32(
            model_types_all="gemma4,",
            requested_dtype=torch.float16,
            supports_bfloat16=False,
        )
        assert not kaggle_forge.should_preserve_gemma4_fp16(
            model_types_all="gemma4,",
            requested_dtype=torch.float16,
            supports_bfloat16=False,
        )


def test_preserve_fp16_with_override_for_gemma4_float16():
    with patch.dict(os.environ, {"UNSLOTH_FORCE_FP16": "1"}, clear=False):
        assert kaggle_forge.should_preserve_gemma4_fp16(
            model_types_all="gemma4,",
            requested_dtype=torch.float16,
            supports_bfloat16=False,
        )
        assert not kaggle_forge.should_force_gemma4_float32(
            model_types_all="gemma4,",
            requested_dtype=torch.float16,
            supports_bfloat16=False,
        )


def test_gemma4_helpers_are_noop_for_non_gemma4_or_non_float16():
    with patch.dict(os.environ, {"UNSLOTH_FORCE_FP16": "1"}, clear=False):
        assert not kaggle_forge.should_preserve_gemma4_fp16(
            model_types_all="llama,",
            requested_dtype=torch.float16,
            supports_bfloat16=False,
        )
        assert not kaggle_forge.should_force_gemma4_float32(
            model_types_all="gemma4,",
            requested_dtype=torch.float32,
            supports_bfloat16=False,
        )


def test_build_kaggle_t4x2_device_map_even_split():
    device_map = kaggle_forge.build_kaggle_t4x2_device_map(num_hidden_layers=8)
    assert device_map["model.embed_tokens"] == "cuda:0"
    assert device_map["model.norm"] == "cuda:1"
    assert device_map["lm_head"] == "cuda:1"

    for idx in range(4):
        assert device_map[f"model.layers.{idx}"] == "cuda:0"
    for idx in range(4, 8):
        assert device_map[f"model.layers.{idx}"] == "cuda:1"


def test_build_kaggle_t4x2_device_map_odd_split():
    device_map = kaggle_forge.build_kaggle_t4x2_device_map(num_hidden_layers=5)
    for idx in range(2):
        assert device_map[f"model.layers.{idx}"] == "cuda:0"
    for idx in range(2, 5):
        assert device_map[f"model.layers.{idx}"] == "cuda:1"


def test_build_kaggle_t4x2_device_map_rejects_invalid_layer_count():
    with pytest.raises(ValueError):
        kaggle_forge.build_kaggle_t4x2_device_map(num_hidden_layers=0)
