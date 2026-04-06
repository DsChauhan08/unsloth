import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest


def _load_kaggle_forge_module():
    module_path = Path(__file__).resolve().parents[1] / "unsloth/models/kaggle_forge.py"
    spec = importlib.util.spec_from_file_location("kaggle_forge", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


kaggle_forge = _load_kaggle_forge_module()


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


def test_kaggle_timebomb_hours_parsing():
    with patch.dict(os.environ, {"UNSLOTH_KAGGLE_TIMEBOMB": "2.5"}, clear=False):
        assert kaggle_forge.kaggle_timebomb_hours() == pytest.approx(2.5)
    with patch.dict(os.environ, {"UNSLOTH_KAGGLE_TIMEBOMB": "0"}, clear=False):
        assert kaggle_forge.kaggle_timebomb_hours() is None
    with patch.dict(
        os.environ, {"UNSLOTH_KAGGLE_TIMEBOMB": "not-a-number"}, clear=False
    ):
        assert kaggle_forge.kaggle_timebomb_hours() is None


def test_kaggle_auto_flags_and_cache_dir_defaults():
    with patch.dict(
        os.environ,
        {
            "UNSLOTH_AUTO_VRAM": "1",
            "UNSLOTH_GHOST_CACHE": "1",
        },
        clear=False,
    ):
        assert kaggle_forge.kaggle_auto_vram_enabled()
        assert kaggle_forge.kaggle_ghost_cache_enabled()
        assert kaggle_forge.kaggle_ghost_cache_dir() == "/kaggle/working/.ghost_cache"

    with patch.dict(
        os.environ,
        {
            "UNSLOTH_GHOST_CACHE_DIR": "/tmp/custom-ghost",
        },
        clear=False,
    ):
        assert kaggle_forge.kaggle_ghost_cache_dir() == "/tmp/custom-ghost"
