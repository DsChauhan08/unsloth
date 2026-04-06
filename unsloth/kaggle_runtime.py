"""[KAGGLE FORGE] Runtime helpers: timebomb, auto-VRAM and ghost cache."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import TrainerCallback

from .models.kaggle_forge import kaggle_timebomb_hours

logger = logging.getLogger(__name__)


def clear_vram(best_effort: bool = True) -> None:
    """Clear Python + CUDA caches to recover from OOMs."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            if not best_effort:
                raise


def is_cuda_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
            "hip out of memory",
            "allocate memory",
        )
    )


def compute_effective_batch_size(
    batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    batch_size = max(1, int(batch_size))
    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    return batch_size * gradient_accumulation_steps


def derive_vram_fallback_batch_params(
    requested_batch_size: int,
    requested_grad_accum: int,
) -> list[tuple[int, int]]:
    """Generate fallback (batch_size, grad_accum) pairs.

    Keeps effective batch size approximately constant by increasing grad_accum
    when per-device batch size is reduced.
    """
    target_effective = compute_effective_batch_size(
        requested_batch_size,
        requested_grad_accum,
    )
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int]] = []

    current_bs = max(1, int(requested_batch_size))
    while current_bs >= 1:
        grad = max(1, math.ceil(target_effective / current_bs))
        pair = (current_bs, grad)
        if pair not in seen:
            candidates.append(pair)
            seen.add(pair)
        if current_bs == 1:
            break
        current_bs = max(1, current_bs // 2)

    # Ensure strict last-resort fallback exists.
    if (1, target_effective) not in seen:
        candidates.append((1, max(1, target_effective)))

    return candidates


def stable_dataset_fingerprint(
    *,
    dataset_source: str,
    local_datasets: Optional[list[str]],
    local_eval_datasets: Optional[list[str]],
    subset: Optional[str],
    train_split: Optional[str],
    eval_split: Optional[str],
    format_type: str,
    custom_format_mapping: Optional[dict[str, Any]],
    dataset_slice_start: Optional[int],
    dataset_slice_end: Optional[int],
    model_name: Optional[str],
    is_vlm: bool,
    is_audio: bool,
    is_audio_vlm: bool,
) -> str:
    payload = {
        "dataset_source": dataset_source or "",
        "local_datasets": sorted(local_datasets or []),
        "local_eval_datasets": sorted(local_eval_datasets or []),
        "subset": subset,
        "train_split": train_split,
        "eval_split": eval_split,
        "format_type": format_type,
        "custom_format_mapping": custom_format_mapping or {},
        "dataset_slice_start": dataset_slice_start,
        "dataset_slice_end": dataset_slice_end,
        "model_name": model_name,
        "is_vlm": bool(is_vlm),
        "is_audio": bool(is_audio),
        "is_audio_vlm": bool(is_audio_vlm),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ghost_cache_paths(cache_root: str, fingerprint: str) -> tuple[Path, Path]:
    root = Path(cache_root)
    cache_dir = root / fingerprint
    metadata_path = cache_dir / "meta.json"
    return cache_dir, metadata_path


class UnslothKaggleTimebombCallback(TrainerCallback):
    """Gracefully stop training after configured wallclock hours."""

    def __init__(
        self,
        *,
        hours: Optional[float] = None,
        start_time: Optional[float] = None,
    ):
        self.hours = hours if hours is not None else kaggle_timebomb_hours()
        self.start_time = time.time() if start_time is None else start_time
        self._fired = False

    @property
    def enabled(self) -> bool:
        return self.hours is not None and self.hours > 0

    def _should_fire(self) -> bool:
        if not self.enabled or self._fired:
            return False
        elapsed_seconds = max(0.0, time.time() - self.start_time)
        return elapsed_seconds >= float(self.hours) * 3600.0

    def _fire(self, control, where: str):
        self._fired = True
        logger.warning(
            "[KAGGLE FORGE] Timebomb reached %.3f hour limit during %s. "
            "Requesting graceful stop + save.",
            self.hours,
            where,
        )
        control.should_training_stop = True
        control.should_save = True
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._should_fire():
            return self._fire(control, "step_end")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._should_fire():
            return self._fire(control, "log")
        return control
