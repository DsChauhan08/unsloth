"""[KAGGLE FORGE] Optional asynchronous checkpoint sync callback."""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class UnslothCloudSyncCallback(TrainerCallback):
    """[KAGGLE FORGE] Push saved checkpoints to HF Hub asynchronously."""

    def __init__(
        self,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        repo_type: str = "model",
    ):
        self.repo_id = (
            repo_id or os.environ.get("UNSLOTH_CLOUD_SYNC_REPO", "")
        ).strip()
        self.token = token
        self.repo_type = repo_type
        self._api = HfApi(token=token)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="unsloth-kaggle-sync",
        )
        self._futures: list[Future] = []
        self._uploaded_checkpoints: set[str] = set()

    def _checkpoint_path(self, args, state) -> Optional[Path]:
        if not self.repo_id:
            return None
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_path.is_dir():
            return checkpoint_path
        if Path(args.output_dir).is_dir():
            return Path(args.output_dir)
        return None

    def _upload_checkpoint(self, checkpoint_path: Path) -> None:
        path_in_repo = checkpoint_path.name
        self._api.upload_folder(
            repo_id=self.repo_id,
            folder_path=str(checkpoint_path),
            path_in_repo=path_in_repo,
            repo_type=self.repo_type,
            commit_message=f"[KAGGLE FORGE] Sync {path_in_repo}",
        )

    def _log_future_result(self, future: Future, checkpoint_path: str) -> None:
        try:
            future.result()
            logger.info(
                "[KAGGLE FORGE] Cloud Sync finished for %s to %s.",
                checkpoint_path,
                self.repo_id,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning(
                "[KAGGLE FORGE] Cloud Sync failed for %s to %s: %s",
                checkpoint_path,
                self.repo_id,
                exc,
            )

    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = self._checkpoint_path(args, state)
        if checkpoint_path is None:
            return control

        checkpoint_key = str(checkpoint_path.resolve())
        if checkpoint_key in self._uploaded_checkpoints:
            return control

        logger.info(
            "[KAGGLE FORGE] Cloud Sync active. Syncing checkpoint to %s...",
            self.repo_id,
        )
        future = self._executor.submit(self._upload_checkpoint, checkpoint_path)
        future.add_done_callback(
            lambda fut, path=checkpoint_path.name: self._log_future_result(fut, path)
        )
        self._futures.append(future)
        self._uploaded_checkpoints.add(checkpoint_key)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        for future in self._futures:
            try:
                future.result()
            except Exception:  # pragma: no cover - already logged
                pass
        self._executor.shutdown(wait=False)
        return control


def cloud_sync_enabled() -> bool:
    """Return True when cloud checkpoint syncing is requested."""
    return os.environ.get("UNSLOTH_CLOUD_SYNC_REPO", "").strip() != ""


def attach_kaggle_cloud_sync_if_needed(trainer) -> None:
    """[KAGGLE FORGE] Attach async Hub sync callback to an SFT trainer."""
    if not cloud_sync_enabled():
        return
    if any(isinstance(cb, UnslothCloudSyncCallback) for cb in trainer.callbacks):
        return
    trainer.add_callback(UnslothCloudSyncCallback())
