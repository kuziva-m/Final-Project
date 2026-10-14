"""
web/backup.py -- keep saved data in a private Hugging Face dataset.

A free Hugging Face Space loses its disk whenever it restarts (after
sleeping, or when a new version is uploaded). So the database -- accounts and
saved documents -- and the page photos of saved documents are copied to a
private dataset repo:

    records.db          a consistent snapshot of the SQLite database
    images/<name>.jpg   page photos and thumbnails of saved documents

On start-up the database is restored from the dataset; photos are fetched
when first viewed. Changes are pushed every LEDGER_BACKUP_SECONDS (default
60) and once more when the server shuts down.

Enabled only when a write token is available (HF_TOKEN) and the dataset is
known: LEDGER_BACKUP_REPO, or by default "<Space id>-data" when running in a
Space (Hugging Face sets SPACE_ID). Locally, nothing is backed up.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

DB_NAME = "records.db"
IMAGES = "images/"
# Squash the dataset's history now and then so old copies of the database
# don't pile up.
_SQUASH_EVERY = 100


class Backup:
    def __init__(self, repo_id: str | None, token: str | None, api=None, interval: float = 60.0):
        self.repo_id = repo_id
        self.token = token
        self.interval = interval
        self._api = api
        self.enabled = bool(repo_id and token)
        # Pushing stays off until the restore has worked, so a failed restore
        # can never overwrite a good backup with an empty database.
        self.can_push = False
        self.remote_images: set[str] = set()
        self._pending_images: dict[str, Path] = {}
        self._deleted_images: set[str] = set()
        self._dirty = False
        self._lock = threading.Lock()
        self._push_lock = threading.Lock()
        self._pushes = 0
        self._db_path: Path | None = None
        self._thread: threading.Thread | None = None

    @classmethod
    def from_environment(cls) -> "Backup":
        repo_id = os.environ.get("LEDGER_BACKUP_REPO")
        if not repo_id and os.environ.get("SPACE_ID"):
            repo_id = os.environ["SPACE_ID"] + "-data"
        try:
            interval = max(10.0, float(os.environ.get("LEDGER_BACKUP_SECONDS", "60")))
        except ValueError:
            interval = 60.0
        return cls(repo_id, os.environ.get("HF_TOKEN"), interval=interval)

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token)
        return self._api

    # --- start-up ---------------------------------------------------------------

    def restore(self, db_path: Path) -> None:
        """Bring back the saved database, if there is one; then allow pushes."""
        self._db_path = db_path
        if not self.enabled:
            return
        for attempt in range(3):
            try:
                try:
                    self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
                except Exception as exc:  # e.g. a token scoped to an existing dataset only
                    log.info("Backup: could not create %s (%s); assuming it exists.", self.repo_id, exc)
                files = self.api.list_repo_files(self.repo_id, repo_type="dataset")
                if DB_NAME in files:
                    with tempfile.TemporaryDirectory() as tmp:
                        local = self.api.hf_hub_download(self.repo_id, DB_NAME, repo_type="dataset", local_dir=tmp)
                        db_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(local, db_path)
                    log.info("Backup: restored the database from %s.", self.repo_id)
                self.remote_images = {f[len(IMAGES):] for f in files if f.startswith(IMAGES)}
                self.can_push = True
                return
            except Exception as exc:
                log.warning("Backup: restore attempt %d from %s failed: %s", attempt + 1, self.repo_id, exc)
                time.sleep(2 * (attempt + 1))
        log.error("Backup: could not restore from %s; changes this run will NOT be backed up.", self.repo_id)

    def start(self) -> None:
        """Push changes in the background, and once more on shutdown."""
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="ledger-backup", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            time.sleep(self.interval)
            self.push()

    # --- while running ------------------------------------------------------------

    def mark_dirty(self, images: list[Path] | None = None) -> None:
        """Record that the database changed (and which new photos to keep)."""
        if not self.enabled:
            return
        with self._lock:
            self._dirty = True
            for path in images or []:
                if path.name not in self.remote_images:
                    self._pending_images[path.name] = path

    def forget_images(self, names: set[str]) -> None:
        """Remove photos of deleted pages and documents from the backup too."""
        if not self.enabled or not names:
            return
        with self._lock:
            for name in names:
                self._pending_images.pop(name, None)
                if name in self.remote_images:
                    self._deleted_images.add(name)
            self._dirty = True

    def has_image(self, name: str) -> bool:
        return name in self.remote_images or name in self._pending_images

    def fetch_image(self, name: str, dest_dir: Path) -> Path | None:
        """Download a backed-up photo into *dest_dir*; None if unavailable."""
        if not self.enabled or name not in self.remote_images:
            return None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                local = self.api.hf_hub_download(self.repo_id, IMAGES + name, repo_type="dataset", local_dir=tmp)
                dest = dest_dir / name
                shutil.copyfile(local, dest)
                return dest
        except Exception as exc:
            log.warning("Backup: could not fetch %s: %s", name, exc)
            return None

    def push(self) -> bool:
        """Upload a snapshot of the database and any new photos. True if pushed."""
        if not (self.enabled and self.can_push and self._db_path):
            return False
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        with self._push_lock:
            with self._lock:
                if not self._dirty:
                    return False
                self._dirty = False
                images = dict(self._pending_images)
                deleted = set(self._deleted_images)
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    snapshot = Path(tmp) / DB_NAME
                    # sqlite's backup API gives a consistent copy even while the
                    # app keeps writing.
                    with sqlite3.connect(str(self._db_path)) as src, sqlite3.connect(str(snapshot)) as dst:
                        src.backup(dst)
                    operations = [CommitOperationAdd(path_in_repo=DB_NAME, path_or_fileobj=str(snapshot))]
                    operations += [
                        CommitOperationAdd(path_in_repo=IMAGES + name, path_or_fileobj=str(path))
                        for name, path in images.items() if path.is_file()
                    ]
                    operations += [CommitOperationDelete(path_in_repo=IMAGES + name) for name in sorted(deleted)]
                    self.api.create_commit(self.repo_id, operations, repo_type="dataset",
                                           commit_message="Back up saved documents")
            except Exception as exc:
                log.warning("Backup: push to %s failed, will retry: %s", self.repo_id, exc)
                with self._lock:
                    self._dirty = True
                return False
            with self._lock:
                for name in images:
                    self._pending_images.pop(name, None)
                    self.remote_images.add(name)
                for name in deleted:
                    self._deleted_images.discard(name)
                    self.remote_images.discard(name)
            self._pushes += 1
            if self._pushes % _SQUASH_EVERY == 0:
                try:
                    self.api.super_squash_history(self.repo_id, repo_type="dataset")
                except Exception as exc:
                    log.info("Backup: could not squash history: %s", exc)
            return True
