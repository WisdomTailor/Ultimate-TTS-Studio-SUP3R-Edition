"""Background history index scheduler that queues meta-path upserts off the main path."""

from __future__ import annotations

import threading
import time
from typing import Any


class HistoryIndexScheduler:
    """Queue history index upserts off the main generation path.

    The scheduler batches meta-path submissions and processes them on a daemon
    thread so that SQLite I/O and bundle parsing do not block the TTS generation
    completion path. Individual failures are silently dropped.
    """

    def __init__(self) -> None:
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop = False

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._stop and not self._queue:
                    break
                batch = self._queue[:]
                self._queue = []
            if not batch:
                time.sleep(0.5)
                continue
            for meta_path in batch:
                try:
                    from output_history_service import upsert_meta_file

                    upsert_meta_file(meta_path)
                except Exception:
                    pass

    def submit(self, meta_path: str | None) -> None:
        """Queue a meta-path for background indexing."""
        if not meta_path:
            return
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._stop = False
                self._worker = threading.Thread(target=self._run, daemon=True)
                self._worker.start()
            self._queue.append(meta_path)

    def shutdown(self) -> None:
        """Signal the worker to stop after draining the queue."""
        with self._lock:
            self._stop = True

    @property
    def pending_count(self) -> int:
        """Return the number of queued items awaiting processing."""
        with self._lock:
            return len(self._queue)

    @property
    def is_running(self) -> bool:
        """Return whether the background worker thread is alive."""
        with self._lock:
            return self._worker is not None and self._worker.is_alive()


__all__ = ["HistoryIndexScheduler"]
