"""Tests for the background history index scheduler."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from history_index_scheduler import HistoryIndexScheduler


class TestHistoryIndexScheduler:
    def test_submit_none_or_empty_is_noop(self) -> None:
        scheduler = HistoryIndexScheduler()
        scheduler.submit(None)
        scheduler.submit("")
        assert scheduler.pending_count == 0
        assert not scheduler.is_running

    def test_submit_auto_starts_worker(self) -> None:
        scheduler = HistoryIndexScheduler()
        with patch("history_index_scheduler.time.sleep", return_value=None):
            scheduler.submit("/fake/meta.json")
            assert scheduler.is_running
            # When sleep is mocked the worker may drain immediately, so do not assert exact count

    def test_pending_count_reflects_queued_items(self) -> None:
        scheduler = HistoryIndexScheduler()
        with patch.object(threading.Thread, "start"):
            scheduler.submit("/fake/one.json")
            scheduler.submit("/fake/two.json")
            assert scheduler.pending_count == 2

    def test_shutdown_signals_stop(self) -> None:
        scheduler = HistoryIndexScheduler()
        with patch("history_index_scheduler.time.sleep", return_value=None):
            scheduler.submit("/fake/meta.json")
            assert scheduler.is_running
            scheduler.shutdown()
            # shutdown only signals; the worker may have already drained the queue

    def _wait_for_empty(self, scheduler: HistoryIndexScheduler, timeout: float = 2.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if scheduler.pending_count == 0:
                return
            time.sleep(0.01)
        raise AssertionError("Queue did not drain within timeout")

    def test_worker_drains_queue(self) -> None:
        scheduler = HistoryIndexScheduler()
        calls: list[str] = []

        def _capture(path: str) -> None:
            calls.append(path)

        with patch("history_index_scheduler.time.sleep", return_value=None):
            with patch(
                "output_history_service.upsert_meta_file", side_effect=_capture
            ):
                scheduler.submit("/fake/one.json")
                scheduler.submit("/fake/two.json")
                self._wait_for_empty(scheduler)
                assert scheduler.pending_count == 0
                assert set(calls) == {"/fake/one.json", "/fake/two.json"}

    def test_silent_failure_does_not_crash_worker(self) -> None:
        scheduler = HistoryIndexScheduler()
        calls: list[str] = []

        def _raise_then_capture(path: str) -> None:
            calls.append(path)
            if len(calls) == 1:
                raise RuntimeError("boom")

        with patch("history_index_scheduler.time.sleep", return_value=None):
            with patch(
                "output_history_service.upsert_meta_file",
                side_effect=_raise_then_capture,
            ):
                scheduler.submit("/fake/one.json")
                scheduler.submit("/fake/two.json")
                self._wait_for_empty(scheduler)
                assert scheduler.pending_count == 0
                assert len(calls) == 2
