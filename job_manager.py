"""Background job manager with subprocess crash isolation for TTS synthesis."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("UTTS_MAX_CONCURRENT_JOBS", "1")))

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"


@dataclass
class JobRequest:
    """Serializable synthesis request for subprocess transport."""

    text: str
    engine: str = "Kokoro TTS"
    audio_format: str = "wav"
    job_type: str = "tts"
    engine_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobInfo:
    """Snapshot of a job's current state."""

    id: str
    status: str
    request: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str = ""
    created_at: float = 0.0
    queue_order: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0


def _worker(job_id: str, jobs_dir: str, request_dict: dict[str, Any]) -> None:
    """Run TTS synthesis in a child process and persist the result.

    Args:
        job_id: Job identifier for the JSON state file.
        jobs_dir: Path to the directory that stores job JSON files.
        request_dict: Serialized JobRequest payload.
    """
    job_path = Path(jobs_dir) / f"{job_id}.json"

    try:
        module_dir = str(Path(__file__).resolve().parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        job_data = json.loads(job_path.read_text(encoding="utf-8"))
        job_data["status"] = RUNNING
        job_data["started_at"] = time.time()
        job_path.write_text(json.dumps(job_data, indent=2), encoding="utf-8")
    except Exception:
        pass

    try:
        job_type = str(request_dict.get("job_type", "tts") or "tts").strip().lower()
        if job_type == "conversation":
            from conversation_job_service import generate_conversation_job

            result_payload = generate_conversation_job(request_dict)
        elif job_type == "single_speaker":
            from single_speaker_job_service import generate_single_speaker_job

            result_payload = generate_single_speaker_job(request_dict)
        else:
            from tts_service import TtsRequest, generate_tts

            engine_params = dict(request_dict.get("engine_params") or {})
            engine_params.setdefault("skip_file_saving", False)
            req = TtsRequest(
                text=request_dict["text"],
                engine=request_dict.get("engine", "Kokoro TTS"),
                audio_format=request_dict.get("audio_format", "wav"),
                engine_params=engine_params,
            )
            result = generate_tts(req)
            result_payload: dict[str, Any] = {
                "job_type": "tts",
                "status": result.status,
                "output_path": result.output_path or "",
            }
            if result.audio is not None:
                sample_rate, _audio = result.audio
                result_payload["sample_rate"] = sample_rate
                result_payload["audio_format"] = request_dict.get("audio_format", "wav")

        job_data = json.loads(job_path.read_text(encoding="utf-8"))
        job_data["status"] = COMPLETED
        job_data["completed_at"] = time.time()
        job_data["result"] = result_payload
        job_path.write_text(json.dumps(job_data, indent=2), encoding="utf-8")
    except Exception as exc:
        try:
            job_data = json.loads(job_path.read_text(encoding="utf-8"))
            job_data["status"] = FAILED
            job_data["completed_at"] = time.time()
            job_data["error"] = str(exc)
            job_path.write_text(json.dumps(job_data, indent=2), encoding="utf-8")
        except Exception:
            pass


class JobManager:
    """Manage background synthesis jobs with subprocess crash isolation."""

    def __init__(
        self,
        jobs_dir: Path | None = None,
        max_concurrent: int | None = None,
        poll_interval_seconds: float = 0.5,
    ) -> None:
        self._jobs_dir = jobs_dir or Path("app_state") / "jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, BaseProcess] = {}
        self._lock = threading.Lock()
        self._max_concurrent = max(1, int(max_concurrent or DEFAULT_MAX_CONCURRENT_JOBS))
        self._poll_interval_seconds = max(0.2, float(poll_interval_seconds))
        self._stop_event = threading.Event()
        with self._lock:
            self._recover_stale_jobs_locked()
            self._launch_pending_jobs_locked()
        self._supervisor_thread = threading.Thread(
            target=self._supervisor_loop,
            name="job-manager-supervisor",
            daemon=True,
        )
        self._supervisor_thread.start()

    @property
    def jobs_dir(self) -> Path:
        """Return the directory that stores job JSON state."""
        return self._jobs_dir

    @property
    def max_concurrent(self) -> int:
        """Return the scheduler concurrency limit."""
        return self._max_concurrent

    def submit(self, request: JobRequest) -> str:
        """Submit a synthesis job and return its job identifier."""
        created_at = time.time()
        job_id = str(uuid.uuid4())
        job_info = JobInfo(
            id=job_id,
            status=PENDING,
            request=asdict(request),
            created_at=created_at,
            queue_order=created_at,
        )
        with self._lock:
            self._save(job_info)
            self._normalize_pending_queue_orders_locked()
            self._reconcile_processes_locked()
            self._launch_pending_jobs_locked()
        logger.info("Job %s submitted as %s", job_id, request.job_type)
        return job_id

    def get_status(self, job_id: str) -> JobInfo:
        """Return the current state for a known job."""
        with self._lock:
            self._reconcile_processes_locked()
            self._launch_pending_jobs_locked()
            info = self._load(job_id)
        if info is None:
            raise KeyError(f"Unknown job: {job_id}")
        return info

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending or running job if possible."""
        with self._lock:
            self._reconcile_processes_locked()
            info = self._load(job_id)
            if info is None:
                raise KeyError(f"Unknown job: {job_id}")
            if info.status not in (PENDING, RUNNING):
                return False

            process = self._processes.get(job_id)
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(timeout=3)
            self._processes.pop(job_id, None)

            info.status = CANCELLED
            info.completed_at = time.time()
            self._save(info)
            self._normalize_pending_queue_orders_locked()
            self._launch_pending_jobs_locked()
        logger.info("Job %s cancelled", job_id)
        return True

    def move_pending(self, job_id: str, direction: str) -> tuple[bool, str]:
        """Move a pending job up or down within the queue."""
        normalized_direction = str(direction or "").strip().lower()
        if normalized_direction not in {"up", "down"}:
            raise ValueError(f"Unsupported queue move direction: {direction}")

        with self._lock:
            self._reconcile_processes_locked()
            info = self._load(job_id)
            if info is None:
                raise KeyError(f"Unknown job: {job_id}")
            if info.status != PENDING:
                return False, "Only pending jobs can be reordered."

            self._normalize_pending_queue_orders_locked()
            pending_jobs = [
                pending_info
                for pending_info in self._load_all_jobs_locked()
                if pending_info.status == PENDING
            ]
            pending_jobs.sort(key=self._pending_sort_key)

            current_index = next(
                (index for index, pending_info in enumerate(pending_jobs) if pending_info.id == job_id),
                -1,
            )
            if current_index < 0:
                return False, "Pending job was not found in the current queue."

            target_index = current_index - 1 if normalized_direction == "up" else current_index + 1
            if target_index < 0 or target_index >= len(pending_jobs):
                edge = "top" if normalized_direction == "up" else "bottom"
                return False, f"Job is already at the {edge} of the pending queue."

            current_job = pending_jobs[current_index]
            target_job = pending_jobs[target_index]
            current_job.queue_order, target_job.queue_order = (
                target_job.queue_order,
                current_job.queue_order,
            )
            self._save(current_job)
            self._save(target_job)
            self._normalize_pending_queue_orders_locked()
            return True, f"Moved job {job_id[:12]}... {normalized_direction} in the pending queue."

    def list_jobs(self, limit: int = 50) -> list[JobInfo]:
        """List jobs with active queue entries first and recent terminal jobs after."""
        with self._lock:
            self._reconcile_processes_locked()
            self._launch_pending_jobs_locked()
            jobs = self._load_all_jobs_locked()
        running_jobs = [info for info in jobs if info.status == RUNNING]
        pending_jobs = [info for info in jobs if info.status == PENDING]
        terminal_jobs = [info for info in jobs if info.status not in {RUNNING, PENDING}]

        running_jobs.sort(key=lambda info: (info.started_at or info.created_at, info.id))
        pending_jobs.sort(key=self._pending_sort_key)
        terminal_jobs.sort(
            key=lambda info: (info.completed_at or info.created_at, info.created_at, info.id),
            reverse=True,
        )
        return [*running_jobs, *pending_jobs, *terminal_jobs][:limit]

    def summarize(self) -> dict[str, int]:
        """Return counts for each job state across the full job store."""
        counts = {
            PENDING: 0,
            RUNNING: 0,
            COMPLETED: 0,
            FAILED: 0,
            CANCELLED: 0,
        }
        with self._lock:
            self._reconcile_processes_locked()
            self._launch_pending_jobs_locked()
            jobs = self._load_all_jobs_locked()

        for info in jobs:
            counts[info.status] = counts.get(info.status, 0) + 1
        return counts

    def _save(self, info: JobInfo) -> None:
        path = self._jobs_dir / f"{info.id}.json"
        path.write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")

    def _load(self, job_id: str) -> JobInfo | None:
        path = self._jobs_dir / f"{job_id}.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return JobInfo(**data)
        except (json.JSONDecodeError, TypeError, KeyError):
            return None

    def _load_all_jobs_locked(self) -> list[JobInfo]:
        jobs: list[JobInfo] = []
        for path in self._jobs_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                jobs.append(JobInfo(**data))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

        jobs.sort(key=lambda info: (info.created_at, info.id))
        return jobs

    def _pending_sort_key(self, info: JobInfo) -> tuple[float, float, str]:
        queue_value = info.queue_order if info.queue_order > 0 else info.created_at
        return (queue_value, info.created_at, info.id)

    def _normalize_pending_queue_orders_locked(self) -> None:
        pending_jobs = [info for info in self._load_all_jobs_locked() if info.status == PENDING]
        pending_jobs.sort(key=self._pending_sort_key)
        for index, info in enumerate(pending_jobs, start=1):
            normalized_order = float(index)
            if info.queue_order != normalized_order:
                info.queue_order = normalized_order
                self._save(info)

    def _recover_stale_jobs_locked(self) -> None:
        for info in self._load_all_jobs_locked():
            if info.status == RUNNING:
                info.status = FAILED
                info.completed_at = time.time()
                info.error = info.error or "Recovered stale running job after restart."
                self._save(info)
        self._normalize_pending_queue_orders_locked()

    def _reconcile_processes_locked(self) -> None:
        finished_job_ids: list[str] = []
        for job_id, process in list(self._processes.items()):
            if process.is_alive():
                continue
            process.join(timeout=0.1)
            info = self._load(job_id)
            if info is not None and info.status in {PENDING, RUNNING}:
                info.status = FAILED
                info.error = info.error or (
                    f"Worker process exited unexpectedly (code {process.exitcode})"
                )
                info.completed_at = time.time()
                self._save(info)
            finished_job_ids.append(job_id)

        for job_id in finished_job_ids:
            self._processes.pop(job_id, None)

    def _start_job_locked(self, info: JobInfo) -> None:
        ctx = multiprocessing.get_context("spawn")
        process = ctx.Process(
            target=_worker,
            args=(info.id, str(self._jobs_dir), dict(info.request or {})),
            daemon=True,
        )
        process.start()
        self._processes[info.id] = process
        logger.info("Job %s started (pid=%s)", info.id, process.pid)

    def _launch_pending_jobs_locked(self) -> None:
        active_count = sum(1 for process in self._processes.values() if process.is_alive())
        available_slots = max(0, self._max_concurrent - active_count)
        if available_slots <= 0:
            return

        self._normalize_pending_queue_orders_locked()
        for info in sorted(self._load_all_jobs_locked(), key=self._pending_sort_key):
            if available_slots <= 0:
                break
            if info.status != PENDING or info.id in self._processes:
                continue
            self._start_job_locked(info)
            available_slots -= 1

    def _supervisor_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval_seconds):
            with self._lock:
                self._reconcile_processes_locked()
                self._launch_pending_jobs_locked()


_manager: JobManager | None = None


def get_job_manager(
    jobs_dir: Path | None = None,
    max_concurrent: int | None = None,
) -> JobManager:
    """Return the process-local JobManager singleton."""
    global _manager
    if _manager is None:
        _manager = JobManager(jobs_dir, max_concurrent=max_concurrent)
    return _manager


__all__ = [
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "DEFAULT_MAX_CONCURRENT_JOBS",
    "JobRequest",
    "JobInfo",
    "JobManager",
    "get_job_manager",
    "_worker",
]
