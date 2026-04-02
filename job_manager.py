"""Background job manager with subprocess crash isolation for TTS synthesis."""

from __future__ import annotations

import json
import logging
import multiprocessing
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
        from tts_service import TtsRequest, generate_tts

        req = TtsRequest(
            text=request_dict["text"],
            engine=request_dict.get("engine", "Kokoro TTS"),
            audio_format=request_dict.get("audio_format", "wav"),
            engine_params=request_dict.get("engine_params", {}),
        )
        result = generate_tts(req)

        job_data = json.loads(job_path.read_text(encoding="utf-8"))
        job_data["status"] = COMPLETED
        job_data["completed_at"] = time.time()
        job_data["result"] = {
            "status": result.status,
            "output_path": result.output_path or "",
        }
        if result.audio is not None:
            sample_rate, _audio = result.audio
            job_data["result"]["sample_rate"] = sample_rate
            job_data["result"]["audio_format"] = request_dict.get("audio_format", "wav")
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

    def __init__(self, jobs_dir: Path | None = None) -> None:
        self._jobs_dir = jobs_dir or Path("app_state") / "jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, BaseProcess] = {}
        self._lock = threading.Lock()

    @property
    def jobs_dir(self) -> Path:
        """Return the directory that stores job JSON state."""
        return self._jobs_dir

    def submit(self, request: JobRequest) -> str:
        """Submit a synthesis job and return its job identifier."""
        job_id = str(uuid.uuid4())
        job_info = JobInfo(
            id=job_id,
            status=PENDING,
            request=asdict(request),
            created_at=time.time(),
        )
        self._save(job_info)

        ctx = multiprocessing.get_context("spawn")
        process = ctx.Process(
            target=_worker,
            args=(job_id, str(self._jobs_dir), asdict(request)),
            daemon=True,
        )
        process.start()
        with self._lock:
            self._processes[job_id] = process
        logger.info("Job %s submitted (pid=%s)", job_id, process.pid)
        return job_id

    def get_status(self, job_id: str) -> JobInfo:
        """Return the current state for a known job."""
        info = self._load(job_id)
        if info is None:
            raise KeyError(f"Unknown job: {job_id}")

        if info.status in (PENDING, RUNNING):
            with self._lock:
                process = self._processes.get(job_id)
            if process is not None and not process.is_alive():
                info.status = FAILED
                info.error = info.error or (
                    f"Worker process exited unexpectedly (code {process.exitcode})"
                )
                info.completed_at = time.time()
                self._save(info)
        return info

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending or running job if possible."""
        info = self._load(job_id)
        if info is None:
            raise KeyError(f"Unknown job: {job_id}")
        if info.status not in (PENDING, RUNNING):
            return False

        with self._lock:
            process = self._processes.get(job_id)
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=3)

        info.status = CANCELLED
        info.completed_at = time.time()
        self._save(info)
        logger.info("Job %s cancelled", job_id)
        return True

    def list_jobs(self, limit: int = 50) -> list[JobInfo]:
        """List recent jobs ordered by creation time, newest first."""
        jobs: list[JobInfo] = []
        for path in self._jobs_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                jobs.append(JobInfo(**data))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

        jobs.sort(key=lambda info: (info.created_at, info.id), reverse=True)
        return jobs[:limit]

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


_manager: JobManager | None = None


def get_job_manager(jobs_dir: Path | None = None) -> JobManager:
    """Return the process-local JobManager singleton."""
    global _manager
    if _manager is None:
        _manager = JobManager(jobs_dir)
    return _manager


__all__ = [
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "JobRequest",
    "JobInfo",
    "JobManager",
    "get_job_manager",
    "_worker",
]
