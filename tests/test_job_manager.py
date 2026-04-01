"""Tests for the job_manager module."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


import job_manager
from job_manager import (
    CANCELLED,
    COMPLETED,
    FAILED,
    PENDING,
    RUNNING,
    JobInfo,
    JobManager,
    JobRequest,
)


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    original = job_manager._manager
    job_manager._manager = None
    try:
        yield
    finally:
        job_manager._manager = original


class TestJobRequest:
    def test_defaults(self) -> None:
        req = JobRequest(text="hello")

        assert req.engine == "Kokoro TTS"
        assert req.audio_format == "wav"
        assert req.engine_params == {}

    def test_custom_params(self) -> None:
        req = JobRequest(text="test", engine="F5-TTS", engine_params={"voice": "af_heart"})

        assert req.engine == "F5-TTS"
        assert req.engine_params["voice"] == "af_heart"


class TestJobInfo:
    def test_defaults(self) -> None:
        info = JobInfo(id="abc", status=PENDING, request={"text": "hi"})

        assert info.result is None
        assert info.error == ""
        assert info.completed_at == 0.0


class TestJobManager:
    @pytest.fixture
    def manager(self, tmp_path: Path) -> JobManager:
        return JobManager(jobs_dir=tmp_path / "jobs")

    def test_submit_creates_job_file(self, manager: JobManager) -> None:
        with patch("job_manager.multiprocessing") as mock_mp:
            mock_ctx = MagicMock()
            mock_process = MagicMock(pid=1234)
            mock_ctx.Process.return_value = mock_process
            mock_mp.get_context.return_value = mock_ctx

            job_id = manager.submit(JobRequest(text="hello"))

        job_path = manager.jobs_dir / f"{job_id}.json"
        assert job_path.exists()
        data = json.loads(job_path.read_text(encoding="utf-8"))
        assert data["status"] == PENDING
        mock_mp.get_context.assert_called_once_with("spawn")
        mock_process.start.assert_called_once()

    def test_get_status_known_job(self, manager: JobManager) -> None:
        info = JobInfo(
            id="test-123",
            status=COMPLETED,
            request={"text": "hi"},
            result={"status": "ok", "output_path": "/tmp/out.wav"},
            completed_at=time.time(),
        )
        manager.jobs_dir.mkdir(parents=True, exist_ok=True)
        (manager.jobs_dir / "test-123.json").write_text(
            json.dumps(
                {
                    "id": info.id,
                    "status": info.status,
                    "request": info.request,
                    "result": info.result,
                    "error": info.error,
                    "created_at": info.created_at,
                    "started_at": info.started_at,
                    "completed_at": info.completed_at,
                }
            ),
            encoding="utf-8",
        )

        result = manager.get_status("test-123")
        assert result.status == COMPLETED
        assert result.result is not None
        assert result.result["output_path"] == "/tmp/out.wav"

    def test_get_status_unknown_job(self, manager: JobManager) -> None:
        with pytest.raises(KeyError, match="Unknown job"):
            manager.get_status("nonexistent")

    def test_cancel_running_job(self, manager: JobManager) -> None:
        job_data = {
            "id": "cancel-me",
            "status": RUNNING,
            "request": {"text": "hi"},
            "result": None,
            "error": "",
            "created_at": time.time(),
            "started_at": time.time(),
            "completed_at": 0.0,
        }
        manager.jobs_dir.mkdir(parents=True, exist_ok=True)
        (manager.jobs_dir / "cancel-me.json").write_text(json.dumps(job_data), encoding="utf-8")

        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        manager._processes["cancel-me"] = mock_process

        result = manager.cancel("cancel-me")
        assert result is True
        mock_process.terminate.assert_called_once()

        updated = json.loads((manager.jobs_dir / "cancel-me.json").read_text(encoding="utf-8"))
        assert updated["status"] == CANCELLED

    def test_cancel_uses_kill_when_process_survives_terminate(self, manager: JobManager) -> None:
        job_data = {
            "id": "hard-kill",
            "status": RUNNING,
            "request": {"text": "hi"},
            "result": None,
            "error": "",
            "created_at": time.time(),
            "started_at": time.time(),
            "completed_at": 0.0,
        }
        (manager.jobs_dir / "hard-kill.json").write_text(json.dumps(job_data), encoding="utf-8")

        mock_process = MagicMock()
        mock_process.is_alive.side_effect = [True, True, False]
        manager._processes["hard-kill"] = mock_process

        result = manager.cancel("hard-kill")

        assert result is True
        mock_process.kill.assert_called_once()

    def test_cancel_completed_job_returns_false(self, manager: JobManager) -> None:
        job_data = {
            "id": "done-job",
            "status": COMPLETED,
            "request": {"text": "hi"},
            "result": {"status": "ok"},
            "error": "",
            "created_at": time.time(),
            "started_at": 0.0,
            "completed_at": time.time(),
        }
        (manager.jobs_dir / "done-job.json").write_text(json.dumps(job_data), encoding="utf-8")

        result = manager.cancel("done-job")
        assert result is False

    def test_cancel_unknown_job(self, manager: JobManager) -> None:
        with pytest.raises(KeyError, match="Unknown job"):
            manager.cancel("nonexistent")

    def test_list_jobs_empty(self, manager: JobManager) -> None:
        assert manager.list_jobs() == []

    def test_list_jobs_returns_most_recent_first(self, manager: JobManager) -> None:
        manager.jobs_dir.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(["aaa", "bbb", "ccc"]):
            job_data = {
                "id": name,
                "status": COMPLETED,
                "request": {"text": "hi"},
                "result": None,
                "error": "",
                "created_at": float(index),
                "started_at": 0.0,
                "completed_at": 0.0,
            }
            (manager.jobs_dir / f"{name}.json").write_text(json.dumps(job_data), encoding="utf-8")

        jobs = manager.list_jobs(limit=2)
        assert [job.id for job in jobs] == ["ccc", "bbb"]

    def test_list_jobs_skips_corrupt_files(self, manager: JobManager) -> None:
        manager.jobs_dir.mkdir(parents=True, exist_ok=True)
        (manager.jobs_dir / "bad.json").write_text("not json", encoding="utf-8")
        job_data = {
            "id": "good",
            "status": COMPLETED,
            "request": {"text": "hi"},
            "result": None,
            "error": "",
            "created_at": time.time(),
            "started_at": 0.0,
            "completed_at": 0.0,
        }
        (manager.jobs_dir / "good.json").write_text(json.dumps(job_data), encoding="utf-8")

        jobs = manager.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].id == "good"

    def test_reconcile_dead_process(self, manager: JobManager) -> None:
        job_data = {
            "id": "zombie",
            "status": RUNNING,
            "request": {"text": "hi"},
            "result": None,
            "error": "",
            "created_at": time.time(),
            "started_at": time.time(),
            "completed_at": 0.0,
        }
        (manager.jobs_dir / "zombie.json").write_text(json.dumps(job_data), encoding="utf-8")

        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        mock_process.exitcode = -9
        manager._processes["zombie"] = mock_process

        info = manager.get_status("zombie")
        assert info.status == FAILED
        assert "exited unexpectedly" in info.error

    def test_get_job_manager_returns_singleton(self, tmp_path: Path) -> None:
        first = job_manager.get_job_manager(tmp_path / "jobs")
        second = job_manager.get_job_manager(tmp_path / "other")

        assert first is second
        assert first.jobs_dir == tmp_path / "jobs"


class TestWorker:
    def test_worker_success(self, tmp_path: Path) -> None:
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        job_id = "worker-test"
        request_dict = {
            "text": "hello",
            "engine": "Kokoro TTS",
            "audio_format": "wav",
            "engine_params": {},
        }
        job_data = {
            "id": job_id,
            "status": PENDING,
            "request": request_dict,
            "result": None,
            "error": "",
            "created_at": time.time(),
            "started_at": 0.0,
            "completed_at": 0.0,
        }
        (jobs_dir / f"{job_id}.json").write_text(json.dumps(job_data), encoding="utf-8")

        @dataclass
        class FakeTtsRequest:
            text: str
            engine: str
            audio_format: str = "wav"
            engine_params: dict[str, object] | None = None

        class FakeResult:
            def __init__(self) -> None:
                self.status = "Generated OK"
                self.output_path = "/tmp/output.wav"
                self.audio = (24000, [0.0, 1.0])

        calls: list[FakeTtsRequest] = []

        def fake_generate_tts(request: FakeTtsRequest) -> FakeResult:
            calls.append(request)
            return FakeResult()

        fake_module = ModuleType("tts_service")
        fake_module.TtsRequest = FakeTtsRequest
        fake_module.generate_tts = fake_generate_tts

        with patch.dict(sys.modules, {"tts_service": fake_module}):
            job_manager._worker(job_id, str(jobs_dir), request_dict)

        assert len(calls) == 1
        data = json.loads((jobs_dir / f"{job_id}.json").read_text(encoding="utf-8"))
        assert data["status"] == COMPLETED
        assert data["result"]["output_path"] == "/tmp/output.wav"
        assert data["result"]["sample_rate"] == 24000
        assert data["result"]["audio_format"] == "wav"

    def test_worker_failure_writes_error(self, tmp_path: Path) -> None:
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        job_id = "fail-test"
        request_dict = {
            "text": "hello",
            "engine": "Kokoro TTS",
            "audio_format": "wav",
            "engine_params": {},
        }
        job_data = {
            "id": job_id,
            "status": PENDING,
            "request": request_dict,
            "result": None,
            "error": "",
            "created_at": time.time(),
            "started_at": 0.0,
            "completed_at": 0.0,
        }
        (jobs_dir / f"{job_id}.json").write_text(json.dumps(job_data), encoding="utf-8")

        @dataclass
        class FakeTtsRequest:
            text: str
            engine: str
            audio_format: str = "wav"
            engine_params: dict[str, object] | None = None

        def fake_generate_tts(_request: FakeTtsRequest) -> object:
            raise RuntimeError("CUDA out of memory")

        fake_module = ModuleType("tts_service")
        fake_module.TtsRequest = FakeTtsRequest
        fake_module.generate_tts = fake_generate_tts

        with patch.dict(sys.modules, {"tts_service": fake_module}):
            job_manager._worker(job_id, str(jobs_dir), request_dict)

        data = json.loads((jobs_dir / f"{job_id}.json").read_text(encoding="utf-8"))
        assert data["status"] == FAILED
        assert "CUDA" in data["error"]


def test_module_has_zero_gradio_imports() -> None:
    source = (APP_DIR / "job_manager.py").read_text(encoding="utf-8")

    assert "import gradio" not in source
    assert "from gradio" not in source