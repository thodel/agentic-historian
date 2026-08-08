"""
Unit tests for TrainingClient (#361).
All HTTP is faked with unittest.mock — no live gateway, no test server needed.
"""

from __future__ import annotations

from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest

from agents.training_client import (
    GatewayUnreachableError,
    TrainJob,
    TrainJobError,
    TrainingClient,
    submit_training_job,
)


# -----------------------------------------------------------------------
# Fake httpx.Response factory
# -----------------------------------------------------------------------

def _make_response(data, status: int = 200) -> MagicMock:
    """Return a mock httpx.Response whose .json() == data and .status_code == status."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.headers = {"content-type": "application/json"}
    resp.text = ""
    return resp


# -----------------------------------------------------------------------
# Test submit
# -----------------------------------------------------------------------

class TestSubmit:
    def test_submit_returns_job_id(self):
        with TrainingClient(base_url="http://localhost:8200") as c:
            with patch.object(c._client, "post") as mock_post:
                mock_post.return_value = _make_response({"job_id": "job-42", "status": "queued"})
                job_id = c.submit({"engine": "kraken", "model_id": "test-v1"})
        assert job_id == "job-42"
        mock_post.assert_called_once()

    def test_submit_500_raises_with_detail(self):
        with TrainingClient(base_url="http://localhost:8200") as c:
            with patch.object(c._client, "post") as mock_post:
                mock_post.return_value = _make_response(
                    {"detail": "network TMPDIR not supported"}, status=500
                )
                with pytest.raises(TrainJobError) as exc:
                    c.submit({"engine": "kraken"})
        assert exc.value.status_code == 500
        assert "network TMPDIR" in exc.value.detail

    def test_submit_422_passes_through_field_errors(self):
        with TrainingClient(base_url="http://localhost:8200") as c:
            with patch.object(c._client, "post") as mock_post:
                mock_post.return_value = _make_response(
                    {"detail": [{"loc": ["body", "engine"], "msg": "unknown engine"}]},
                    status=422,
                )
                with pytest.raises(TrainJobError) as exc:
                    c.submit({"engine": "invalid"})
        assert exc.value.status_code == 422

    def test_submit_no_job_id_raises(self):
        with TrainingClient(base_url="http://localhost:8200") as c:
            with patch.object(c._client, "post") as mock_post:
                mock_post.return_value = _make_response({"status": "queued"})
                with pytest.raises(TrainJobError) as exc:
                    c.submit({"engine": "kraken"})
        assert "no job_id" in str(exc.value)


# -----------------------------------------------------------------------
# Test get
# -----------------------------------------------------------------------

class TestGet:
    def test_get_returns_train_job(self):
        with TrainingClient(base_url="http://localhost:8200") as c:
            with patch.object(c._client, "get") as mock_get:
                mock_get.return_value = _make_response({
                    "id": "job-99",
                    "status": "training",
                    "stage": "train",
                    "request": {"engine": "kraken"},
                    "stages": [],
                    "log_tail": [],
                })
                job = c.get("job-99")
        assert isinstance(job, TrainJob)
        assert job.id == "job-99"
        assert job.status == "training"
        assert job.stage == "train"

    def test_get_404_raises_TrainJobError(self):
        with TrainingClient(base_url="http://localhost:8200") as c:
            with patch.object(c._client, "get") as mock_get:
                mock_get.return_value = _make_response({"detail": "not found"}, status=404)
                with pytest.raises(TrainJobError) as exc:
                    c.get("nonexistent")
        assert exc.value.status_code == 404

    def test_unknown_fields_are_tolerated(self):
        """Acceptance: unknown response fields are ignored (#361 acceptance)."""
        with TrainingClient(base_url="http://localhost:8200") as c:
            with patch.object(c._client, "get") as mock_get:
                mock_get.return_value = _make_response({
                    "id": "job-future",
                    "status": "completed",
                    "request": {},
                    "stages": [],
                    "log_tail": [],
                    "future_field": "should not break",
                    "nested": {"unknown": 1},
                })
                job = c.get("job-future")  # must not raise
        assert job.id == "job-future"


# -----------------------------------------------------------------------
# Test list
# -----------------------------------------------------------------------

class TestList:
    def test_list_returns_job_list(self):
        with TrainingClient(base_url="http://localhost:8200") as c:
            with patch.object(c._client, "get") as mock_get:
                mock_get.return_value = _make_response([
                    {"id": "job-1", "status": "completed", "request": {}, "stages": [], "log_tail": []},
                    {"id": "job-2", "status": "failed", "request": {}, "stages": [], "log_tail": []},
                ])
                jobs = c.list()
        assert len(jobs) == 2
        assert jobs[0].id == "job-1"
        assert jobs[1].status == "failed"


# -----------------------------------------------------------------------
# Test log
# -----------------------------------------------------------------------

class TestLog:
    def test_log_returns_lines(self):
        with TrainingClient(base_url="http://localhost:8200") as c:
            with patch.object(c._client, "get") as mock_get:
                mock_get.return_value = _make_response({
                    "lines": ["epoch 1 loss=0.5", "epoch 2 loss=0.3"]
                })
                lines = c.log("myjob")
        assert lines == ["epoch 1 loss=0.5", "epoch 2 loss=0.3"]

    def test_log_passes_stage_and_lines_params(self):
        with TrainingClient(base_url="http://localhost:8200") as c:
            with patch.object(c._client, "get") as mock_get:
                mock_get.return_value = _make_response({"lines": []})
                c.log("j", stage="prepare", lines=50)
        path, kwargs = mock_get.call_args[0], mock_get.call_args[1]
        assert path[0] == "/train/jobs/j/log"
        assert kwargs["params"]["stage"] == "prepare"
        assert kwargs["params"]["lines"] == 50


# -----------------------------------------------------------------------
# Test cancel / delete
# -----------------------------------------------------------------------

class TestCancelDelete:
    def test_cancel_returns_updated_job(self):
        with TrainingClient(base_url="http://localhost:8200") as c:
            with patch.object(c._client, "post") as mock_post:
                mock_post.return_value = _make_response({
                    "id": "job-42",
                    "status": "cancelled",
                    "request": {},
                    "stages": [],
                    "log_tail": [],
                })
                job = c.cancel("job-42")
        assert job.status == "cancelled"

    def test_delete_returns_job(self):
        with TrainingClient(base_url="http://localhost:8200") as c:
            with patch.object(c._client, "delete") as mock_delete:
                mock_delete.return_value = _make_response({
                    "id": "job-42",
                    "status": "cancelled",
                    "request": {},
                    "stages": [],
                    "log_tail": [],
                })
                job = c.delete("job-42")
        assert job.status == "cancelled"


# -----------------------------------------------------------------------
# Test unreachable gateway
# -----------------------------------------------------------------------

class TestGatewayUnreachable:
    def test_unreachable_raises_GatewayUnreachableError_naming_url(self):
        """Acceptance: unreachable gateway raises typed error naming the URL."""
        with TrainingClient(base_url="http://localhost:19999") as c:
            with pytest.raises(GatewayUnreachableError) as exc:
                c.get("any-job")
        assert "localhost:19999" in str(exc.value)


# -----------------------------------------------------------------------
# Test TrainJob helpers
# -----------------------------------------------------------------------

class TestTrainJobHelpers:
    def test_is_terminal(self):
        for status in ("completed", "failed", "cancelled"):
            job = TrainJob(id="x", status=status)
            assert job.is_terminal(), f"{status} should be terminal"
        for status in ("queued", "preparing", "training", "testing"):
            job = TrainJob(id="x", status=status)
            assert not job.is_terminal(), f"{status} should not be terminal"

    def test_is_running(self):
        for status in ("queued", "preparing", "compiling", "training", "testing", "registering"):
            job = TrainJob(id="x", status=status)
            assert job.is_running(), f"{status} should be running"
        job = TrainJob(id="x", status="completed")
        assert not job.is_running()


# -----------------------------------------------------------------------
# Test convenience helper
# -----------------------------------------------------------------------

class TestSubmitTrainingJobHelper:
    def test_one_shot_submit_returns_job_id(self):
        with patch("agents.training_client.TrainingClient") as MockClient:
            instance = MagicMock()
            instance.__enter__ = MagicMock(return_value=instance)
            instance.__exit__ = MagicMock(return_value=None)
            instance.submit.return_value = "job-helper-99"
            MockClient.return_value = instance
            job_id = submit_training_job({"engine": "kraken"})
        assert job_id == "job-helper-99"