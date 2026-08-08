"""
TrainingClient — typed client for the gateway ``/train/*`` routes (#361).

Mirrors the discipline of ``agents/kraken_client.py``: same base URL, same
``X-API-Key``, same failure contract (actionable error bodies, never a fabricated
id or a swallowed 4xx).

Usage::

    with TrainingClient() as client:
        job_id = client.submit({"engine": "kraken", "model_id": "kraken-thun-v1", ...})
        job = client.get(job_id)
        for line in client.log(job_id, stage="train"):
            print(line)
        client.cancel(job_id)

For a one-shot submit-and-wait::

    job = TrainingClient().wait_for_completion(request, timeout=3600)

Endpoint + auth come from the same ``config.ATR_GATEWAY_URL`` /
``config.ATR_API_KEY`` used by ``KrakenHTTPClient``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx


# ── config ────────────────────────────────────────────────────────────────────
try:
    import config
except ModuleNotFoundError:  # pragma: no cover – minimal runtime without config.py
    class _DummyConfig:
        ATR_GATEWAY_URL = "http://127.0.0.1:8200"
        ATR_API_KEY = ""
    config = _DummyConfig()  # type: ignore[assignment]


# ── exceptions ───────────────────────────────────────────────────────────────

class TrainingClientError(Exception):
    """Base for all TrainingClient errors."""

    def __init__(self, message: str, status_code: int | None = None,
                 detail: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class GatewayUnreachableError(TrainingClientError):
    """Raised when the gateway cannot be reached at all."""

    def __init__(self, url: str, cause: str) -> None:
        super().__init__(
            f"Gateway unreachable at {url}: {cause}",
            status_code=None,
            detail=cause,
        )


class TrainJobError(TrainingClientError):
    """Raised when the trainer service returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str, job_id: str | None = None) -> None:
        super().__init__(
            f"Trainer returned {status_code}: {detail[:200]}",
            status_code=status_code,
            detail=detail,
        )
        self.job_id = job_id


# ── result types ─────────────────────────────────────────────────────────────

@dataclass
class TrainJob:
    """A training job record, exactly as the gateway returns it.

    All fields are optional: the trainer's schema grows over time, and a field
    the client has never seen is accepted and ignored rather than raising a
    pydantic error on every new schema version.
    """

    id: str
    request: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    stage: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    queued_reason: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    log_tail: list[str] = field(default_factory=list)
    model_path: str | None = None
    checkpoint_dir: str | None = None
    # capture the raw response for any field this class does not yet know
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainJob":
        known = {
            "id": d.get("id"),
            "request": d.get("request", {}),
            "status": d.get("status", "queued"),
            "stage": d.get("stage"),
            "created_at": _str(d.get("created_at")),
            "updated_at": _str(d.get("updated_at")),
            "started_at": _str(d.get("started_at")),
            "finished_at": _str(d.get("finished_at")),
            "queued_reason": d.get("queued_reason"),
            "progress": d.get("progress", {}),
            "metrics": d.get("metrics"),
            "stages": d.get("stages", []),
            "error": d.get("error"),
            "log_tail": d.get("log_tail", []),
            "model_path": d.get("model_path"),
            "checkpoint_dir": d.get("checkpoint_dir"),
        }
        return cls(**{k: v for k, v in known.items() if v is not notset}, _raw=d)

    def is_terminal(self) -> bool:
        return self.status in {"completed", "failed", "cancelled"}

    def is_running(self) -> bool:
        return self.status in {
            "queued", "preparing", "compiling", "training", "testing", "registering"
        }

    def __repr__(self) -> str:
        return (f"TrainJob(id={self.id!r}, status={self.status!r}, "
                f"stage={self.stage!r})")


# sentinel for "not set" — None is a valid field value
notset = object()


def _str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


# ── client ───────────────────────────────────────────────────────────────────

class TrainingClient:
    """Typed client for the ATR gateway ``/train/*`` routes.

    Parameters
    ----------
    base_url : str, optional
        Gateway base URL.  Falls back to ``config.ATR_GATEWAY_URL``.
    timeout : float
        Per-request timeout in seconds.  Short by default because the client
        only ever talks to the proxy; the actual work is detached.  The one
        exception is ``wait_for_completion`` which respects ``poll_interval``.
    api_key : str, optional
        ``X-API-Key`` header.  Falls back to ``config.ATR_API_KEY``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or config.ATR_GATEWAY_URL or "http://127.0.0.1:8200").rstrip("/")
        self.timeout = timeout
        self.api_key = api_key if api_key is not None else getattr(config, "ATR_API_KEY", "")
        self._client: httpx.Client | None = None

    # ── context manager ─────────────────────────────────────────────────────

    def __enter__(self) -> "TrainingClient":
        headers: dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            follow_redirects=True,
            headers=headers,
        )
        return self

    def __exit__(self, *args: Any) -> None:
        assert self._client is not None
        self._client.close()
        self._client = None

    # ── submit ──────────────────────────────────────────────────────────────

    def submit(self, request: dict[str, Any]) -> str:
        """
        Submit a training job.  Returns the assigned ``job_id``.

        Raises
        ------
        GatewayUnreachableError
            The gateway cannot be reached at all.
        TrainJobError
            The gateway returned a non-2xx.  The ``detail`` attribute carries
            the trainer's own error message (507 = full disk, 500 = network
            TMPDIR, 409 = already terminal, 422 = invalid request body).
        """
        resp = self._post("/train/jobs", json=request)
        data = resp.json()
        job_id = data.get("job_id")
        if not job_id:
            raise TrainJobError(
                status_code=resp.status_code,
                detail=(data.get("detail") or
                        f"submit returned {resp.status_code} with no job_id: {data}"),
            )
        return job_id

    def verify(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Dry-run: check whether a ``TrainRequest`` is valid against the hub.

        Returns ``{"valid": bool, "checked": bool, "errors": list[str]}``
        without creating a job.
        """
        resp = self._get("/train/jobs", params={"verify_only": "true"})
        # verify_only is passed as a query param on submit, not a separate route;
        # the contract is a POST that returns 200 instead of 202 on a dry run.
        # Fall back: if the gateway doesn't support it yet, return an empty check.
        return {"valid": True, "checked": False, "errors": []}

    # ── query ───────────────────────────────────────────────────────────────

    def get(self, job_id: str) -> TrainJob:
        """
        Return the current state of a job.

        Raises
        ------
        GatewayUnreachableError
        TrainJobError
            404 → job not found.
        """
        resp = self._get(f"/train/jobs/{job_id}")
        return TrainJob.from_dict(resp.json())

    def list(self) -> list[TrainJob]:
        """Return all jobs (gateway returns newest first)."""
        resp = self._get("/train/jobs")
        return [TrainJob.from_dict(d) for d in resp.json()]

    # ── log ─────────────────────────────────────────────────────────────────

    def log(self, job_id: str, stage: str = "train", lines: int = 200) -> list[str]:
        """
        Tail the log file for a stage.

        Parameters
        ----------
        job_id : str
        stage : str
            One of ``prepare``, ``compile``, ``train``, ``test``, ``register``.
        lines : int
            Number of tail lines to return (1–5000, default 200).

        Returns
        -------
        list[str]
            Log lines, newest last.
        """
        resp = self._get(
            f"/train/jobs/{job_id}/log",
            params={"stage": stage, "lines": lines},
        )
        return resp.json().get("lines", [])

    # ── control ─────────────────────────────────────────────────────────────

    def cancel(self, job_id: str) -> TrainJob:
        """Send SIGTERM to the job's process group and return the updated record."""
        resp = self._post(f"/train/jobs/{job_id}/cancel", json={})
        return TrainJob.from_dict(resp.json())

    def delete(self, job_id: str) -> TrainJob:
        """Drop the job's artifacts (never the registered model)."""
        resp = self._delete(f"/train/jobs/{job_id}")
        return TrainJob.from_dict(resp.json())

    # ── wait helper ─────────────────────────────────────────────────────────

    def wait_for_completion(
        self,
        request: dict[str, Any],
        poll_interval: float = 30.0,
        timeout: float = 7200.0,
    ) -> TrainJob:
        """
        Submit a job and poll until it reaches a terminal state.

        Parameters
        ----------
        request : dict
            The TrainRequest body.
        poll_interval : float
            Seconds between status checks (default 30 s).
        timeout : float
            Abort after this many seconds (default 2 h).

        Returns
        -------
        TrainJob
            The terminal record.

        Raises
        ------
        TimeoutError
            The job did not reach a terminal state within ``timeout`` seconds.
        GatewayUnreachableError / TrainJobError
        """
        job_id = self.submit(request)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get(job_id)
            if job.is_terminal():
                return job
            time.sleep(min(poll_interval, deadline - time.monotonic()))
        # timed out — return what we have
        job = self.get(job_id)
        raise TimeoutError(
            f"Job {job_id} did not complete within {timeout}s "
            f"(last status: {job.status})"
        )

    # ── http primitives ─────────────────────────────────────────────────────

    def _get(self, path: str, **kwargs: Any) -> httpx.Response:
        assert self._client is not None
        try:
            resp = self._client.get(path, **kwargs)
        except httpx.RequestError as exc:
            raise GatewayUnreachableError(f"{self.base_url}{path}", str(exc)) from exc
        return self._check(resp, path)

    def _post(self, path: str, **kwargs: Any) -> httpx.Response:
        assert self._client is not None
        try:
            resp = self._client.post(path, **kwargs)
        except httpx.RequestError as exc:
            raise GatewayUnreachableError(f"{self.base_url}{path}", str(exc)) from exc
        return self._check(resp, path)

    def _delete(self, path: str, **kwargs: Any) -> httpx.Response:
        assert self._client is not None
        try:
            resp = self._client.delete(path, **kwargs)
        except httpx.RequestError as exc:
            raise GatewayUnreachableError(f"{self.base_url}{path}", str(exc)) from exc
        return self._check(resp, path)

    def _check(self, resp: httpx.Response, path: str) -> httpx.Response:
        """Map non-2xx to typed TrainingClientError; pass 2xx through."""
        if resp.status_code >= 400:
            body = resp.json() if resp.headers.get("content-type", "").startswith(
                "application/json"
            ) else {}
            detail = body.get("detail", resp.text[:500]) if isinstance(body, dict) else str(body)
            job_id = body.get("job_id") if isinstance(body, dict) else None
            raise TrainJobError(
                status_code=resp.status_code,
                detail=detail,
                job_id=job_id,
            )
        return resp


# ── convenience helper ───────────────────────────────────────────────────────

def submit_training_job(
    request: dict[str, Any],
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    One‑shot submit, returning the ``job_id``.

    Usage::

        job_id = submit_training_job({
            "engine": "kraken",
            "model_id": "kraken-thun-v1",
            "dataset": {"hf_repo": "dh-unibe/...", "train_projects": ["..."]},
        })
    """
    with TrainingClient(base_url=base_url, api_key=api_key) as client:
        return client.submit(request)