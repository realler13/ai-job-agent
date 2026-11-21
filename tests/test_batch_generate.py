"""Batch CV generation runs as a background job (2026-09-26).

At the xhigh generation pin one job takes minutes, so the old synchronous
POST outlived every proxy timeout: the Kanban reported failure while the CVs
kept arriving. POST now starts a worker and returns at once; the page polls
/api/batch-generate/status.
"""
import threading
import time

import pytest


@pytest.fixture
def batch(tmp_path, monkeypatch):
    """TestClient with a scratch DB, fresh batch state, a stubbed profile
    load, and a per-job worker that blocks until the test releases it."""
    from fastapi.testclient import TestClient
    from job_agent import api, gpt_filter
    from job_agent.db_manager import DBManager

    monkeypatch.setattr(api, "_db", DBManager(db_path=str(tmp_path / "batch.db")))
    monkeypatch.setattr(api, "_batch_state", {
        "running": False, "total": 0, "done": 0, "current": None, "results": [],
        "error": None, "started_at": None, "finished_at": None})
    monkeypatch.setattr(gpt_filter, "load_profile_text", lambda path: "profile")

    gate = threading.Event()
    seen = []

    def fake_job(job_id, profile_text):
        seen.append(job_id)
        gate.wait(5)
        return {"job_id": job_id, "cv": f"cv/{job_id}.docx", "intel": "generated", "error": None}

    monkeypatch.setattr(api, "_process_batch_job", fake_job)
    yield SimpleBatch(TestClient(api.app), gate, seen)
    gate.set()   # never leave a worker thread blocked past the test


class SimpleBatch:
    def __init__(self, client, gate, seen):
        self.client, self.gate, self.seen = client, gate, seen

    def status(self):
        return self.client.get("/api/batch-generate/status").json()

    def wait_until(self, pred, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            s = self.status()
            if pred(s):
                return s
            time.sleep(0.02)
        raise AssertionError(f"batch never reached the expected state: {self.status()}")


def test_post_returns_at_once_and_the_status_tracks_progress(batch):
    r = batch.client.post("/api/batch-generate", json={"job_ids": [7, 8, 7]})
    assert r.status_code == 200
    assert r.json() == {"status": "started", "total": 2}   # duplicate id dropped

    s = batch.wait_until(lambda s: s["current"] == 7)
    assert s["running"] and s["done"] == 0 and s["total"] == 2

    batch.gate.set()
    s = batch.wait_until(lambda s: not s["running"])
    assert s["done"] == 2 and s["current"] is None and s["error"] is None
    assert [r["job_id"] for r in s["results"]] == [7, 8]
    assert s["finished_at"] and s["started_at"]
    assert batch.seen == [7, 8]


def test_second_batch_is_refused_while_one_runs(batch):
    """A double-click must not start two xhigh generation loops at once."""
    assert batch.client.post("/api/batch-generate", json={"job_ids": [1]}).status_code == 200
    batch.wait_until(lambda s: s["current"] == 1)

    r = batch.client.post("/api/batch-generate", json={"job_ids": [2]})
    assert r.status_code == 409

    batch.gate.set()
    s = batch.wait_until(lambda s: not s["running"])
    assert batch.seen == [1] and s["total"] == 1   # the refused batch never touched state


def test_a_finished_batch_frees_the_slot(batch):
    batch.gate.set()
    batch.client.post("/api/batch-generate", json={"job_ids": [1]})
    batch.wait_until(lambda s: not s["running"] and s["done"] == 1)
    r = batch.client.post("/api/batch-generate", json={"job_ids": [2]})
    assert r.status_code == 200
    s = batch.wait_until(lambda s: not s["running"] and s["done"] == 1 and s["results"][0]["job_id"] == 2)
    assert len(s["results"]) == 1   # results reset per batch


def test_empty_selection_is_rejected(batch):
    assert batch.client.post("/api/batch-generate", json={"job_ids": []}).status_code == 400
    assert batch.status()["running"] is False


def test_an_unreadable_profile_ends_the_batch_with_an_error(batch, monkeypatch):
    """If the worker dies before the first job, the page must see a finished
    batch with the reason — not a spinner that polls forever."""
    from job_agent import gpt_filter

    def boom(path):
        raise RuntimeError("profile unreadable")

    monkeypatch.setattr(gpt_filter, "load_profile_text", boom)
    batch.client.post("/api/batch-generate", json={"job_ids": [1, 2]})
    s = batch.wait_until(lambda s: not s["running"])
    assert s["error"] == "profile unreadable"
    assert s["done"] == 0 and batch.seen == []
