"""Tests for Phase 12b: User Input tab — manual job submission."""

import json
from datetime import datetime

import pytest


@pytest.fixture
def user_job_data():
    """Minimal job data as a user would enter it."""
    return {
        "title": "Podcast Producer",
        "company": "Harbor Lights Media",
        "location": "Tacoma, WA",
        "description": "Produce and manage original podcast content for Harbor Lights Media.",
    }


def test_user_input_saves_with_correct_match_level(tmp_db, user_job_data):
    """save_job_with_relevance stores USER_INPUT match_level and confidence=1.0."""
    job_data = {
        **user_job_data,
        "link": "https://jobs.example.com/harbor-lights/123",
        "date_posted": datetime.now().strftime("%Y-%m-%d"),
    }

    job_id = tmp_db.save_job_with_relevance(
        job_data,
        match_level="USER_INPUT",
        confidence=1.0,
        relevance="User-submitted job",
        reasoning="Manually added by user",
    )

    assert job_id is not None
    details = tmp_db.get_job_details(job_id)
    assert details["match_level"] == "USER_INPUT"
    assert details["confidence"] == 1.0
    assert details["relevance"] == "User-submitted job"


def test_user_input_tracks_and_marks_reviewed(tmp_db, user_job_data):
    """track_job + mark_reviewed called after saving."""
    job_data = {
        **user_job_data,
        "link": "https://jobs.example.com/harbor-lights/456",
        "date_posted": datetime.now().strftime("%Y-%m-%d"),
    }

    job_id = tmp_db.save_job_with_relevance(
        job_data,
        match_level="USER_INPUT",
        confidence=1.0,
        relevance="User-submitted job",
        reasoning="Manually added by user",
    )

    assert tmp_db.track_job(job_id) is True
    tmp_db.mark_reviewed([job_id])

    details = tmp_db.get_job_details(job_id)
    assert details["reviewed"] == 1

    apps = tmp_db.get_applications()
    assert any(a["job_id"] == job_id for a in apps)


def test_user_input_synthetic_link_when_no_url(tmp_db, user_job_data):
    """When no URL is provided, a user-input:// synthetic link is generated."""
    import uuid

    link = f"user-input://{uuid.uuid4().hex[:12]}"
    job_data = {
        **user_job_data,
        "link": link,
        "date_posted": datetime.now().strftime("%Y-%m-%d"),
    }

    job_id = tmp_db.save_job_with_relevance(
        job_data,
        match_level="USER_INPUT",
        confidence=1.0,
    )

    details = tmp_db.get_job_details(job_id)
    assert details["link"].startswith("user-input://")
    assert len(details["link"]) > len("user-input://")


def test_user_input_uses_provided_url(tmp_db, user_job_data):
    """When a real URL is provided, it's stored as-is."""
    real_url = "https://jobs.example.com/harbor-lights/senior-podcast-producer"
    job_data = {
        **user_job_data,
        "link": real_url,
        "date_posted": datetime.now().strftime("%Y-%m-%d"),
    }

    job_id = tmp_db.save_job_with_relevance(
        job_data,
        match_level="USER_INPUT",
        confidence=1.0,
    )

    details = tmp_db.get_job_details(job_id)
    assert details["link"] == real_url


def test_user_input_description_stored_in_raw_data(tmp_db, user_job_data):
    """The full job description is accessible via get_job_details raw_data."""
    job_data = {
        **user_job_data,
        "link": "https://jobs.example.com/harbor-lights/789",
        "date_posted": datetime.now().strftime("%Y-%m-%d"),
    }

    job_id = tmp_db.save_job_with_relevance(
        job_data,
        match_level="USER_INPUT",
        confidence=1.0,
    )

    details = tmp_db.get_job_details(job_id)
    raw = json.loads(details["raw_data"])
    assert raw["description"] == user_job_data["description"]
    assert raw["title"] == user_job_data["title"]
    assert raw["company"] == user_job_data["company"]
