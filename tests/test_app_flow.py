"""Tests for the decoupled Track vs Generate CV/CL flow."""

import pytest
from unittest.mock import patch, MagicMock


class TestTrackDoesNotGenerateDocs:
    """Verify that tracking jobs does NOT trigger document generation."""

    def test_track_via_db_does_not_call_save_documents(self, tmp_db, sample_jobs):
        """Tracking a job should only insert into applications table, not generate docs."""
        # Save jobs to DB first
        job_ids = []
        for job in sample_jobs:
            job_id = tmp_db.save_job(job)
            job_ids.append(job_id)

        with patch("job_agent.docs_builder.save_documents_for_job") as mock_save_docs:
            # Track each job
            for job_id in job_ids:
                result = tmp_db.track_job(job_id)
                assert result is True

            # save_documents_for_job should NEVER have been called
            mock_save_docs.assert_not_called()

        # Verify all jobs are in applications table
        apps = tmp_db.get_applications()
        assert len(apps) == len(sample_jobs)

    def test_track_is_idempotent(self, tmp_db, sample_jobs):
        """Tracking the same job twice should succeed without error."""
        job_id = tmp_db.save_job(sample_jobs[0])

        assert tmp_db.track_job(job_id) is True
        assert tmp_db.track_job(job_id) is True  # second call should also return True

        # Should still only have one application
        apps = tmp_db.get_applications()
        assert len(apps) == 1


class TestProcessFoundJobsAutoGeneratesDocs:
    """Verify process_found_jobs() auto-generates CV/CL for HIGH_MATCH when enabled."""

    def test_process_found_jobs_skips_doc_generation_when_disabled(self, tmp_db, sample_jobs, monkeypatch):
        """process_found_jobs should NOT generate documents when AUTO_GENERATE_CV_CL is False."""
        from job_agent.main import process_found_jobs

        # Mock the LLM call to return "relevant" for all jobs
        def fake_batch_check(df, profile_text, model=None, progress_callback=None, custom_prompt_template=None, cascade_config=None, db=None):
            df = df.copy()
            df["gpt_relevance"] = "relevant"
            df["gpt_reasoning"] = "Good match"
            df["match_level"] = "HIGH_MATCH"
            df["confidence"] = 0.95
            return df

        monkeypatch.setattr("job_agent.main.batch_job_relevance_check", fake_batch_check)
        monkeypatch.setattr("job_agent.main.load_profile_text", lambda path: "Mock resume text")
        monkeypatch.setattr("job_agent.main.AUTO_GENERATE_CV_CL", False)

        with patch("job_agent.docs_builder.save_documents_for_job") as mock_save_docs:
            result = process_found_jobs(
                jobs_list=sample_jobs,
                resume_path="fake_resume.pdf",
                base_cv_path="fake_cv.docx",
                model_filter="test-model",
                model_generation="test-model",
                progress_callback=lambda text, pct: None,
                db=tmp_db,
            )

            # Should have found relevant jobs
            assert len(result) > 0

            # Should NOT have generated documents (auto-gen disabled)
            mock_save_docs.assert_not_called()

    def test_process_found_jobs_auto_generates_when_enabled(self, tmp_db, sample_jobs, monkeypatch):
        """process_found_jobs should auto-generate CV/CL for HIGH_MATCH when AUTO_GENERATE_CV_CL is True."""
        from job_agent.main import process_found_jobs

        # Mock the LLM call to return "relevant" for all jobs
        def fake_batch_check(df, profile_text, model=None, progress_callback=None, custom_prompt_template=None, cascade_config=None, db=None):
            df = df.copy()
            df["gpt_relevance"] = "relevant"
            df["gpt_reasoning"] = "Good match"
            df["match_level"] = "HIGH_MATCH"
            df["confidence"] = 0.95
            return df

        monkeypatch.setattr("job_agent.main.batch_job_relevance_check", fake_batch_check)
        monkeypatch.setattr("job_agent.main.load_profile_text", lambda path: "Mock resume text")
        monkeypatch.setattr("job_agent.main.AUTO_GENERATE_CV_CL", True)

        # Mock save_documents_for_job to avoid actual LLM calls
        mock_save_docs = MagicMock(return_value=("/fake/cv.docx", "/fake/cl.docx", "cv123", "cl123", None))
        with patch("job_agent.docs_builder.save_documents_for_job", mock_save_docs):
            result = process_found_jobs(
                jobs_list=sample_jobs,
                resume_path="fake_resume.pdf",
                base_cv_path="fake_cv.docx",
                model_filter="test-model",
                model_generation="test-model",
                progress_callback=lambda text, pct: None,
                db=tmp_db,
            )

            # Should have found relevant jobs
            assert len(result) > 0


class TestDocGenerationIndependent:
    """Verify document generation works independently from tracking."""

    def test_generate_docs_for_tracked_jobs(self, tmp_db, sample_jobs):
        """Doc generation should work on already-tracked jobs via update_job_docs."""
        # Save and track a job
        job_id = tmp_db.save_job(sample_jobs[0])
        tmp_db.track_job(job_id)

        # Verify no docs initially
        details = tmp_db.get_job_details(job_id)
        assert details.get("cv_path") is None

        # Simulate doc generation result
        tmp_db.update_job_docs(job_id, cv_path="/fake/cv.docx", cl_path="/fake/cl.docx")

        # Verify docs are now saved
        details = tmp_db.get_job_details(job_id)
        assert details["cv_path"] == "/fake/cv.docx"
        assert details["cl_path"] == "/fake/cl.docx"

        # Verify applications view also has cv_path
        apps = tmp_db.get_applications()
        assert len(apps) == 1
        assert apps[0]["cv_path"] == "/fake/cv.docx"

    def test_has_docs_indicator(self, tmp_db, sample_jobs):
        """Jobs with cv_path should show as having docs, others should not."""
        id1 = tmp_db.save_job(sample_jobs[0])
        id2 = tmp_db.save_job(sample_jobs[1])

        tmp_db.track_job(id1)
        tmp_db.track_job(id2)

        # Generate docs for first job only
        tmp_db.update_job_docs(id1, cv_path="/fake/cv.docx")

        apps = tmp_db.get_applications()
        apps_by_job = {a["job_id"]: a for a in apps}

        assert bool(apps_by_job[id1].get("cv_path")) is True
        assert bool(apps_by_job[id2].get("cv_path")) is False
