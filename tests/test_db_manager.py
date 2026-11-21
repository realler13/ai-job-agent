import threading
import pytest

from tests.fixtures.persona import HOME


class TestDBManager:
    def test_save_job_returns_id(self, tmp_db, sample_jobs):
        job_id = tmp_db.save_job(sample_jobs[0])
        assert job_id is not None
        assert isinstance(job_id, int)

    def test_save_job_dedup_by_link(self, tmp_db, sample_jobs):
        """Saving the same job twice returns the same ID."""
        id1 = tmp_db.save_job(sample_jobs[0])
        id2 = tmp_db.save_job(sample_jobs[0])
        assert id1 == id2

    def test_save_job_no_link_returns_none(self, tmp_db):
        job = {"title": "No Link Job", "company": "X"}
        result = tmp_db.save_job(job)
        assert result is None

    def test_job_exists(self, tmp_db, sample_jobs):
        assert not tmp_db.job_exists("https://example.com/job/1")
        tmp_db.save_job(sample_jobs[0])
        assert tmp_db.job_exists("https://example.com/job/1")

    def test_job_exists_empty_link(self, tmp_db):
        assert not tmp_db.job_exists("")
        assert not tmp_db.job_exists(None)

    def test_save_job_with_relevance(self, tmp_db, sample_jobs):
        job_id = tmp_db.save_job_with_relevance(
            sample_jobs[0],
            relevance="relevant",
            reasoning="Great match",
            match_level="HIGH_MATCH",
            confidence=0.95,
            session_id="test-session-1",
        )
        assert job_id is not None

        details = tmp_db.get_job_details(job_id)
        assert details["relevance"] == "relevant"
        assert details["match_level"] == "HIGH_MATCH"
        assert details["confidence"] == 0.95

    def test_save_job_with_relevance_updates_existing(self, tmp_db, sample_jobs):
        """If job already exists, save_job_with_relevance updates relevance."""
        id1 = tmp_db.save_job(sample_jobs[0])
        id2 = tmp_db.save_job_with_relevance(
            sample_jobs[0],
            relevance="relevant",
            match_level="HIGH_MATCH",
            confidence=0.9,
        )
        assert id1 == id2
        details = tmp_db.get_job_details(id1)
        assert details["relevance"] == "relevant"

    def test_update_job_match_result(self, tmp_db, sample_jobs):
        job_id = tmp_db.save_job(sample_jobs[0])
        result = tmp_db.update_job_match_result(job_id, "SOCIAL_TRAP", 0.2, "social media role", "SOCIAL_TRAP")
        assert result is True

        details = tmp_db.get_job_details(job_id)
        assert details["match_level"] == "SOCIAL_TRAP"
        assert details["relevance"] == "irrelevant"

    def test_get_session_jobs(self, tmp_db, sample_jobs):
        tmp_db.save_job(sample_jobs[0], session_id="sess-A")
        tmp_db.save_job(sample_jobs[1], session_id="sess-A")
        tmp_db.save_job(sample_jobs[2], session_id="sess-B")

        sess_a_jobs = tmp_db.get_session_jobs("sess-A")
        assert len(sess_a_jobs) == 2

        sess_b_jobs = tmp_db.get_session_jobs("sess-B")
        assert len(sess_b_jobs) == 1

    def test_migration_adds_new_columns(self, tmp_db):
        """Check that migration columns exist."""
        conn = tmp_db._get_thread_conn()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(scrapes)")
        columns = [info[1] for info in cursor.fetchall()]

        assert "session_id" in columns
        assert "confidence" in columns
        assert "match_level" in columns

    def test_transaction_rollback(self, tmp_db, sample_jobs):
        """Ensure transaction rolls back on error."""
        try:
            with tmp_db.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO scrapes (date, title, company, location, link, raw_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, ("2025-01-01", "Test", "Co", "Loc", "https://rollback-test.com", "{}"))
                # Force an error
                raise ValueError("Simulated error")
        except ValueError:
            pass

        assert not tmp_db.job_exists("https://rollback-test.com")

    def test_thread_safety(self, tmp_db, sample_jobs):
        """Multiple threads can save jobs concurrently."""
        results = []

        def save_job(job):
            job_id = tmp_db.save_job(job)
            results.append(job_id)

        threads = [threading.Thread(target=save_job, args=(job,)) for job in sample_jobs[:3]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_get_history_includes_match_level(self, tmp_db, sample_jobs):
        tmp_db.save_job_with_relevance(
            sample_jobs[0],
            relevance="relevant",
            match_level="HIGH_MATCH",
            confidence=0.95,
        )
        history = tmp_db.get_history()
        assert len(history) == 1
        assert history[0]["match_level"] == "HIGH_MATCH"
        assert history[0]["confidence"] == 0.95

    def test_clear_history_preserves_tracked(self, tmp_db, sample_jobs):
        id1 = tmp_db.save_job(sample_jobs[0])
        id2 = tmp_db.save_job(sample_jobs[1])
        tmp_db.track_job(id1)

        tmp_db.clear_history()

        # Tracked job preserved
        assert tmp_db.get_job_details(id1) is not None
        # Untracked job deleted
        assert tmp_db.get_job_details(id2) is None

    def test_context_manager(self, tmp_path):
        from job_agent.db_manager import DBManager
        db_path = str(tmp_path / "ctx_test.db")
        with DBManager(db_path=db_path) as db:
            db.save_job({"title": "Test", "job_url": "https://ctx.com"})
            assert db.job_exists("https://ctx.com")

    def test_get_session_results(self, tmp_db, sample_jobs):
        """get_session_results returns display-ready dicts for a session."""
        tmp_db.save_job_with_relevance(
            sample_jobs[0], relevance="relevant", match_level="HIGH_MATCH",
            confidence=0.95, session_id="sess-R",
        )
        tmp_db.save_job_with_relevance(
            sample_jobs[1], relevance="irrelevant", match_level="NO_MATCH",
            confidence=0.2, session_id="sess-R",
        )
        tmp_db.save_job(sample_jobs[2], session_id="other-sess")

        results = tmp_db.get_session_results("sess-R")
        assert len(results) == 2
        # Sorted by confidence DESC
        assert results[0]["confidence"] == 0.95
        # No raw_data in returned dicts
        assert "raw_data" not in results[0]

    def test_migration_adds_reviewed_column(self, tmp_db):
        """Check that the reviewed column exists after migration."""
        conn = tmp_db._get_thread_conn()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(scrapes)")
        columns = [info[1] for info in cursor.fetchall()]
        assert "reviewed" in columns

    def test_get_unreviewed_jobs(self, tmp_db, sample_jobs):
        """get_unreviewed_jobs returns only TIERED, unreviewed jobs — tier is
        the single feed axis (the pipeline always stamps it)."""
        tmp_db.save_job_with_relevance(
            sample_jobs[0], relevance="relevant", match_level="HIGH_MATCH",
            confidence=0.95, session_id="s1", tier="strong",
        )
        tmp_db.save_job_with_relevance(
            sample_jobs[1], relevance="relevant", match_level="POTENTIAL_MATCH",
            confidence=0.6, session_id="s1", tier="worth_a_look",
        )
        tmp_db.save_job_with_relevance(
            sample_jobs[2], relevance="irrelevant", match_level="NO_MATCH",
            confidence=0.1, session_id="s1",
        )

        unreviewed = tmp_db.get_unreviewed_jobs()
        assert len(unreviewed) == 2
        # HIGH_MATCH first (higher confidence)
        assert unreviewed[0]["match_level"] == "HIGH_MATCH"

    def test_untiered_high_match_stays_out_of_feed(self, tmp_db, sample_jobs):
        """A HIGH_MATCH row with tier=None (comp floor / prompt-wall sweeps
        untier deliberately and keep match_level) must NOT be re-admitted —
        the old ``OR match_level IN (...)`` clause leaked exactly that (a row
        the pay floor had removed reappeared in the Review feed)."""
        tmp_db.save_job_with_relevance(
            sample_jobs[0], relevance="relevant", match_level="HIGH_MATCH",
            confidence=0.95, session_id="s1", tier=None,
        )
        assert tmp_db.get_unreviewed_jobs() == []

    def test_mark_reviewed(self, tmp_db, sample_jobs):
        """mark_reviewed removes jobs from unreviewed queue."""
        id1 = tmp_db.save_job_with_relevance(
            sample_jobs[0], relevance="relevant", match_level="HIGH_MATCH",
            confidence=0.9, session_id="s1", tier="strong",
        )
        id2 = tmp_db.save_job_with_relevance(
            sample_jobs[1], relevance="relevant", match_level="POTENTIAL_MATCH",
            confidence=0.6, session_id="s1", tier="worth_a_look",
        )

        assert len(tmp_db.get_unreviewed_jobs()) == 2

        tmp_db.mark_reviewed([id1])
        unreviewed = tmp_db.get_unreviewed_jobs()
        assert len(unreviewed) == 1
        assert unreviewed[0]["id"] == id2

    def test_mark_reviewed_empty_list(self, tmp_db):
        """mark_reviewed with empty list does nothing."""
        tmp_db.mark_reviewed([])  # Should not raise

    def test_date_posted_stored_and_returned(self, tmp_db):
        """date_posted from job data is stored in DB and returned in queries."""
        job = {
            "title": "Podcast Producer",
            "company": "PodCo",
            "location": "Remote",
            "job_url": "https://podco.com/jobs/1",
            "date_posted": "2026-01-15",
        }
        tmp_db.save_job(job, session_id="s-dp")

        # get_session_results should include date_posted
        results = tmp_db.get_session_results("s-dp")
        assert len(results) == 1
        assert results[0]["date_posted"] == "2026-01-15"

        # get_history should include date_posted
        history = tmp_db.get_history(limit=10)
        match = [h for h in history if h["link"] == "https://podco.com/jobs/1"]
        assert len(match) == 1
        assert match[0]["date_posted"] == "2026-01-15"

    def test_date_posted_via_save_job_with_relevance(self, tmp_db):
        """date_posted is stored when using save_job_with_relevance."""
        job = {
            "title": "Audio Producer",
            "company": "AudioCo",
            "location": "Remote",
            "job_url": "https://audioco.com/1",
            "date_posted": "2026-02-01",
        }
        tmp_db.save_job_with_relevance(
            job, relevance="relevant", match_level="HIGH_MATCH",
            confidence=0.9, session_id="s-dp2",
        )
        results = tmp_db.get_session_results("s-dp2")
        assert results[0]["date_posted"] == "2026-02-01"


class TestFeedbackLoop:
    """Location suppression + company-scrutiny derived from the candidate's dismissals."""

    def test_add_and_get_suppressed_location(self, tmp_db):
        assert tmp_db.add_location_suppression("Austin, TX", source_job_id=1) is True
        assert "austin, tx" in tmp_db.get_suppressed_locations()

    def test_refuses_to_suppress_target_locations(self, tmp_db):
        # A mis-tagged dismissal must never blacklist the candidate's targets:
        # remote work or the profile's primary metro.
        assert tmp_db.add_location_suppression("Remote") is False
        assert tmp_db.add_location_suppression(HOME) is False
        assert tmp_db.add_location_suppression("Bellevue, WA") is False
        assert tmp_db.get_suppressed_locations() == set()

    def test_refuses_too_short_location_keys(self, tmp_db):
        # Bare abbreviations (two or three letters) are too ambiguous to store and must be refused.
        for k in ("LA", "CA", "SF", "US", "NYC"):
            assert tmp_db.add_location_suppression(k) is False
        assert tmp_db.get_suppressed_locations() == set()

    def test_remove_suppression(self, tmp_db):
        tmp_db.add_location_suppression("New York, NY")
        assert "new york, ny" in tmp_db.get_suppressed_locations()
        tmp_db.remove_location_suppression("New York, NY")
        assert "new york, ny" not in tmp_db.get_suppressed_locations()

    def test_get_job_location(self, tmp_db, sample_jobs):
        jid = tmp_db.save_job(sample_jobs[0])
        assert tmp_db.get_job_location(jid) == sample_jobs[0]["location"]

    def test_dismissed_company_counts_only_company_reasons(self, tmp_db, sample_jobs):
        """The caution must count ONLY company-level reasons — role/location
        dismissals (wrong_location etc.) must NOT flag a company, else a large
        employer gets penalised for posting roles in the wrong place."""
        def _save(company, url, reason):
            jid = tmp_db.save_job({**sample_jobs[0], "company": company, "job_url": url})
            tmp_db.update_rejection_reason(jid, reason)
        # "GoodCo" dismissed twice but only for WRONG LOCATION → must NOT flag.
        _save("GoodCo", "u/g1", "wrong_location")
        _save("GoodCo", "u/g2", "wrong_location")
        # "BadFitCo" dismissed twice for BAD COMPANY FIT → must flag.
        _save("BadFitCo", "u/b1", "bad_company_fit")
        _save("BadFitCo", "u/b2", "bad_company_fit")
        # "AgencyCo" — agency/vendor is a company signal too.
        _save("AgencyCo", "u/a1", "agency_vendor")
        _save("AgencyCo", "u/a2", "agency_vendor")
        counts = tmp_db.get_dismissed_company_counts(min_count=2)
        assert "goodco" not in counts               # location dismissals ignored
        assert counts.get("badfitco") == 2
        assert counts.get("agencyco") == 2


def test_graded_merge_fills_missing_pay_and_date(tmp_db):
    """A cross-source merge copies pay and date onto the surviving copy in
    memory (e.g. a LinkedIn copy inheriting pay from an Indeed copy). The DB
    row was saved before the merge, so the graded save must fill the missing
    pay and date, or the email and the Review feed disagree about the pay floor."""
    link = "https://www.linkedin.com/jobs/view/1"
    tmp_db.save_job({"title": "Event Producer", "company": "Analog", "location": "Portland, OR",
                     "job_url": link, "description": "d"})
    merged = {"title": "Event Producer", "company": "Analog", "location": "Portland, OR", "job_url": link,
              "description": "d", "min_amount": 90000, "max_amount": 110000, "interval": "yearly",
              "currency": "USD", "date_posted": "2026-03-04"}
    jid = tmp_db.save_job_with_relevance(merged, relevance="relevant", match_level="WORTH_A_LOOK",
                                         confidence=0.6, reasoning="fit", tier="worth_a_look")
    row = tmp_db._get_thread_conn().execute(
        "SELECT comp_yearly_max, comp_text, date_posted FROM scrapes WHERE id = ?", (jid,)).fetchone()
    assert row[0] == 110000 and row[1] and row[2] == "2026-03-04"
    # never overwrites what is stored
    merged2 = dict(merged, min_amount=200000, max_amount=250000, date_posted="2026-01-20")
    tmp_db.save_job_with_relevance(merged2, relevance="relevant", match_level="WORTH_A_LOOK",
                                   confidence=0.6, reasoning="fit", tier="worth_a_look")
    row = tmp_db._get_thread_conn().execute(
        "SELECT comp_yearly_max, date_posted FROM scrapes WHERE id = ?", (jid,)).fetchone()
    assert row == (110000, "2026-03-04")
