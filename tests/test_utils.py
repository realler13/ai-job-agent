import pandas as pd
import pytest


class TestProcessAndFilterJobs:
    def test_drops_internal_columns(self):
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([{
            "title": "Senior Video Producer",
            "company": "TechCorp",
            "location": "Seattle, WA",
            "job_url": "https://example.com/1",
        }])

        result = process_and_filter_jobs(df)

        # Internal columns should be dropped
        assert "norm_title" not in result.columns
        assert "norm_company" not in result.columns
        assert "is_allowed_loc" not in result.columns

        # is_remote should still be present (user-facing)
        if not result.empty:
            assert "is_remote" in result.columns

    def test_exclusion_filtering(self):
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([
            {"title": "Senior Video Producer", "company": "A", "location": "Seattle, WA", "job_url": "https://a.com"},
            {"title": "Software Engineer", "company": "B", "location": "Seattle, WA", "job_url": "https://b.com"},
            {"title": "Video Editor", "company": "C", "location": "Seattle, WA", "job_url": "https://c.com"},
        ])

        result = process_and_filter_jobs(df)

        # Software Engineer and Video Editor should be filtered by EXCLUDE_PHRASES
        titles = result["title"].tolist()
        assert "Software Engineer" not in titles
        assert "Video Editor" not in titles

    def test_content_editor_not_excluded(self):
        """'editor' was removed from EXCLUDE_PHRASES — Content Editor should survive."""
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([
            {"title": "Content Editor", "company": "A", "location": "Seattle, WA", "job_url": "https://a.com"},
            {"title": "Editorial Director", "company": "B", "location": "Remote", "job_url": "https://b.com"},
        ])

        result = process_and_filter_jobs(df)
        titles = result["title"].tolist()
        assert "Content Editor" in titles
        assert "Editorial Director" in titles

    def test_location_filtering(self):
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([
            {"title": "Content Producer", "company": "A", "location": "Seattle, WA", "job_url": "https://a.com"},
            {"title": "Content Producer", "company": "B", "location": "New York, NY", "job_url": "https://b.com"},
            {"title": "Content Producer", "company": "C", "location": "Remote", "job_url": "https://c.com"},
        ])

        result = process_and_filter_jobs(df)
        locations = result["location"].tolist()

        assert "Seattle, WA" in locations
        assert "Remote" in locations
        assert "New York, NY" not in locations

    def test_null_location_passes_through(self):
        """Jobs with NULL/empty/None location should pass the pre-filter (let AI decide)."""
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([
            {"title": "Podcast Producer", "company": "Harbor Lights Media", "location": None, "job_url": "https://a.com"},
            {"title": "Audio Producer", "company": "PodCo", "location": "", "job_url": "https://b.com"},
            {"title": "Content Producer", "company": "MediaCo", "location": "None", "job_url": "https://c.com"},
        ])

        result = process_and_filter_jobs(df)
        assert len(result) == 3
        titles = result["title"].tolist()
        assert "Podcast Producer" in titles
        assert "Audio Producer" in titles
        assert "Content Producer" in titles

    def test_country_labelled_onsite_leaks_now_dropped(self):
        """On-site jobs must not slip through because the location carries a
        country label ('United States'/'USA') or a two-letter state code that
        is also a word ('LA', 'OK'). All must drop."""
        from job_agent.utils import process_and_filter_jobs
        df = pd.DataFrame([
            {"title": "Producer", "company": "Keep1", "location": "Seattle, WA", "job_url": "https://k1"},
            {"title": "Producer", "company": "Keep2", "location": "Remote, US", "job_url": "https://k2"},
            {"title": "Producer", "company": "Drop1", "location": "Orlando, Florida, USA", "job_url": "https://d1"},
            {"title": "Producer", "company": "Drop2", "location": "Lake Charles, LA", "job_url": "https://d2"},
            {"title": "Producer", "company": "Drop3", "location": "New York, New York, United States", "job_url": "https://d3"},
            {"title": "Producer", "company": "Drop4", "location": "Jay, OK", "job_url": "https://d4"},
        ])
        kept = process_and_filter_jobs(df)["company"].tolist()
        assert "Keep1" in kept and "Keep2" in kept
        for gone in ("Drop1", "Drop2", "Drop3", "Drop4"):
            assert gone not in kept

    def test_remote_in_title_does_not_rescue_onsite(self):
        """'Supervisor, Remote Operations' @ Bristol, CT is on-site — the word
        'remote' in the TITLE must not classify it as work-from-home."""
        from job_agent.utils import process_and_filter_jobs
        df = pd.DataFrame([
            {"title": "Supervisor, Remote Operations", "company": "ESPN",
             "location": "Bristol, CT, US", "job_url": "https://espn"},
        ])
        assert len(process_and_filter_jobs(df)) == 0

    def test_suppressed_locations_dropped(self):
        """the candidate's dismissed no-go locations remove future matches (loop closed)."""
        from job_agent.utils import process_and_filter_jobs
        df = pd.DataFrame([
            {"title": "Producer", "company": "SD", "location": "San Diego, CA", "job_url": "https://sd"},
            {"title": "Producer", "company": "SEA", "location": "Seattle, WA", "job_url": "https://sea"},
        ])
        kept = process_and_filter_jobs(df, suppressed_locations={"san diego, ca"})["company"].tolist()
        assert "SEA" in kept and "SD" not in kept

    def test_dedup_by_link(self):
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([
            {"title": "Content Producer", "company": "A", "location": "Seattle, WA", "job_url": "https://same.com"},
            {"title": "Content Producer Dupe", "company": "A", "location": "Seattle, WA", "job_url": "https://same.com"},
        ])

        result = process_and_filter_jobs(df)
        assert len(result) == 1

    def test_negative_keywords_filter_manufacturing(self):
        """Production Manager in a manufacturing context should be filtered out."""
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([
            {"title": "Production Manager", "company": "A", "location": "Seattle, WA",
             "job_url": "https://a.com", "description": "Oversee manufacturing plant operations"},
            {"title": "Production Operations Manager", "company": "B", "location": "Remote",
             "job_url": "https://b.com", "description": "Manage warehouse and assembly line"},
            {"title": "Content Production Manager", "company": "C", "location": "Seattle, WA",
             "job_url": "https://c.com", "description": "Lead video content production team"},
        ])

        result = process_and_filter_jobs(df)
        titles = result["title"].tolist()
        # Manufacturing/warehouse roles filtered out
        assert "Production Manager" not in titles
        assert "Production Operations Manager" not in titles
        # Media production role survives
        assert "Content Production Manager" in titles

    def test_negative_keywords_no_false_positive_on_clean_production(self):
        """Production Manager without negative keywords in desc should survive."""
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([
            {"title": "Production Manager", "company": "Netflix", "location": "Remote",
             "job_url": "https://n.com", "description": "Manage production schedule for original series"},
        ])

        result = process_and_filter_jobs(df)
        assert len(result) == 1
        assert result.iloc[0]["title"] == "Production Manager"

    def test_tech_pm_excluded_by_phrase(self):
        """Technical Program Manager and similar tech PM titles should be excluded."""
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([
            {"title": "Technical Program Manager", "company": "Google", "location": "Remote",
             "job_url": "https://a.com", "description": "Lead engineering programs"},
            {"title": "Technical Project Manager", "company": "Meta", "location": "Seattle, WA",
             "job_url": "https://b.com", "description": "Manage software delivery"},
            {"title": "Engineering Program Manager", "company": "Stripe", "location": "Remote",
             "job_url": "https://c.com", "description": "Drive eng roadmap"},
            {"title": "Creative Project Manager", "company": "Netflix", "location": "Remote",
             "job_url": "https://d.com", "description": "Manage content production pipeline"},
        ])

        result = process_and_filter_jobs(df)
        titles = result["title"].tolist()
        assert "Technical Program Manager" not in titles
        assert "Technical Project Manager" not in titles
        assert "Engineering Program Manager" not in titles
        # Creative PM should survive
        assert "Creative Project Manager" in titles

    def test_tech_pm_filtered_by_negative_keywords_in_description(self):
        """Project Manager with tech stack in description should be filtered."""
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([
            {"title": "Senior Project Manager", "company": "TechCo", "location": "Seattle, WA",
             "job_url": "https://a.com",
             "description": "Manage CI/CD pipelines, coordinate with engineering on microservices architecture"},
            {"title": "Program Manager", "company": "CloudCo", "location": "Remote",
             "job_url": "https://b.com",
             "description": "Drive AWS infrastructure migration, coordinate Kubernetes deployments"},
            {"title": "Program Manager", "company": "MediaCo", "location": "Remote",
             "job_url": "https://c.com",
             "description": "Manage podcast production schedule and coordinate with audio editors"},
        ])

        result = process_and_filter_jobs(df)
        titles = result["title"].tolist()
        companies = result["company"].tolist()
        # Tech PM roles with tech stack in description should be filtered
        assert "TechCo" not in companies
        assert "CloudCo" not in companies
        # Media PM should survive
        assert "MediaCo" in companies

    def test_product_manager_excluded(self):
        """Product Manager (not Producer) should be excluded."""
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([
            {"title": "Product Manager", "company": "A", "location": "Seattle, WA",
             "job_url": "https://a.com"},
            {"title": "Senior Product Manager, Growth", "company": "B", "location": "Remote",
             "job_url": "https://b.com"},
            {"title": "Content Producer", "company": "C", "location": "Remote",
             "job_url": "https://c.com"},
        ])

        result = process_and_filter_jobs(df)
        titles = result["title"].tolist()
        assert "Product Manager" not in titles
        assert "Senior Product Manager, Growth" not in titles
        assert "Content Producer" in titles

    def test_editor_producer_hybrid_rescued(self):
        """A craft title with a producer signal is rescued (kept for the cascade
        to tier); a pure craft title is still hard-dropped."""
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([
            {"title": "Video Editor", "company": "A", "location": "Remote", "job_url": "https://a.com"},
            {"title": "Video Editor / Producer", "company": "B", "location": "Remote", "job_url": "https://b.com"},
            {"title": "Camera Operator", "company": "C", "location": "Remote", "job_url": "https://c.com"},
            {"title": "Field Producer (Camera Operator)", "company": "D", "location": "Remote", "job_url": "https://d.com"},
            {"title": "Software Engineer / Video Editor", "company": "E", "location": "Remote", "job_url": "https://e.com"},
        ])
        titles = process_and_filter_jobs(df)["title"].tolist()
        assert "Video Editor" not in titles            # pure craft → dropped
        assert "Camera Operator" not in titles
        assert "Software Engineer / Video Editor" not in titles  # hard exclude wins
        assert "Video Editor / Producer" in titles      # hybrid → rescued
        assert "Field Producer (Camera Operator)" in titles

    def test_creative_product_manager_rescued(self):
        """A content/creative Product Manager is rescued; a generic one is dropped."""
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([
            {"title": "Creative Product Manager", "company": "A", "location": "Remote", "job_url": "https://a.com"},
            {"title": "Content Product Manager", "company": "B", "location": "Remote", "job_url": "https://b.com"},
            {"title": "Product Manager", "company": "C", "location": "Remote", "job_url": "https://c.com"},
            {"title": "Product Manager, Payments", "company": "D", "location": "Remote", "job_url": "https://d.com"},
        ])
        titles = process_and_filter_jobs(df)["title"].tolist()
        assert "Creative Product Manager" in titles
        assert "Content Product Manager" in titles
        assert "Product Manager" not in titles
        assert "Product Manager, Payments" not in titles

    def test_scrum_master_excluded(self):
        """Scrum Master should be excluded."""
        from job_agent.utils import process_and_filter_jobs

        df = pd.DataFrame([
            {"title": "Scrum Master", "company": "A", "location": "Remote", "job_url": "https://a.com"},
            {"title": "Video Producer", "company": "B", "location": "Remote", "job_url": "https://b.com"},
        ])

        result = process_and_filter_jobs(df)
        titles = result["title"].tolist()
        assert "Scrum Master" not in titles
        assert "Video Producer" in titles

    def test_empty_dataframe(self):
        from job_agent.utils import process_and_filter_jobs

        result = process_and_filter_jobs(pd.DataFrame())
        assert result.empty


class TestSessionManager:
    def test_session_lifecycle(self):
        from job_agent.session import SessionManager, SessionPhase

        session = SessionManager()
        assert session.phase == SessionPhase.IDLE
        assert not session.is_stopped
        assert not session.is_paused

        session.update_progress("SCRAPING", 0.5, "Scraping LinkedIn")
        assert session.phase == SessionPhase.SCRAPING
        assert session.progress == 0.5

    def test_stop_request(self):
        from job_agent.session import SessionManager

        session = SessionManager()
        assert not session.check_stop()

        session.request_stop()
        assert session.check_stop()
        assert session.is_stopped

    def test_toggle_pause(self):
        from job_agent.session import SessionManager

        session = SessionManager()
        assert not session.is_paused

        session.toggle_pause()
        assert session.is_paused

        session.toggle_pause()
        assert not session.is_paused

    def test_as_control_flags(self):
        from job_agent.session import SessionManager

        session = SessionManager()
        flags = session.as_control_flags()
        assert flags == {"stop": False, "pause": False}

        session.request_stop()
        flags = session.as_control_flags()
        assert flags["stop"] is True

    def test_session_id_uniqueness(self):
        from job_agent.session import SessionManager

        s1 = SessionManager()
        s2 = SessionManager()
        assert s1.id != s2.id

    def test_progress_callback(self):
        from job_agent.session import SessionManager

        session = SessionManager()
        cb = session.as_progress_callback()

        cb("Scraping LinkedIn...", 0.3)
        assert session.progress == 0.3
        assert "Scraping" in session.message or "LinkedIn" in session.message


class TestNormalizeDatePosted:
    def test_iso_datetime(self):
        from job_agent.utils import normalize_date_posted
        assert normalize_date_posted("2026-01-15T10:30:00-05:00") == "2026-01-15"

    def test_iso_date_only(self):
        from job_agent.utils import normalize_date_posted
        assert normalize_date_posted("2026-01-15") == "2026-01-15"

    def test_lever_unix_ms(self):
        from job_agent.utils import normalize_date_posted
        # 1706745600000 ms = 2024-02-01 00:00:00 UTC
        result = normalize_date_posted("1706745600000")
        assert result == "2024-02-01" or result.startswith("2024-0")  # timezone may shift day

    def test_pandas_timestamp(self):
        from job_agent.utils import normalize_date_posted
        ts = pd.Timestamp("2026-03-15")
        assert normalize_date_posted(ts) == "2026-03-15"

    def test_pandas_nat(self):
        from job_agent.utils import normalize_date_posted
        assert normalize_date_posted(pd.NaT) == ""

    def test_none(self):
        from job_agent.utils import normalize_date_posted
        assert normalize_date_posted(None) == ""

    def test_empty_string(self):
        from job_agent.utils import normalize_date_posted
        assert normalize_date_posted("") == ""

    def test_nan_float(self):
        from job_agent.utils import normalize_date_posted
        assert normalize_date_posted(float("nan")) == ""


class TestIsPostingOld:
    def test_old_posting(self):
        from job_agent.utils import is_posting_old
        from datetime import datetime, timedelta
        old_date = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
        assert is_posting_old(old_date, 30) is True

    def test_recent_posting(self):
        from job_agent.utils import is_posting_old
        from datetime import datetime, timedelta
        recent_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        assert is_posting_old(recent_date, 30) is False

    def test_empty_date(self):
        from job_agent.utils import is_posting_old
        assert is_posting_old("", 30) is False

    def test_invalid_date(self):
        from job_agent.utils import is_posting_old
        assert is_posting_old("not-a-date", 30) is False


class TestClosedListingsFilter:
    def test_linkedin_no_longer_accepting(self):
        from job_agent.utils import process_and_filter_jobs
        df = pd.DataFrame([
            {"title": "Content Producer", "company": "A", "location": "Seattle, WA",
             "job_url": "https://a.com", "description": "Great role. No longer accepting applications."},
            {"title": "Video Producer", "company": "B", "location": "Remote",
             "job_url": "https://b.com", "description": "Come join our team!"},
        ])
        result = process_and_filter_jobs(df)
        assert len(result) == 1
        assert result.iloc[0]["title"] == "Video Producer"

    def test_indeed_expired(self):
        from job_agent.utils import process_and_filter_jobs
        df = pd.DataFrame([
            {"title": "Content Producer", "company": "A", "location": "Seattle, WA",
             "job_url": "https://a.com",
             "description": "This job has expired on Indeed. The employer is not accepting applications."},
            {"title": "Video Producer", "company": "B", "location": "Remote",
             "job_url": "https://b.com", "description": "Exciting opportunity in media production"},
        ])
        result = process_and_filter_jobs(df)
        assert len(result) == 1
        assert result.iloc[0]["title"] == "Video Producer"

    def test_date_posted_normalization_in_pipeline(self):
        # Epoch-ms must normalize to YYYY-MM-DD. Uses a FRESH timestamp — a
        # hardcoded 2024 epoch now (correctly) trips the stale-posting gate.
        from datetime import datetime
        from job_agent.utils import process_and_filter_jobs
        now = datetime.now()
        df = pd.DataFrame([
            {"title": "Content Producer", "company": "A", "location": "Seattle, WA",
             "job_url": "https://a.com", "date_posted": str(int(now.timestamp() * 1000))},
        ])
        result = process_and_filter_jobs(df)
        assert not result.empty
        dp = result.iloc[0]["date_posted"]
        assert dp == now.strftime("%Y-%m-%d")


class TestGetJobHash:
    def test_basic_hash(self):
        from job_agent.utils import get_job_hash

        job = {"title": "Producer", "company": "ACME", "location": "Seattle"}
        h = get_job_hash(job)
        assert h == "producer|acme|seattle"

    def test_hash_case_insensitive(self):
        from job_agent.utils import get_job_hash

        j1 = {"title": "Producer", "company": "ACME", "location": "Seattle"}
        j2 = {"title": "PRODUCER", "company": "acme", "location": "SEATTLE"}
        assert get_job_hash(j1) == get_job_hash(j2)


class TestStripMarkdown:
    def test_strips_bold(self):
        from job_agent.cv_tailor import _strip_markdown
        assert _strip_markdown("**Strategic Leader** with experience") == "Strategic Leader with experience"

    def test_strips_italic(self):
        from job_agent.cv_tailor import _strip_markdown
        assert _strip_markdown("a *results-driven* professional") == "a results-driven professional"

    def test_strips_headers(self):
        from job_agent.cv_tailor import _strip_markdown
        assert _strip_markdown("## Summary\nSome text") == "Summary\nSome text"

    def test_strips_inline_code(self):
        from job_agent.cv_tailor import _strip_markdown
        assert _strip_markdown("Used `Python` and `SQL`") == "Used Python and SQL"

    def test_handles_empty(self):
        from job_agent.cv_tailor import _strip_markdown
        assert _strip_markdown("") == ""
        assert _strip_markdown(None) is None

    def test_preserves_plain_text(self):
        from job_agent.cv_tailor import _strip_markdown
        plain = "Senior producer with 7 years of experience in content strategy."
        assert _strip_markdown(plain) == plain

    def test_real_world_example(self):
        from job_agent.cv_tailor import _strip_markdown
        md = "**Hands-on Video Producer** with 7 years of experience driving the **planning, budgeting, and delivery** of branded series."
        expected = "Hands-on Video Producer with 7 years of experience driving the planning, budgeting, and delivery of branded series."
        assert _strip_markdown(md) == expected


class TestNegativeKeywordWordBoundaries:
    """Word-boundary negative matching — substring hits were killing flagship roles."""

    def test_civil_in_eeo_boilerplate_does_not_kill(self):
        """'civil' in EEO boilerplate must not kill a video role: the negative
        keyword is phrase-anchored to the engineering discipline
        ('civil engineer(ing)')."""
        from job_agent.utils import process_and_filter_jobs
        df = pd.DataFrame([
            {"title": "Head of Video Production", "company": "Harbor Lights Media", "location": "Seattle, WA",
             "job_url": "https://e.com",
             "description": "Lead video production. EEO: race, religion, civil union status protected."},
            {"title": "Production Manager", "company": "Ridgeline Civil Works", "location": "Seattle, WA",
             "job_url": "https://a.com",
             "description": "Oversee civil engineering project delivery."},
        ])
        titles = process_and_filter_jobs(df)["title"].tolist()
        assert "Head of Video Production" in titles      # boilerplate survives
        assert "Production Manager" not in titles        # real civil-eng role killed

    def test_aws_inside_laws_does_not_kill(self):
        from job_agent.utils import process_and_filter_jobs
        df = pd.DataFrame([
            {"title": "Video Production Manager", "company": "Red Cross", "location": "Remote",
             "job_url": "https://r.com",
             "description": "Comply with all applicable laws and regulations while producing video."},
        ])
        titles = process_and_filter_jobs(df)["title"].tolist()
        assert "Video Production Manager" in titles  # "laws" must not match "aws"

    def test_single_tech_mention_in_description_survives(self):
        from job_agent.utils import process_and_filter_jobs
        df = pd.DataFrame([
            {"title": "Content Production Manager", "company": "X", "location": "Remote",
             "job_url": "https://x.com",
             "description": "Manage our video pipeline. Familiarity with SQL dashboards a plus."},
        ])
        titles = process_and_filter_jobs(df)["title"].tolist()
        assert "Content Production Manager" in titles  # one passing tool mention

    def test_two_distinct_tech_terms_in_description_killed(self):
        from job_agent.utils import process_and_filter_jobs
        df = pd.DataFrame([
            {"title": "Technical Production Manager", "company": "Y", "location": "Remote",
             "job_url": "https://y.com",
             "description": "Deploy with Kubernetes and Terraform across our infrastructure."},
        ])
        assert len(process_and_filter_jobs(df)) == 0

    def test_tech_term_in_title_killed(self):
        from job_agent.utils import process_and_filter_jobs
        df = pd.DataFrame([
            {"title": "Production Manager, AWS Infrastructure", "company": "Z", "location": "Remote",
             "job_url": "https://z.com", "description": "Run cloud operations."},
        ])
        assert len(process_and_filter_jobs(df)) == 0


class TestLocationTokenMatching:
    def test_country_level_remote_passes(self):
        from job_agent.utils import process_and_filter_jobs
        df = pd.DataFrame([
            {"title": "Content Producer", "company": "A", "location": "United States", "job_url": "https://a.com"},
            {"title": "Video Producer", "company": "B", "location": "USA", "job_url": "https://b.com"},
            {"title": "Podcast Producer", "company": "C", "location": "Anywhere", "job_url": "https://c.com"},
        ])
        assert len(process_and_filter_jobs(df)) == 3

    def test_two_letter_terms_require_whole_token(self):
        from job_agent.utils import process_and_filter_jobs
        df = pd.DataFrame([
            # The home-state code is a whole word: "wa" inside "Ottawa" must NOT
            # read as Washington; a ", WA" token must.
            {"title": "Content Producer", "company": "A", "location": "Ottawa, IL", "job_url": "https://a.com"},
            {"title": "Content Producer", "company": "B", "location": "Spokane, WA", "job_url": "https://b.com"},
        ])
        locs = process_and_filter_jobs(df)["location"].tolist()
        assert "Ottawa, IL" not in locs
        assert "Spokane, WA" in locs


class TestThreadTimeoutFallback:
    """_run_with_timeout must enforce the budget OFF the main thread too —
    it used to silently run with NO timeout there (the FastAPI search thread),
    so one hung scrape froze a UI-triggered search until a service restart."""

    def test_timeout_enforced_in_background_thread(self):
        import threading
        import time as _time
        from job_agent.main import _run_with_timeout, _ScrapeTimeout

        outcome = {}

        def worker():
            try:
                _run_with_timeout(_time.sleep, args=(10,), timeout_sec=1, label="hang-test")
                outcome["result"] = "no-timeout"
            except _ScrapeTimeout:
                outcome["result"] = "timed-out"

        t = threading.Thread(target=worker)
        start = _time.time()
        t.start()
        t.join(timeout=8)
        assert outcome.get("result") == "timed-out"
        assert _time.time() - start < 8  # pipeline advanced, didn't wait 10s

    def test_result_passthrough_in_background_thread(self):
        import threading
        from job_agent.main import _run_with_timeout

        outcome = {}

        def worker():
            outcome["result"] = _run_with_timeout(lambda x: x * 2, args=(21,), timeout_sec=5, label="ok-test")

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=8)
        assert outcome.get("result") == 42


class TestClassifyLocation:
    """Geographic classification against the fixture profile (Seattle metro
    primary, Portland metro secondary), incl. two-letter state-code traps."""
    import pytest as _pytest

    @_pytest.mark.parametrize("loc,expected", [
        # Primary metro — top priority
        ("Seattle, WA", "primary"),
        ("Bellevue, WA", "primary"),
        ("Redmond", "primary"),
        # Remote (from the location field) — actionable
        ("Remote", "remote"),
        ("Remote, US", "remote"),
        ("Remote (Work from home)", "remote"),
        ("Anywhere", "remote"),
        # Secondary metro on-site — capped, not rejected
        ("Portland", "secondary"),
        ("Portland, OR", "secondary"),
        ("Beaverton, OR", "secondary"),
        ("Vancouver, WA", "secondary"),
        # Elsewhere in the home state (search.locations.state_code); passes to the judge
        ("Spokane, WA", "home_state"),
        ("Olympia, Washington", "home_state"),
        ("Walla Walla WA", "home_state"),
        # Another state is out of area
        ("San Diego, CA", "reject"),
        ("Sacramento, California", "reject"),
        # Unknown — empty or bare national label, pass to judge
        ("", "unknown"),
        ("None", "unknown"),
        ("United States", "unknown"),
        ("USA", "unknown"),
        # REJECT — concrete on-site places outside the profile's area
        ("New York, NY", "reject"),
        ("Austin, TX", "reject"),
        ("Lake Charles, LA", "reject"),        # Louisiana state code
        ("Jay, OK", "reject"),
        ("Orlando, Florida, USA", "reject"),
        ("Bristol, CT, US", "reject"),
        ("Denver, Colorado, United States", "reject"),
        ("Washington, DC", "reject"),
        ("Boise, ID", "reject"),
        # REJECT — remote pinned to a non-US region, or a foreign city
        ("Remote - India", "reject"),
        ("Remote, EMEA", "reject"),
        ("London, United Kingdom", "reject"),
        ("Vancouver, BC, Canada", "reject"),   # not Vancouver, WA
    ])
    def test_classify(self, loc, expected):
        from job_agent.config import classify_location
        assert classify_location(loc) == expected

    @_pytest.mark.parametrize("loc,expected", [
        # Multi-location with an in-area option → actionable (must NOT reject on
        # the co-listed out-of-area state).
        ("Austin, Texas, United States; Spokane, Washington, United States", "home_state"),
        ("Dallas, Texas; Olympia, WA", "home_state"),
        ("New York, NY; Seattle, WA", "primary"),   # primary-metro option wins
        ("Austin, TX; Bellevue, WA", "primary"),
        # Multi-location with NO in-area option → still reject.
        ("Austin, TX; Denver, CO", "reject"),
        # Foreign 'CA' (Canada) must not read as a metro or the home state.
        ("Vancouver, CA", "reject"),
        # "Washington, DC" is the District, not the home state.
        ("Washington, DC; Austin, TX", "reject"),
    ])
    def test_classify_multi_location(self, loc, expected):
        from job_agent.config import classify_location
        assert classify_location(loc) == expected

    def test_remote_from_title_is_ignored(self):
        from job_agent.config import classify_location
        # 'remote' only in the title, real location is on-site CT → reject
        assert classify_location("Bristol, CT", "Supervisor, Remote Operations") == "reject"

    @_pytest.mark.parametrize("loc,expected", [
        # Multi-location: surface the in-area option that makes it actionable.
        ("Austin, TX; Bellevue, WA; Denver, CO", "Bellevue, WA"),
        ("Austin, Texas, United States; Spokane, Washington, United States; Denver, CO",
         "Spokane, Washington, United States"),
        ("New York, NY; Seattle, WA", "Seattle, WA"),       # primary metro preferred
        ("Dallas, TX / Olympia, WA", "Olympia, WA"),
        ("Austin, TX; Portland, OR", "Portland, OR"),       # secondary metro, last resort
        # Nothing to disambiguate → None.
        ("Seattle, WA", None),                  # single location
        ("Remote", None),                       # single, remote
        ("Seattle, WA; Tacoma, WA", None),      # all in-area already
        ("Austin, TX; Denver, CO", None),       # no in-area option at all
    ])
    def test_extract_actionable_location(self, loc, expected):
        from job_agent.config import extract_actionable_location
        assert extract_actionable_location(loc) == expected

    def test_location_priority_maps_from_classify(self):
        from job_agent.config import location_priority
        from tests.fixtures.persona import HOME
        assert location_priority(HOME) == 0               # primary metro
        assert location_priority("Remote") == 1           # remote
        assert location_priority("Portland, OR") == 3     # secondary-metro cap
        assert location_priority("United States") == 2    # unknown/other
        assert location_priority("Austin, TX") == 2       # reject→2 (pre-filter drops it)


class TestLocationSuppression:
    """is_location_suppressed: exact SEGMENT match, respects multi-location.
    Regression coverage for a substring match that dropped unrelated places."""
    import pytest as _pytest

    def test_exact_single_location_dropped(self):
        from job_agent.config import is_location_suppressed
        assert is_location_suppressed("San Diego, CA", {"san diego, ca"}) is True

    @_pytest.mark.parametrize("job_loc,supp", [
        ("Atlanta, GA", "la"),          # 'la' must NOT substring-match Atlanta
        ("Tallahassee, FL", "la"),      # nor Tallahassee
        ("Lakewood, WA", "la"),         # nor Lakewood
        ("Chicago, IL", "ca"),          # 'ca' must NOT substring-match Chicago
        ("Fresno, CA", "san diego, ca"),  # another city in that state, not suppressed
    ])
    def test_no_substring_collateral(self, job_loc, supp):
        from job_agent.config import is_location_suppressed
        assert is_location_suppressed(job_loc, {supp}) is False

    def test_multilocation_with_unsuppressed_in_area_option_kept(self):
        from job_agent.config import is_location_suppressed
        # NY suppressed, but Seattle is still an actionable option → keep (the
        # case the multi-location rule protects; substring matching broke it).
        assert is_location_suppressed(
            "New York, NY; Seattle, WA", {"new york, ny"}) is False

    def test_multilocation_all_actionable_suppressed_dropped(self):
        from job_agent.config import is_location_suppressed
        assert is_location_suppressed(
            "Austin, TX; Spokane, WA", {"spokane, wa"}) is True

    @_pytest.mark.parametrize("variant", [
        "Spokane, WA",
        "Spokane, WA, US",
        "Spokane, WA, USA",
        "Spokane, Washington",
        "Spokane, Washington, United States",
        "Spokane,Washington,United States of America",
        "Spokane, WA (Hybrid)",
    ])
    def test_suppression_matches_location_variants(self, variant):
        """A stored 'spokane, wa' must catch the same place written any of the
        common ways (the home state's name reads as its code) — else the ', US'
        suffix form leaks through."""
        from job_agent.config import is_location_suppressed
        assert is_location_suppressed(variant, {"spokane, wa"}) is True

    def test_normalization_does_not_conflate_different_cities(self):
        from job_agent.config import is_location_suppressed
        # normalizing the suffix must NOT make Olympia look like Spokane
        assert is_location_suppressed("Olympia, WA, US", {"spokane, wa"}) is False
        assert is_location_suppressed("Seattle, WA", {"spokane, wa"}) is False

    def test_prefilter_multilocation_kept_when_in_area_option_not_suppressed(self):
        """End-to-end through the pre-filter: a stored 'new york, ny' must NOT
        drop a NY+Seattle posting (defeating this would undo the multi-location rule)."""
        import pandas as pd
        from job_agent.utils import process_and_filter_jobs
        df = pd.DataFrame([
            {"title": "Producer", "company": "Multi", "location": "New York, NY; Seattle, WA", "job_url": "https://m"},
        ])
        kept = process_and_filter_jobs(df, suppressed_locations={"new york, ny"})
        assert len(kept) == 1


class TestActionableLocationSeparators:
    """extract_actionable_location must recognize ' or ' (a common multi-loc join)."""

    def test_or_separator_surfaces_the_in_area_option(self):
        from job_agent.config import extract_actionable_location as ex
        assert ex("New York or Seattle") == "Seattle"
        assert ex("New York, NY Or Portland, OR") == "Portland, OR"

    def test_or_does_not_oversplit_single_place(self):
        from job_agent.config import extract_actionable_location as ex
        # 'Oregon'/'Corvallis' contain 'or' but not as a whitespace-bounded token.
        assert ex("Portland, Oregon") is None
        assert ex("Corvallis, OR") is None

    def test_jd_blob_not_surfaced_as_badge(self):
        from job_agent.config import extract_actionable_location as ex
        blob = "We are hiring in Seattle " + ("x" * 200) + "; New York, NY"
        # too long to be a real location segment → no badge
        assert ex(blob) is None


class TestStalePostingGate:
    """The stale-posting gate is wired into the pre-filter: postings older
    than STALE_POSTING_DAYS drop; blank/unknown dates pass."""

    def _df(self, dates):
        import pandas as pd
        return pd.DataFrame({
            "title": ["Producer"] * len(dates),
            "company": [f"Co{i}" for i in range(len(dates))],
            "location": ["Remote"] * len(dates),
            "link": [f"https://x.example/{i}" for i in range(len(dates))],
            "description": ["Video production role"] * len(dates),
            "date_posted": dates,
        })

    def test_stale_dropped_fresh_and_blank_kept(self):
        from datetime import datetime, timedelta
        from job_agent.utils import process_and_filter_jobs
        old = (datetime.now() - timedelta(days=80)).strftime("%Y-%m-%d")
        fresh = datetime.now().strftime("%Y-%m-%d")
        out = process_and_filter_jobs(self._df([old, fresh, ""]))
        assert len(out) == 2  # 80-day-old dropped; fresh + unknown kept

    def test_exactly_at_threshold_kept(self):
        from datetime import datetime, timedelta
        from job_agent.config import STALE_POSTING_DAYS
        from job_agent.utils import process_and_filter_jobs
        edge = (datetime.now() - timedelta(days=STALE_POSTING_DAYS)).strftime("%Y-%m-%d")
        assert len(process_and_filter_jobs(self._df([edge]))) == 1


class TestCrossSourceDedup:
    """One employer listed as 'Harbor Lights Media' on one site and 'Harbor
    Lights Media LLC' on another (same JD, same location modulo ', US') is one
    role. Corporate suffixes and country-suffix location variants must not
    defeat the dedup key."""

    def test_llc_suffix_and_us_location_variant_deduped(self):
        import pandas as pd
        from datetime import datetime
        from job_agent.utils import process_and_filter_jobs
        fresh = datetime.now().strftime("%Y-%m-%d")
        df = pd.DataFrame([
            {"title": "Event Manager", "company": "Harbor Lights Media",
             "location": "Seattle, WA, US", "link": "https://indeed.example/1",
             "description": "Event production role", "date_posted": fresh},
            {"title": "Event Manager", "company": "Harbor Lights Media LLC",
             "location": "Seattle, WA", "link": "https://zip.example/2",
             "description": "Event production role", "date_posted": fresh},
        ])
        assert len(process_and_filter_jobs(df)) == 1

    def test_different_companies_not_over_deduped(self):
        import pandas as pd
        from datetime import datetime
        from job_agent.utils import process_and_filter_jobs
        fresh = datetime.now().strftime("%Y-%m-%d")
        df = pd.DataFrame([
            {"title": "Producer", "company": "Company A",
             "location": "Remote", "link": "https://x.example/1",
             "description": "Video production", "date_posted": fresh},
            {"title": "Producer", "company": "Company B",
             "location": "Remote", "link": "https://x.example/2",
             "description": "Video production", "date_posted": fresh},
        ])
        assert len(process_and_filter_jobs(df)) == 2


class TestFuzzyCompanyDedup:
    """One job listed four times across Indeed/LinkedIn/Zip/BuiltIn under a
    brand name and a full legal name. Policy (post-adversarial-review): token-tuple company matching,
    whole-token brand-prefix needs location corroboration, suffix strip is
    trailing-anchored, LinkedIn wins location-compatibly and inherits fields."""

    def test_company_key_normalization(self):
        from job_agent.utils import company_key
        assert company_key("Bramble's") == "brambles"
        assert company_key("Harbor Lights Media LLC") == "harborlightsmedia"
        # Trailing-anchored: mid-name/leading suffix words survive
        assert company_key("Co-Star") == "costar"
        assert company_key("Company 3") == "company3"
        assert company_key("CO Architects") == "coarchitects"

    def test_companies_match_policy(self):
        from job_agent.utils import companies_match
        city = "Tacoma, WA"
        # The motivating pair: brand vs full legal name, same city
        assert companies_match("Brambles", "Bramble's Small-Batch Sauces & Spices",
                               "Tacoma, WA, US", city)
        # Exact after suffix strip — no location needed
        assert companies_match("Harbor Lights Media", "Harbor Lights Media LLC")
        # In-token prefix never matches (adversarial review pairs)
        assert not companies_match("Apple", "Applebee's", city, city)
        assert not companies_match("Square", "Squarespace", city, city)
        assert not companies_match("Meta", "Metacritic", city, city)
        assert not companies_match("Bramble's", "Bramblesford Transport Ltd", city, city)
        # Token-prefix relation across DIFFERENT cities: two employers
        assert not companies_match("Epic", "Epic Games", "Verona, WI", "Cary, NC")
        assert not companies_match("Delta", "Delta Dental", "Atlanta, GA", "Seattle, WA")
        # Same relation with corroborating (or blank) location: one employer
        assert companies_match("Epic Games", "Epic Games Store Team", "Cary, NC", "Cary, NC")
        assert companies_match("Brambles", "Bramble's Small-Batch Sauces & Spices", city, "")

    def _brambles_batch(self):
        return [
            {"title": "Post-Production Coordinator", "company": "Brambles",
             "location": "Tacoma, WA, US", "job_url": "https://indeed.example/1",
             "site": "indeed", "min_amount": 75000, "max_amount": 90000, "interval": "yearly"},
            {"title": "Post-Production Coordinator",
             "company": "Bramble's Small-Batch Sauces & Spices",
             "location": "Tacoma, WA", "job_url": "https://www.linkedin.com/jobs/view/1"},
            {"title": "Post-Production Coordinator", "company": "Bramble's",
             "location": "Tacoma, WA", "job_url": "https://zip.example/1", "site": "zip_recruiter"},
            {"title": "Post-Production Coordinator",
             "company": "Bramble's Small-Batch Sauces & Spices",
             "location": "", "job_url": "https://builtin.example/1", "site": "builtin"},
        ]

    def test_brand_family_collapses_linkedin_wins_and_inherits_comp(self):
        from job_agent.utils import dedup_scraped_jobs
        out = dedup_scraped_jobs(self._brambles_batch(), set(), {})
        assert len(out) == 1
        assert "linkedin.com" in out[0]["job_url"]
        # Field merge: the LinkedIn survivor inherits Indeed's structured comp
        assert out[0]["min_amount"] == 75000 and out[0]["max_amount"] == 90000

    def test_in_token_prefix_company_not_eaten_by_brand_group(self):
        from job_agent.utils import dedup_scraped_jobs
        batch = self._brambles_batch() + [
            {"title": "Post-Production Coordinator", "company": "Bramblesford Transport Ltd",
             "location": "Tacoma, WA", "job_url": "https://indeed.example/trucks", "site": "indeed"}]
        out = dedup_scraped_jobs(batch, set(), {})
        assert len(out) == 2

    def test_cross_city_brand_prefix_keeps_both_no_wrong_swap(self):
        """Delta (Atlanta, Indeed) + Delta Dental (Seattle, LinkedIn): two employers.
        The old draft collapsed them AND swapped in the wrong employer's link."""
        from job_agent.utils import dedup_scraped_jobs
        out = dedup_scraped_jobs([
            {"title": "Executive Assistant", "company": "Delta",
             "location": "Atlanta, GA", "job_url": "https://indeed.example/delta", "site": "indeed"},
            {"title": "Executive Assistant", "company": "Delta Dental",
             "location": "Seattle, WA", "job_url": "https://www.linkedin.com/jobs/view/9"},
        ], set(), {})
        assert len(out) == 2

    def test_replace_requires_location_compatibility(self):
        """A NY LinkedIn copy must not replace (and then location-kill) a
        Tacoma job — incompatible locations means two rows survive."""
        from job_agent.utils import dedup_scraped_jobs
        out = dedup_scraped_jobs([
            {"title": "Producer", "company": "Brambles",
             "location": "Tacoma, WA", "job_url": "https://indeed.example/1", "site": "indeed"},
            {"title": "Producer", "company": "Bramble's Small-Batch Sauces & Spices",
             "location": "New York, NY", "job_url": "https://www.linkedin.com/jobs/view/2"},
        ], set(), {})
        assert len(out) == 2

    def test_db_snapshot_fuzzy_skip_and_non_suppression(self):
        from job_agent.utils import company_tokens, dedup_scraped_jobs, location_key
        pairs = {"post-production coordinator": [(company_tokens("Brambles"), location_key("Tacoma, WA"))],
                 "project manager": [(company_tokens("Apple"), location_key("Austin, TX"))]}
        # The whole brand-family batch is the DB's job
        assert dedup_scraped_jobs(self._brambles_batch(), set(), pairs) == []
        # But an Apple row must NOT suppress Applebee's (permanent-suppression finding)
        out = dedup_scraped_jobs([
            {"title": "Project Manager", "company": "Applebee's",
             "location": "Austin, TX", "job_url": "https://indeed.example/applebees"}], set(), pairs)
        assert len(out) == 1

    def test_empty_company_rss_rows_not_collapsed(self):
        from job_agent.utils import dedup_scraped_jobs
        rows = [
            {"title": "Podcast Producer", "company": "", "job_url": "https://rss.example/a"},
            {"title": "Podcast Producer", "company": "", "job_url": "https://rss.example/b"},
        ]
        assert len(dedup_scraped_jobs(rows, set(), {})) == 2

    def test_known_link_skipped(self):
        from job_agent.utils import dedup_scraped_jobs
        rows = [{"title": "Producer", "company": "A", "job_url": "https://known.example/1"}]
        assert dedup_scraped_jobs(rows, {"https://known.example/1"}, {}) == []


# ── The fixture profile's metros ────────────────────────────────────────────
import pytest as _pytest  # noqa: E402

from tests.fixtures.persona import PRIMARY_METRO_TERMS, SECONDARY_METRO_TERMS  # noqa: E402

@_pytest.mark.parametrize("loc", [
    "Seattle, WA, US", "Bellevue, WA", "Redmond, WA", "Kirkland, WA", "Tacoma, WA", "Everett, WA",
    "Renton, WA, US", "Bothell, WA", "Puget Sound region", "Seattle, Washington, United States",
])
def test_primary_metro_places_are_primary(loc):
    from job_agent.config import classify_location, location_is_suppressible
    assert classify_location(loc) == "primary"
    assert location_is_suppressible(loc) is False


def test_primary_metro_table_covers_every_profile_term():
    from job_agent.config import PRIMARY_METRO_TERMS as CFG_PRIMARY, SECONDARY_METRO_TERMS as CFG_SECONDARY
    assert tuple(CFG_PRIMARY) == PRIMARY_METRO_TERMS
    assert tuple(CFG_SECONDARY) == SECONDARY_METRO_TERMS


@_pytest.mark.parametrize("loc", [
    "Dublin, Ireland", "Newark, NJ", "Brisbane, Australia", "Vancouver, BC, Canada", "Washington, DC",
    # Same-name towns in another state or country: a bare metro term does not
    # count where the location names a state outside the metro states.
    "Bellevue, NE", "Everett, MA", "Kirkland, QC, Canada", "Portland, ME",
    # A town the profile does not list is not a metro place, whatever its state.
    "Dublin, OH", "Saratoga, WY", "Alameda, NM", "Carson, NV", "Downey, ID",
])
def test_places_outside_the_profile_metros_are_neither_metro(loc):
    from job_agent.config import classify_location
    assert classify_location(loc) not in ("primary", "secondary")


@_pytest.mark.parametrize("loc,ok", [
    ("United States", False), ("Hybrid", False),
    ("Seattle, WA", False), ("Remote", False), ("Portland, OR", False), ("Beaverton, OR, US", False),
    ("Vancouver, WA, US", False), ("San Diego, CA", True), ("Austin, TX", True), ("Spokane, WA", True),
    # A bare home-state label must never be blockable: it would drop every
    # statewide posting in the candidate's own state.
    ("Washington", False), ("WA, US", False),
    # Another state's bare label is out of area like any other place there.
    ("California", True), ("CA, US", True),
])
def test_location_is_suppressible(loc, ok):
    from job_agent.config import location_is_suppressible
    assert location_is_suppressible(loc) is ok


@_pytest.mark.parametrize("loc", ["Portland, OR", "Beaverton, OR, US", "Hillsboro, OR", "Gresham, OR",
                                  "Vancouver, WA", "Portland, Oregon, United States"])
def test_secondary_metro_places_are_secondary_and_never_blockable(loc):
    from job_agent.config import classify_location, location_is_suppressible
    assert classify_location(loc) == "secondary"
    assert location_is_suppressible(loc) is False


@_pytest.mark.parametrize("loc", ["Salem, OR", "Eugene, OR", "Carson City, NV", "Vancouver, BC", "San Diego, CA"])
def test_other_places_are_not_secondary(loc):
    from job_agent.config import classify_location
    assert classify_location(loc) != "secondary"


# ── The home state and the metro terms come from the profile ────────────────
def test_home_state_comes_from_the_profile():
    from job_agent import config
    assert (config.HOME_STATE_CODE, config.HOME_STATE_NAME) == ("wa", "washington")
    assert config.classify_location("Spokane, WA") == "home_state"
    assert config.classify_location("Olympia, WA") == "home_state"
    assert config.classify_location("Sacramento, CA") == "reject"


def test_no_secondary_abbreviation_means_no_bare_abbreviation_match():
    """secondary_metro_abbrev is empty in the fixture: a bare 'PDX' is not the
    secondary metro."""
    from job_agent.config import SECONDARY_METRO_ABBREV, classify_location
    assert SECONDARY_METRO_ABBREV == ""
    assert classify_location("PDX") != "secondary"


def test_secondary_abbreviation_from_the_profile(monkeypatch):
    from job_agent import config
    monkeypatch.setattr(config, "SECONDARY_METRO_ABBREV", "pdx")
    assert config.classify_location("PDX") == "secondary"
    assert config.classify_location("PDX Airport, OR") == "reject"   # exact match only


@_pytest.mark.parametrize("term,expected", [
    ("vancouver, wa", ("vancouver", "wa")),
    ("vancouver, washington", ("vancouver", "wa")),
    ("dublin, oh", ("dublin", "oh")),
    ("seattle", ("seattle", None)),
    ("london, uk", ("london, uk", None)),     # not a US state: a plain term
])
def test_parse_metro_term(term, expected):
    from job_agent.config import _parse_metro_term
    assert _parse_metro_term(term) == expected


def test_a_term_with_its_state_counts_only_with_that_state(monkeypatch):
    """An ambiguous town is written in the profile as 'town, st'."""
    from job_agent import config
    monkeypatch.setattr(config, "_PRIMARY_PARSED",
                        config._PRIMARY_PARSED + [config._parse_metro_term("salem, or")])
    assert config.classify_location("Salem, OR") == "primary"
    assert config.classify_location("Salem, Oregon, United States") == "primary"
    assert config.classify_location("Salem, MA") == "reject"
    assert config.classify_location("Vancouver, Washington") == "secondary"   # fixture's 'vancouver, wa'


def test_a_bare_term_counts_in_a_state_named_in_allowed():
    """The secondary metro sits across a state line: 'oregon' / 'or' in
    search.locations.allowed lets bare 'portland' count there, nowhere else."""
    from job_agent.config import _METRO_STATES, classify_location
    assert _METRO_STATES == {"wa", "or"}
    assert classify_location("Portland, OR") == "secondary"
    assert classify_location("Portland, ME") == "reject"
    assert classify_location("Portland, Maine; Tacoma, WA") == "primary"   # another option still counts
