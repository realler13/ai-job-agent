"""SerpAPI Google Jobs scraper: mapping, budget guard, rotation."""

import pytest

from job_agent import scraper_google_jobs as gj


@pytest.fixture
def db(tmp_path):
    from job_agent.db_manager import DBManager
    return DBManager(db_path=str(tmp_path / "gj.db"))


SAMPLE_JOB = {
    "title": "Video Producer",
    "company_name": "Harbor Lights Media",
    "location": "Denver, CO",
    "via": "iCIMS",
    "share_link": "https://www.google.com/search?q=...",
    "description": "Produce videos.",
    "job_highlights": [{"title": "Qualifications", "items": ["5 years exp"]}],
    "detected_extensions": {"posted_at": "3 days ago", "schedule_type": "Full-time"},
    "apply_options": [
        {"title": "LinkedIn", "link": "https://linkedin.com/jobs/view/1"},
        {"title": "Harbor Lights Media Careers", "link": "https://careers.harborlights.com/job/1"},
    ],
}


class TestMapping:
    def test_maps_core_fields(self):
        j = gj._map_job(SAMPLE_JOB)
        assert j["title"] == "Video Producer"
        assert j["company"] == "Harbor Lights Media"
        assert j["site"] == "google_jobs"
        assert j["via"] == "iCIMS"
        assert "Qualifications: 5 years exp" in j["description"]

    def test_prefers_employer_ats_apply_link(self):
        # employer careers domain beats LinkedIn (maximizes dedup matches)
        j = gj._map_job(SAMPLE_JOB)
        assert j["job_url"] == "https://careers.harborlights.com/job/1"

    def test_anywhere_becomes_remote(self):
        job = {**SAMPLE_JOB, "location": "Anywhere"}
        assert gj._map_job(job)["location"] == "Remote"

    def test_no_employer_link_drops_row(self):
        # aggregator-only postings are not actionable — no share_link fallback
        job = {**SAMPLE_JOB, "apply_options": [
            {"title": "WhatJobs", "link": "https://www.whatjobs.com/jobs/x?id=1"},
            {"title": "JobLeads", "link": "https://www.jobleads.com/us/job/y"},
        ]}
        assert gj._map_job(job) is None

    def test_empty_apply_options_drops_row(self):
        job = {**SAMPLE_JOB, "apply_options": []}
        assert gj._map_job(job) is None


class TestEmployerGrade:
    """Employer-grade link policy: an apply link counts only when it is the
    employer's own application surface (its ATS or its own domain), never a
    paywalled, redirecting or aggregator board. Both failure directions encoded."""

    # -- true accepts --
    def test_known_ats_hosts(self):
        assert gj.is_employer_grade("https://jobs.ashbyhq.com/pinterest/x", "Pinterest")
        assert gj.is_employer_grade("https://ngs.wd1.myworkdayjobs.com/x", "National Geographic Society")
        assert gj.is_employer_grade("https://apply.workable.com/tidepoolradio/j/1", "Tidepool Radio")
        assert gj.is_employer_grade("https://app.dover.com/apply/x", "Bluefin Learning")
        # unreachable-native ATS: the source's reason to exist
        assert gj.is_employer_grade("https://careers-harborlightsmedia.icims.com/jobs/1", "Harbor Lights Media")

    def test_company_domain_match(self):
        assert gj.is_employer_grade("https://careers.google.com/jobs/x", "Google")
        assert gj.is_employer_grade("https://jobs.ebayinc.com/job/1", "eBay")  # ebay+inc residual
        assert gj.is_employer_grade("https://www.aidoos.com/job/1", "AiDOOS")
        assert gj.is_employer_grade("https://careers.replika.com/x", "Replika")

    # -- true rejects (paywalled, redirecting or aggregator boards) --
    def test_paywall_and_redirect_chum_rejected(self):
        assert not gj.is_employer_grade(
            "https://www.hollylist.com/job/podcast-producer-x", "Premium Employer")
        assert not gj.is_employer_grade(
            "https://www.whatjobs.com/jobs/production-coordinator/remote?id=1", "Saltmarsh Talent")
        assert not gj.is_employer_grade("https://www.learn4good.com/jobs/x/e/", "Indie Media")
        assert not gj.is_employer_grade("https://www.mediabistro.com/jobs/1", "Gullwing Studios")

    def test_apple_contingent_marketplace_is_not_employer_grade(self):
        # 2026-10-03: removed from the trusted hosts — the contingent
        # marketplace is not the employer's own application surface.
        assert not gj.is_employer_grade(
            "https://directsource.magnitglobal.com/x", "Apple Contingent Workforce")
        assert not gj.is_employer_grade(
            "https://applecontingentworkforce.willhire.co/jobs/1", "Apple Contingent Workforce")

    def test_jobs_subdomain_alone_proves_nothing(self):
        # football-club site serving staffing-agency chum via a jobs. prefix
        assert not gj.is_employer_grade(
            "https://jobs.stevenagefc.com/job/1", "Ridgeway Resourcing LLC")

    def test_subset_trap_anchored_matching(self):
        # 'capital' ⊂ 'venturecapitalcareers' must NOT match (board, not employer)
        assert not gj.is_employer_grade(
            "https://venturecapitalcareers.com/job/1", "Northwind Capital")

    def test_name_collision_cannot_certify_known_board(self):
        # adversarial review 2026-07-05: token collisions self-certified boards
        assert not gj.is_employer_grade(
            "https://www.monster.com/job-openings/x", "Monster Energy")
        assert not gj.is_employer_grade(
            "https://en-us.whatjobs.com/job/12345", "What If Media Group")
        assert not gj.is_employer_grade(
            "https://www.talent.com/view?id=1", "24 Seven Talent")
        assert not gj.is_employer_grade(
            "https://lensa.com/video-producer-jobs/x", "Lens Technology")
        assert not gj.is_employer_grade(
            "https://www.care.com/p/jobs/123", "HealthCare Partners")
        assert not gj.is_employer_grade(
            "https://www.tealhq.com/job/x", "Teal")

    def test_every_built_in_metro_site_is_a_board(self):
        # Built In has one site per metro; none of them is an employer surface,
        # whatever the company is called.
        for url in ("https://www.builtinseattle.com/job/producer/1",
                    "https://www.builtinnyc.com/job/producer/2",
                    "https://builtin.com/job/producer/3"):
            assert not gj.is_employer_grade(url, "Built In"), url
            assert not gj.is_employer_grade(url, "Seattle Builtin Studios"), url

    def test_generic_market_tokens_prove_nothing(self):
        assert not gj.is_employer_grade("https://getwork.com/details/abc", "Work & Co")
        assert not gj.is_employer_grade("https://remotejobs.org/job/x", "Remote Technology, Inc.")
        assert not gj.is_employer_grade("https://www.localjobs.com/job/x", "Local Media Co")

    def test_token_concatenation_domains_match(self):
        # single-token anchoring rejected all of these (adversarial review)
        assert gj.is_employer_grade("https://www.epicgames.com/careers/en-US/jobs/1", "Epic Games")
        assert gj.is_employer_grade("https://careers.riotgames.com/job/1", "Riot Games")
        assert gj.is_employer_grade("https://www.sonymusic.com/careers/job/1", "Sony Music")
        assert gj.is_employer_grade("https://www.coca-cola.com/careers/job/1", "The Coca-Cola Company")

    def test_ats_browse_and_marketing_pages_rejected(self):
        assert not gj.is_employer_grade("https://jobs.workable.com/search?query=producer", "Acme")
        assert not gj.is_employer_grade("https://www.workable.com/jobs?query=producer", "Acme")
        assert not gj.is_employer_grade("https://www.greenhouse.io/blog/hiring-trends", "Acme")
        # deep posting pages still pass
        assert gj.is_employer_grade("https://jobs.workable.com/view/abc/producer", "Larkspur Legal")
        assert gj.is_employer_grade("https://www.governmentjobs.com/careers/juniperridge/jobs/1", "County of Juniper Ridge")

    def test_native_boards_dont_count(self):
        # our native scrapers cover these with better data — google_jobs
        # duplicates of them are pure noise
        assert not gj.is_employer_grade("https://www.linkedin.com/jobs/view/1", "The Bramble Group")
        assert not gj.is_employer_grade("https://www.indeed.com/viewjob?jk=1", "Academy")
        assert not gj.is_employer_grade("https://www.ziprecruiter.com/c/x/job/1", "Gotham Enterprises Ltd")

    # -- junk-row guards --
    def test_junk_company_dropped(self):
        job = {**SAMPLE_JOB, "company_name": "Premium Employer"}
        assert gj._map_job(job) is None

    def test_junk_company_variants_dropped_even_on_ats_host(self):
        # confidential postings on a real ATS are still not researchable/tailorable
        for name in ("Confidential Company", "A Confidential Company",
                     "Company Confidential", "Undisclosed", "Private"):
            job = {**SAMPLE_JOB, "company_name": name, "apply_options": [
                {"title": "Apply", "link": "https://acme.applytojob.com/apply/x"}]}
            assert gj._map_job(job) is None, name

    def test_real_companies_with_scary_names_kept(self):
        # 'Anonymous Content' and 'Private Division' are real production companies
        assert not gj._is_junk_company("Anonymous Content")
        assert not gj._is_junk_company("Private Division")

    def test_listing_page_title_dropped(self):
        job = {**SAMPLE_JOB, "title": "Contract Podcast Producer Jobs in Springfield"}
        assert gj._map_job(job) is None

    def test_seo_title_shapes_dropped(self):
        for t in ("Video Producer Jobs Near Me",
                  "20 Best video producer jobs (Hiring Now!)",
                  "Hiring Now: Video Producer",
                  "Producer Jobs at Netflix",
                  "Video Producer Jobs"):
            assert gj._is_listing_page_title(t), t

    def test_normal_title_not_flagged_as_listing(self):
        assert not gj._is_listing_page_title("Video Producer")
        assert not gj._is_listing_page_title("Producer, Jobs-to-be-Done Research")
        assert not gj._is_listing_page_title("Production Coordinator - Steve Jobs Theater")


class TestBudget:
    def test_budget_creates_and_tracks(self, db):
        month, used, idx = gj._budget_state(db)
        assert used == 0 and idx == 0
        gj._spend(db, month, 2, 2)
        _, used2, idx2 = gj._budget_state(db)
        assert used2 == 2 and idx2 == 2

    def test_hard_stop_skips(self, db, monkeypatch):
        month, _, _ = gj._budget_state(db)
        gj._spend(db, month, gj.MONTHLY_HARD_STOP, 0)
        monkeypatch.setattr(gj, "SERPAPI_KEY", "k")
        called = []
        monkeypatch.setattr(gj, "search_google_jobs", lambda *a, **k: called.append(1) or [])
        assert gj.scrape_google_jobs(db) == []
        assert called == []  # no API spend past the stop

    def test_rotation_advances(self, db, monkeypatch):
        monkeypatch.setattr(gj, "SERPAPI_KEY", "k")
        queries = []
        monkeypatch.setattr(gj, "search_google_jobs",
                            lambda q, **kw: queries.append(q) or [])
        gj.scrape_google_jobs(db)
        gj.scrape_google_jobs(db)
        assert len(queries) == 4
        assert len(set(queries)) == 4  # four distinct rotating keywords

    def test_no_key_no_op(self, db, monkeypatch):
        monkeypatch.setattr(gj, "SERPAPI_KEY", "")
        monkeypatch.setattr(gj.os, "getenv", lambda *a, **k: "")
        assert gj.scrape_google_jobs(db) == []
