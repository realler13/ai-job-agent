"""dedup_baseline: every known link blocks a re-scrape, but only GRADED rows
(or rows added by hand) block a same-title copy from another source. A row
dropped before the AI (for example by a location rule) must not hide a later
copy of the same role that carries a better location."""
from job_agent.db_manager import DBManager
from job_agent.main import dedup_baseline
from job_agent.utils import dedup_scraped_jobs

BOARD = "https://careers.harborlights.example/jobs"


def _row(db, link, title, company, location, match_level=None):
    jid = db.save_job({"title": title, "company": company, "location": location,
                       "job_url": link, "description": "d"})
    if match_level:
        conn = db._get_thread_conn()
        conn.execute("UPDATE scrapes SET match_level = ? WHERE id = ?", (match_level, jid))
        conn.commit()
    return jid


def test_never_graded_rows_do_not_block_a_cross_source_copy(tmp_path):
    db = DBManager(db_path=str(tmp_path / "d.db"))
    try:
        co = "Harbor Lights Media"
        _row(db, f"{BOARD}/1", "Senior Video Producer", co, "United States of America")
        _row(db, "https://li/2", "Event Producer", co, "Seattle, WA", match_level="HIGH_MATCH")
        _row(db, "https://manual/3", "Field Producer", co, "", match_level="USER_INPUT")
        links, pairs = dedup_baseline(db)
        assert {f"{BOARD}/1", "https://li/2", "https://manual/3"} <= links
        assert "senior video producer" not in pairs              # never graded: no block
        assert "event producer" in pairs and "field producer" in pairs

        batch = [
            {"title": "Senior Video Producer", "company": co, "location": "Seattle, WA",
             "job_url": "https://indeed/9"},                      # copy of the ungraded row: kept
            {"title": "Event Producer", "company": co, "location": "Seattle, WA",
             "job_url": "https://indeed/8"},                      # copy of a graded row: dropped
            {"title": "Senior Video Producer", "company": co, "location": "x",
             "job_url": f"{BOARD}/1"},                            # same link: dropped
        ]
        kept = dedup_scraped_jobs(batch, links, pairs)
        assert [j["job_url"] for j in kept] == ["https://indeed/9"]
    finally:
        db.close()
