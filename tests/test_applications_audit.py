"""The Kanban's document panel reads a few audit fields from /api/applications
(why a CV or a letter is the general version). get_applications returns those
fields only, not the whole audit. Hermetic: a scratch database."""
import json


def _job(title="Video Producer"):
    return {"title": title, "company": "Harbor Lights Media", "location": "Seattle, WA",
            "job_url": f"https://example.com/jobs/{title.replace(' ', '-').lower()}",
            "description": "Produce video.", "site": "linkedin"}


def _set(db, job_id, audit, domain="producer_core"):
    conn = db.get_connection()
    conn.execute("update scrapes set cv_audit_summary = ?, cv_domain_used = ? where id = ?",
                 (audit, domain, job_id))
    conn.commit()


def test_the_panel_fields_come_back_and_nothing_else(tmp_db):
    job_id = tmp_db.save_job(_job())
    tmp_db.track_job(job_id)
    full = {"states": {"input": "input_incomplete", "artifact": "ok"}, "jd_chars": 0,
            "note": "thin posting (0 chars after cleaning) - general version",
            "cl": {"state": "input_incomplete", "template_used": True, "words": 310},
            "judge": {"claims": ["x" * 5000]}}
    _set(tmp_db, job_id, json.dumps(full))
    app = tmp_db.get_applications()[0]
    assert app["cv_domain_used"] == "producer_core"
    assert json.loads(app["cv_audit_summary"]) == {
        "states": {"input": "input_incomplete", "artifact": "ok"},
        "jd_chars": 0, "cl": {"state": "input_incomplete"}}


def test_an_old_or_missing_audit_does_not_break_the_board(tmp_db):
    a = tmp_db.save_job(_job("Producer"))
    b = tmp_db.save_job(_job("Segment Producer"))
    c = tmp_db.save_job(_job("Associate Producer"))
    for job_id in (a, b, c):
        tmp_db.track_job(job_id)
    _set(tmp_db, a, "Plain-text audit from an older version", domain=None)
    _set(tmp_db, b, json.dumps({"overall_pass": True}), domain=None)
    apps = {x["job_id"]: x for x in tmp_db.get_applications()}
    assert apps[a]["cv_audit_summary"] == "Plain-text audit from an older version"
    assert json.loads(apps[b]["cv_audit_summary"]) == {"states": None, "jd_chars": None,
                                                       "cl": {"state": None}}
    assert apps[c]["cv_audit_summary"] is None
