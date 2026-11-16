"""
ATS token health-check — the detection half of "auto-update scrapers".

Probes every ATS_COMPANIES token against its live API and classifies each as:
  ALIVE  — HTTP 200 with >=1 posting
  EMPTY  — HTTP 200 but 0 postings (board exists, nothing open right now)
  DEAD   — 404 / error / SmartRecruiters totalFound==0 (token likely wrong/gone)
  SKIP   — Workday/Teamtailor/Eightfold/Apple/Comeet (not cheaply probeable here)

Run:
  PYTHONPATH=src .venv/bin/python scripts/ats_health_check.py            # report to stdout
  PYTHONPATH=src .venv/bin/python scripts/ats_health_check.py --alert    # + email DEAD list

Exit code is non-zero when any token is DEAD, so this can gate a scheduled job.
Healing a DEAD token (rediscovery) is the next layer — see ats_autodiscovery.py
(resolve_careers_page via Perplexity -> discover_ats_from_careers_page).
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, "src")
from job_agent.config import ATS_COMPANIES  # noqa: E402

TIMEOUT = 12
# Accept-Encoding excludes 'br' on purpose: some ATS (Ashby) serve brotli, which
# requests only decodes when the brotli package + correct negotiation align. The
# live scrapers do the same (scraper_ashby_sr._SAFE_HEADERS) to stay robust.
HEADERS = {
    "User-Agent": "AIJobAgent/1.0 (+ATS health check)",
    "Accept-Encoding": "gzip, deflate",
}
# Families we can cheaply verify with a single GET.
PROBEABLE = {"greenhouse", "lever", "ashby", "smartrecruiters", "workable", "recruitee", "bamboohr"}


def _count(ats: str, token: str) -> tuple[str, str]:
    """Return (status, detail) for one token. status in ALIVE/EMPTY/DEAD/SKIP."""
    if ats not in PROBEABLE:
        return "SKIP", f"{ats} not cheaply probeable"
    try:
        if ats == "greenhouse":
            url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code != 200:
                return "DEAD", f"HTTP {r.status_code}"
            n = len(r.json().get("jobs", []))
        elif ats == "lever":
            r = requests.get(f"https://api.lever.co/v0/postings/{token}?mode=json", headers=HEADERS, timeout=TIMEOUT)
            if r.status_code != 200:
                return "DEAD", f"HTTP {r.status_code}"
            n = len(r.json())
        elif ats == "ashby":
            r = requests.get(f"https://api.ashbyhq.com/posting-api/job-board/{token}", headers=HEADERS, timeout=TIMEOUT)
            if r.status_code != 200:
                return "DEAD", f"HTTP {r.status_code}"
            n = len((r.json() or {}).get("jobs", []))
        elif ats == "smartrecruiters":
            r = requests.get(f"https://api.smartrecruiters.com/v1/companies/{token}/postings", headers=HEADERS, timeout=TIMEOUT)
            if r.status_code != 200:
                return "DEAD", f"HTTP {r.status_code}"
            n = int((r.json() or {}).get("totalFound", 0))  # SR returns 200 for anything
        elif ats == "workable":
            r = requests.get(f"https://apply.workable.com/api/v1/widget/accounts/{token}?details=true", headers=HEADERS, timeout=TIMEOUT)
            if r.status_code != 200:
                return "DEAD", f"HTTP {r.status_code}"
            n = len((r.json() or {}).get("jobs", []))
        elif ats == "recruitee":
            r = requests.get(f"https://{token}.recruitee.com/api/offers/", headers=HEADERS, timeout=TIMEOUT)
            if r.status_code != 200:
                return "DEAD", f"HTTP {r.status_code}"
            n = len((r.json() or {}).get("offers", []))
        elif ats == "bamboohr":
            r = requests.get(f"https://{token}.bamboohr.com/careers/list", headers=HEADERS, timeout=TIMEOUT)
            if r.status_code != 200:
                return "DEAD", f"HTTP {r.status_code}"
            n = len((r.json() or {}).get("result", []))
        else:
            return "SKIP", ats
    except Exception as e:
        return "DEAD", f"{type(e).__name__}: {str(e)[:60]}"
    return ("ALIVE" if n > 0 else "EMPTY"), f"{n} jobs"


def main() -> int:
    rows = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_count, c["ats"], c["token"]): c for c in ATS_COMPANIES}
        for fut in as_completed(futs):
            c = futs[fut]
            status, detail = fut.result()
            rows.append((status, c["company"], c["ats"], c["token"], detail))

    buckets = {k: [] for k in ("ALIVE", "EMPTY", "DEAD", "SKIP")}
    for row in rows:
        buckets[row[0]].append(row)

    print(f"\n=== ATS health check: {len(ATS_COMPANIES)} tokens ===")
    for k in ("ALIVE", "EMPTY", "DEAD", "SKIP"):
        print(f"  {k:6}: {len(buckets[k])}")

    if buckets["DEAD"]:
        print("\n--- DEAD (fix or remove these tokens) ---")
        for _s, company, ats, token, detail in sorted(buckets["DEAD"], key=lambda r: r[1].lower()):
            print(f"  {company}  [{ats}:{token}]  -> {detail}")

    if buckets["EMPTY"]:
        print(f"\n--- EMPTY (live board, 0 open roles now) : {len(buckets['EMPTY'])} ---")
        print("  " + ", ".join(sorted(r[1] for r in buckets["EMPTY"])))

    dead_lines = [f"{c}  [{a}:{t}]  -> {d}" for _s, c, a, t, d in sorted(buckets["DEAD"], key=lambda r: r[1].lower())]

    if "--alert" in sys.argv and dead_lines:
        try:
            from job_agent.postprocess import send_error_alert  # type: ignore
            send_error_alert(
                f"ATS health check: {len(dead_lines)} dead tokens",
                "These ATS tokens failed their live probe and need fixing/removal:\n\n"
                + "\n".join(dead_lines),
                pipeline_stage="ATS health check",
            )
            print(f"\nEmailed dead-token alert ({len(dead_lines)} tokens).")
        except Exception as e:
            print(f"\nAlert email failed: {e}")

    return 1 if buckets["DEAD"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
