"""REPLAY the Stage 3 judge on its own real cases before changing its model.

Stage 3 judges a handful of borderline postings per run, so a model swap can
not be A/B-tested live. LangFuse keeps every judge call (the exact prompt and
the answer). This tool takes the most recent real prompts that were built from
TODAY's prompt template and sends each one, unchanged, to every model under
test through the production code path (CascadeFilter._stage3_single ->
call_llm: the same request parameters, JSON parsing and verdict mapping).
A re-run of the current judge is the control: verdicts vary from run to run,
so the candidate model is compared with a same-day replay of the old model, not
only with the stored answers. ("Candidate" in this file means the model under
test; the person whose search it is, is "the job seeker".)

  export   LangFuse -> results/stage3_replay/cases.json (newest first, spread
           over the four verdicts; only prompts built from today's template)
  run      cases x arms -> results/stage3_replay/run_<stamp>.json. Real spend,
           capped by --budget and counted on the usage the API reports. An arm
           named "model#2" is a second sample of the same model; --only 3,9
           re-runs just those cases (is a contested verdict stable?).
  report   agreement, shifts, what the job seeker did with each posting
           (applied / dismissed, read from the local database), cost and time
           per arm, and the gate below.

The gate (the first arm is the control, the second the candidate):
  G1  the candidate answered every case with a verdict the parser read
  G2  candidate-vs-stored agreement is within max(2, 10%) cases of the control's
  G3  of the cases the job seeker applied to, the candidate keeps visible at
      least as many as the control
  G4  the mean tier shift against the stored verdicts is within 0.35 of the
      control's (tiers: reject 0, worth a look 1, potential 2, high 3)
A pass means "no gross change"; read the differing cases before deciding.

Private: the prompts contain the job seeker's profile, so everything is written
under results/ (gitignored). Only ids, verdicts and counts go to stdout.

Run from the LIVE checkout directory (its env files hold the keys). The code
under test is the checkout this script lives in:
  .venv/bin/python <checkout>/scripts/replay_stage3_models.py export --n 24
  .venv/bin/python <checkout>/scripts/replay_stage3_models.py run --arms claude-fable-5-1,claude-fable-5-2 --budget 10
  .venv/bin/python <checkout>/scripts/replay_stage3_models.py report
"""
import argparse
import json
import math
import os
import re
import sqlite3
import string
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# The replay must never be traced: its calls would come back as "real" judge
# cases in the next export.
os.environ["JOB_AGENT_DISABLE_LANGFUSE"] = "1"

from dotenv import find_dotenv, load_dotenv  # noqa: E402

load_dotenv(find_dotenv(usecwd=True))                                   # API keys
load_dotenv(Path.cwd() / "ops" / "langfuse" / "langfuse-keys.conf")     # LangFuse keys

from job_agent import cascade_filter as cfm  # noqa: E402
from job_agent import config as cfg  # noqa: E402

OBS_NAME = "stage3-judge-llm"
TIER = {"IRRELEVANT": 0, "WORTH_A_LOOK": 1, "POTENTIAL_MATCH": 2, "HIGH_MATCH": 3}
SHORT = {"IRRELEVANT": "REJECT", "WORTH_A_LOOK": "WORTH", "POTENTIAL_MATCH": "POTENTIAL",
         "HIGH_MATCH": "HIGH"}
_LISTING_RE = re.compile(r"<job_listing>\nTitle: (.*)\nCompany: (.*)\nLocation: (.*)\n")


# ── the production judge, fed a ready-made prompt ─────────────────────────

class _Capture:
    """Stands in for the tracer: keeps what the judge would have logged."""

    def __init__(self):
        self.kw = None

    def log_stage3(self, row, **kw):
        self.kw = kw


def judge(prompt: str, model: str, row: dict, stored_answer: str = None) -> dict:
    """One Stage 3 verdict through the production code. With stored_answer the
    model is not called: the stored text goes through the same parser."""
    cf = cfm.CascadeFilter(profile_text="", stage3_model=model, stage1_fallbacks=[],
                           stage2_fallbacks=[], stage3_fallbacks=[], db=None)
    cf.stage3_prompt = prompt.replace("{", "{{").replace("}", "}}")   # .format() returns it as is
    cap = _Capture()
    cf._tracer = cap
    if stored_answer is not None:
        cf._call_llm_with_fallback = lambda *a, **k: (stored_answer, model)
    started = time.time()
    res = cf._stage3_single(dict(row))
    out = {
        "match_level": res["match_level"],
        "apply_priority": res.get("apply_priority"),
        "reason": res.get("reason"),
        "seconds": round(time.time() - started, 1),
        "failed": cap.kw is None,                       # every model failed: no answer at all
        "parse_error": str(res.get("reason", "")).startswith("Stage 3 parse error"),
    }
    if cap.kw:
        usage = cap.kw.get("usage") or {}
        out.update(model_used=cap.kw.get("model"), completion=cap.kw.get("completion"),
                   in_tokens=int(usage.get("input_tokens") or 0),
                   out_tokens=int(usage.get("output_tokens") or 0),
                   cost=float(cap.kw.get("cost") or 0.0))
    return out


# ── export ────────────────────────────────────────────────────────────────

def _literals(template: str) -> list:
    """The fixed text between a template's fields, in order."""
    return [lit for lit, _f, _s, _c in string.Formatter().parse(template) if lit]


def built_from(template: str, prompt: str) -> bool:
    pos = 0
    for lit in _literals(template):
        i = prompt.find(lit, pos)
        if i < 0:
            return False
        pos = i + len(lit)
    return True


def _langfuse_pages(page_size: int = 50):
    import requests
    if not (cfg.LANGFUSE_PUBLIC_KEY and cfg.LANGFUSE_SECRET_KEY):
        sys.exit("LangFuse keys not found: run from the live checkout directory")
    url = cfg.LANGFUSE_HOST.rstrip("/") + "/api/public/v2/observations"
    params = {"name": OBS_NAME, "type": "GENERATION", "limit": page_size,
              "fields": "core,basic,io,model,usage,metrics"}   # v4 hides payloads without fields=
    while True:
        r = requests.get(url, params=params, timeout=60,
                         auth=(cfg.LANGFUSE_PUBLIC_KEY, cfg.LANGFUSE_SECRET_KEY))
        r.raise_for_status()
        body = r.json()
        rows = body.get("data") or []
        if not rows:
            return
        yield rows
        cursor = (body.get("meta") or {}).get("cursor")
        if not cursor:
            return
        params["cursor"] = cursor


def cmd_export(args) -> int:
    template = cfg.get_stage3_prompt(cfg.CASCADE_STAGE3_MODEL)
    per_class = math.ceil(args.n / 4) + 2
    population, scanned, other_template, other_model, stale_streak = [], 0, 0, 0, 0
    for rows in _langfuse_pages():
        for o in rows:
            scanned += 1
            prompt, answer = o.get("input"), o.get("output")
            model = o.get("model") or ""
            if not isinstance(prompt, str) or not isinstance(answer, str):
                continue
            if not built_from(template, prompt):
                other_template += 1
                stale_streak += 1
                continue
            stale_streak = 0
            if not model.startswith(args.stored_family):
                other_model += 1          # answered by a fallback: not the judge's own verdict
                continue
            m = _LISTING_RE.search(prompt)
            row = {"title": m.group(1) if m else "", "company": m.group(2) if m else "",
                   "location": m.group(3) if m else ""}
            stored = judge(prompt, model, row, stored_answer=answer)
            population.append({
                "id": o.get("id"), "trace_id": o.get("traceId"), "time": o.get("startTime"),
                "stored_model": model, "row": row, "prompt": prompt, "stored_answer": answer,
                "stored": {k: stored[k] for k in ("match_level", "apply_priority", "reason", "parse_error")},
            })
        if stale_streak >= args.stop_after_stale or len(population) >= args.scan_cap:
            break
    population.sort(key=lambda c: c["time"] or "", reverse=True)
    picked, counts = [], {}
    for c in population:                                  # newest first, at most per_class each
        lvl = c["stored"]["match_level"]
        if len(picked) < args.n and counts.get(lvl, 0) < per_class:
            picked.append(c)
            counts[lvl] = counts.get(lvl, 0) + 1
    chosen = {c["id"] for c in picked}
    for c in population:                                  # then fill up with the newest left over
        if len(picked) >= args.n:
            break
        if c["id"] not in chosen:
            picked.append(c)
            chosen.add(c["id"])
    picked.sort(key=lambda c: c["time"] or "", reverse=True)
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(
        {"exported_at": datetime.now().isoformat(timespec="seconds"), "template_model": cfg.CASCADE_STAGE3_MODEL,
         "cases": picked}, indent=1))

    def spread(cases):
        d = {}
        for c in cases:
            d[SHORT.get(c["stored"]["match_level"], "?")] = d.get(SHORT.get(c["stored"]["match_level"], "?"), 0) + 1
        return dict(sorted(d.items()))
    print(f"scanned {scanned} judge calls: {other_template} from an older prompt template, "
          f"{other_model} answered by a fallback model, {len(population)} usable")
    if population:
        print(f"usable range {population[-1]['time'][:10]} .. {population[0]['time'][:10]}; verdicts {spread(population)}")
    print(f"picked {len(picked)} -> {out / 'cases.json'}; verdicts {spread(picked)}; "
          f"stored parse errors {sum(1 for c in picked if c['stored']['parse_error'])}")
    return 0


# ── run ───────────────────────────────────────────────────────────────────

def cmd_run(args) -> int:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a.split("#")[0] not in cfm.MODEL_PRICING:
            sys.exit(f"no price for {a}: the budget cap could not see its spend")
    data = json.loads((Path(args.dir) / "cases.json").read_text())
    cases = data["cases"][: args.limit] if args.limit else data["cases"]
    if args.only:   # 1-based numbers as the report prints them
        wanted = {int(x) for x in args.only.split(",")}
        cases = [c for i, c in enumerate(data["cases"], 1) if i in wanted]
    lock = threading.Lock()
    state = {"spent": 0.0, "reserved": 0.0, "skipped": 0}
    results = {c["id"]: {} for c in cases}

    def one(case, arm):
        model = arm.split("#")[0]     # "model#2" = a second sample of the same model
        worst = cfm._cost_for_model(model, len(case["prompt"]) // 3, 8000)
        with lock:
            if state["spent"] + state["reserved"] + worst > args.budget:
                state["skipped"] += 1
                return
            state["reserved"] += worst
        try:
            res = judge(case["prompt"], model, case["row"])
        except Exception as e:   # never lose the paid answers of the other tasks
            res = {"match_level": "POTENTIAL_MATCH", "failed": True, "parse_error": False,
                   "reason": f"replay error: {e}", "seconds": 0.0}
        with lock:
            state["reserved"] -= worst
            state["spent"] += res.get("cost", 0.0)
            results[case["id"]][arm] = res
            done = sum(len(v) for v in results.values())
            print(f"  [{done}/{len(cases) * len(arms)}] {arm}: {SHORT.get(res['match_level'], '?')}"
                  f"{' FAILED' if res.get('failed') else ''}{' PARSE-ERROR' if res.get('parse_error') else ''}"
                  f" {res.get('seconds')}s  spent ${state['spent']:.2f}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(args.dir) / f"run_{stamp}.json"
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for case in cases:
                for arm in arms:
                    pool.submit(one, case, arm)
    finally:
        path.write_text(json.dumps({"arms": arms, "budget": args.budget, "spent": state["spent"],
                                    "skipped_for_budget": state["skipped"],
                                    "cases_file": str(Path(args.dir) / "cases.json"),
                                    "results": results}, indent=1))
    print(f"spent ${state['spent']:.2f} of ${args.budget:.2f}; {state['skipped']} call(s) skipped for budget -> {path}")
    return 0


# ── report ────────────────────────────────────────────────────────────────

def _candidate_outcomes(db_path: str, cases: list) -> dict:
    """case id -> what the job seeker did with that posting (read-only)."""
    titles = sorted({c["row"]["title"].lower() for c in cases if c["row"]["title"]})
    if not titles or not Path(db_path).exists():
        return {}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    marks = ",".join("?" * len(titles))
    rows = con.execute(
        f"select s.id, lower(s.title), lower(s.company), s.created_at, s.reviewed, s.rejection_reason, a.status "
        f"from scrapes s left join applications a on a.job_id = s.id where lower(s.title) in ({marks})", titles).fetchall()
    con.close()
    out = {}
    for c in cases:
        t, co = c["row"]["title"].lower(), c["row"]["company"].lower()
        mine = [r for r in rows if r[1] == t and (r[2] or "") == co]
        if not mine:
            continue
        applied = next((r for r in mine if r[6]), None)
        dismissed = next((r for r in mine if r[5]), None)
        out[c["id"]] = {"job_ids": [r[0] for r in mine],
                        "applied": applied[6] if applied else None,
                        "dismissed": bool(dismissed) and not applied}
    return out


def cmd_report(args) -> int:
    d = Path(args.dir)
    run_path = Path(args.run) if args.run else sorted(d.glob("run_*.json"))[-1]
    run = json.loads(run_path.read_text())
    cases = json.loads((d / "cases.json").read_text())["cases"]
    arms, results = run["arms"], run["results"]
    cases = [c for c in cases if c["id"] in results and all(a in results[c["id"]] for a in arms)]
    outcomes = _candidate_outcomes(args.db, cases)
    print(f"{run_path.name}: {len(cases)} cases x {arms}; spent ${run['spent']:.2f}")
    print(f"{'#':>2} {'date':10} {'stored':9} " + " ".join(f"{a.replace('claude-', ''):>14}" for a in arms) + "  job seeker's outcome")
    for i, c in enumerate(cases, 1):
        cells = []
        for a in arms:
            r = results[c["id"]][a]
            tag = "FAILED" if r.get("failed") else "PARSE-ERR" if r.get("parse_error") else SHORT.get(r["match_level"], "?")
            cells.append(f"{tag + ' p' + str(r.get('apply_priority')):>14}")
        o = outcomes.get(c["id"]) or {}
        mark = f"applied ({o['applied']})" if o.get("applied") else "dismissed" if o.get("dismissed") else ""
        title = f"  {c['row']['title'][:44]} @ {c['row']['company'][:22]}" if args.show_titles else ""
        print(f"{i:>2} {str(c['time'])[:10]} {SHORT.get(c['stored']['match_level'], '?') + ' p' + str(c['stored']['apply_priority']):9} "
              + " ".join(cells) + f"  {mark}{title}")

    def level(c, a):
        return c["stored"]["match_level"] if a == "stored" else results[c["id"]][a]["match_level"]

    def ok(c, a):
        r = results[c["id"]][a]
        return not r.get("failed") and not r.get("parse_error")

    print("\nper arm")
    stats = {}
    for a in arms:
        rs = [results[c["id"]][a] for c in cases]
        good = [c for c in cases if ok(c, a)]
        agree = sum(1 for c in good if level(c, a) == level(c, "stored"))
        visible = sum(1 for c in good if (level(c, a) == "IRRELEVANT") == (level(c, "stored") == "IRRELEVANT"))
        shift = sum(TIER[level(c, a)] - TIER[level(c, "stored")] for c in good) / max(1, len(good))
        spread = {}
        for c in good:
            spread[SHORT[level(c, a)]] = spread.get(SHORT[level(c, a)], 0) + 1
        stats[a] = {"good": len(good), "agree": agree, "shift": shift}
        print(f"  {a}: answered {len(good)}/{len(cases)} (failed {sum(1 for r in rs if r.get('failed'))}, "
              f"parse errors {sum(1 for r in rs if r.get('parse_error'))}) | same verdict as stored {agree}/{len(good)} | "
              f"same visible/rejected {visible}/{len(good)} | mean tier shift {shift:+.2f} | {dict(sorted(spread.items()))}")
        print(f"      mean {sum(r.get('seconds', 0) for r in rs) / max(1, len(rs)):.0f}s, "
              f"{sum(r.get('in_tokens', 0) for r in rs):,} in / {sum(r.get('out_tokens', 0) for r in rs):,} out tokens, "
              f"${sum(r.get('cost', 0) for r in rs):.2f} total, ${sum(r.get('cost', 0) for r in rs) / max(1, len(rs)):.3f} per call")
    stored_spread = {}
    for c in cases:
        stored_spread[SHORT[level(c, "stored")]] = stored_spread.get(SHORT[level(c, "stored")], 0) + 1
    print(f"  stored: {dict(sorted(stored_spread.items()))}")
    if len(arms) >= 2:
        ctrl, cand = arms[0], arms[1]
        both = [c for c in cases if ok(c, ctrl) and ok(c, cand)]
        print(f"\n{cand} vs {ctrl} today: same verdict {sum(1 for c in both if level(c, ctrl) == level(c, cand))}/{len(both)}")
        diff = [(i, c) for i, c in enumerate(cases, 1) if c in both and level(c, ctrl) != level(c, cand)]
        print("  differing cases (#, control -> candidate, stored): "
              + (", ".join(f"#{i} {SHORT[level(c, ctrl)]}->{SHORT[level(c, cand)]} (stored {SHORT[level(c, 'stored')]})" for i, c in diff) or "none"))
        applied = [c for c in cases if (outcomes.get(c["id"]) or {}).get("applied")]
        dismissed = [c for c in cases if (outcomes.get(c["id"]) or {}).get("dismissed")]

        def kept(group, a):
            return sum(1 for c in group if ok(c, a) and level(c, a) != "IRRELEVANT")
        print(f"\njob seeker's outcomes: {len(applied)} case(s) applied to, {len(dismissed)} dismissed")
        for name, group in (("applied", applied), ("dismissed", dismissed)):
            if group:
                print(f"  {name}: kept visible by stored {sum(1 for c in group if level(c, 'stored') != 'IRRELEVANT')}/{len(group)}, "
                      f"{ctrl} {kept(group, ctrl)}/{len(group)}, {cand} {kept(group, cand)}/{len(group)}")
        n = len(cases)
        slack = max(2, round(0.1 * n))
        gates = [
            ("G1 candidate answered every case", stats[cand]["good"] == n),
            (f"G2 agreement with stored within {slack} of the control "
             f"({stats[cand]['agree']} vs {stats[ctrl]['agree']})", stats[cand]["agree"] >= stats[ctrl]["agree"] - slack),
            ("G3 applied cases kept visible >= control", kept(applied, cand) >= kept(applied, ctrl)),
            (f"G4 tier shift within 0.35 of the control ({stats[cand]['shift']:+.2f} vs {stats[ctrl]['shift']:+.2f})",
             abs(stats[cand]["shift"] - stats[ctrl]["shift"]) <= 0.35),
        ]
        print("\ngate")
        for name, passed in gates:
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        return 0 if all(p for _n, p in gates) else 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=str(Path.cwd() / "results" / "stage3_replay"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--n", type=int, default=24)
    e.add_argument("--stored-family", default="claude-fable", help="keep cases the judge itself answered")
    e.add_argument("--stop-after-stale", type=int, default=40,
                   help="stop after this many consecutive calls built from an older template")
    e.add_argument("--scan-cap", type=int, default=400)
    r = sub.add_parser("run")
    r.add_argument("--arms", default="claude-fable-5-1,claude-fable-5-2", help="control first, candidate second")
    r.add_argument("--budget", type=float, default=10.0, help="hard cap on real spend (USD)")
    r.add_argument("--limit", type=int, default=0, help="only the first N cases (smoke test)")
    r.add_argument("--only", default="", help="only these case numbers, e.g. 3,9,11 (a second look at contested cases)")
    r.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("report")
    p.add_argument("--run", default=None, help="a run_*.json (default: the newest)")
    p.add_argument("--db", default=str(Path.cwd() / "job_search_data.db"))
    p.add_argument("--show-titles", action="store_true", help="print titles and companies (private)")
    args = ap.parse_args(argv)
    return {"export": cmd_export, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
