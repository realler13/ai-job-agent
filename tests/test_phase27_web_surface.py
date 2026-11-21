"""Phase 27 review fixes on the web surface (Fixer U).

The TS logic tests run the real page code: the helper block is cut out of the
.tsx file, transpiled with the repo's own TypeScript (web/node_modules) and
run in node — no network, no build. They skip when node or TypeScript is not
installed. Text / markup checks scan the source like test_hiring_health does.
"""
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

from job_agent.comp_utils import contract_pay_note

ROOT = Path(__file__).resolve().parents[1]
WEB_SRC = ROOT / "web" / "src"
TS_LIB = ROOT / "web" / "node_modules" / "typescript" / "lib" / "typescript.js"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(not (NODE and TS_LIB.exists()),
                                reason="node + web/node_modules/typescript are needed")

_RUNNER = r"""
const ts = require(process.argv[2]);
const job = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const js = ts.transpileModule(job.code, { compilerOptions: {
  target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS } }).outputText;
const mod = { exports: {} };
new Function('module', 'exports', js + '\n;module.exports = {' + job.names.join(',') + '};')(mod, mod.exports);
process.stdout.write(JSON.stringify(job.calls.map(([fn, args]) => mod.exports[fn](...args))));
"""


def _src(rel: str) -> str:
    return (WEB_SRC / rel).read_text()


def _block(rel: str, start: str, end: str) -> str:
    text = _src(rel)
    i = text.index(start)
    return text[i:text.index(end, i)]


def _run_ts(tmp_path, code: str, names: list, calls: list) -> list:
    runner = tmp_path / "run_ts.js"
    runner.write_text(_RUNNER)
    payload = json.JSONEncoder().encode({"code": code, "names": names, "calls": calls})
    r = subprocess.run([NODE, str(runner), str(TS_LIB)], input=payload, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


# ── 1. Contract-pay note: the two feed pages say exactly what the digest says ─
# Fictional employers, plus agency names from the tool's own staffing list
# (market data) for the named-agency branch.
CONTRACT_CASES = [
    ("Contract", None, "Driftwood Labs"),
    ("Contract", "$140K–$200K/yr", "Harbor Lights Media"),
    ("Temp", "$115K–$150K/yr", "Larkspur Public Media"),
    ("Full-time/Contract", "$55K–$70K/yr", "Saltmarsh Finance LLC."),
    ("Contract", "$500–$800/day (≈$130K–$208K/yr)", "Gullwing Pictures"),
    ("Contract", "$45–$50/hr (≈$94K–$104K/yr)", "Brightwater Software"),
    ("Contract · W2", "$72–$78/hr (≈$150K–$162K/yr)", "Kelly Services"),
    ("Contract · 1099", None, None),
    ("W2", "$57.25/hr (≈$119K/yr)", "Northbank Talent (staffing)"),
    ("Contract", None, "Harbor Lights Media (via Brightline Talent)"),
    (None, "$40–$55/hr (≈$83K–$114K/yr)", None),
    ("W2", "$40/hr (≈$83K/yr)", "Some Studio"),
    ("Temp", "$32–$38/hr (≈$67K–$79K/yr)", "Larkspur Public Media"),
    ("Full-time", "$60/hr (≈$125K/yr)", "Apex Systems"),
    ("Full-time", "$120K–$150K/yr", None),
    (None, None, None),
    ("nan", "nan", "nan"),
]


@needs_node
@pytest.mark.parametrize("page", ["app/review/page.tsx", "app/jobs/page.tsx"])
def test_feed_page_contract_note_matches_the_digest(tmp_path, page):
    code = _block(page, "const CONTRACT_PAY_TAIL", "export default function")
    got = _run_ts(tmp_path, code, ["contractPayNote"],
                  [["contractPayNote", list(case)] for case in CONTRACT_CASES])
    assert got == [contract_pay_note(*case) for case in CONTRACT_CASES]


@pytest.mark.parametrize("page", ["app/review/page.tsx", "app/jobs/page.tsx"])
def test_feed_pages_no_longer_carry_the_fixed_hourly_note(page):
    text = _src(page)
    assert "Contract \\u2014 paid hourly (W2 through an agency, or 1099)" not in text
    assert "CONTRACT_PAY_NOTE" not in text
    assert "contractPayNote(job.employment_type, job.comp_text, job.company)" in text


# ── 5. Kanban: a proxy (inferred) stage is shown as uncertain ────────────────
def _kanban_code():
    return _block("app/kanban/page.tsx", "// Stages where an interview is on the table",
                  "/** Debounced notes input")


def _app(inferred: bool):
    y = date.today().year
    return {
        "status": "Rejected", "highest_stage": "Interview",
        "reached_at": f"{y}-03-10", "reached_inferred": inferred,
        "history": [
            {"status": "Applied", "at": f"{y}-03-03", "inferred": False},
            {"status": "Interview", "at": f"{y}-03-10", "inferred": inferred},
            {"status": "Rejected", "at": f"{y}-04-02", "inferred": False},
        ],
    }


@needs_node
def test_kanban_inferred_reach_is_shown_as_an_unconfirmed_stage(tmp_path):
    undated = dict(_app(True), reached_at=None)
    got = _run_ts(tmp_path, _kanban_code(), ["reachCaption", "historyLine"], [
        ["reachCaption", [_app(True)]],
        ["historyLine", [_app(True)]],
        ["reachCaption", [undated]],
    ])
    assert got[0] == "reached an interview stage · ~Mar 10"
    assert got[1] == "Applied Mar 3 → an interview stage ~Mar 10 → Rejected Apr 2"
    assert got[2] == "reached an interview stage"
    assert all("Interview" not in g for g in got)       # the guessed stage never shows


@needs_node
def test_kanban_confirmed_reach_names_the_stage(tmp_path):
    got = _run_ts(tmp_path, _kanban_code(), ["reachCaption", "historyLine"], [
        ["reachCaption", [_app(False)]],
        ["historyLine", [_app(False)]],
    ])
    assert got == ["reached Interview · Mar 10", "Applied Mar 3 → Interview Mar 10 → Rejected Apr 2"]


@needs_node
def test_kanban_prep_sheet_stays_readable_on_a_pre_ledger_dead_card(tmp_path):
    """A Rejected card created before the status history existed has only a
    Rejected snapshot, so highest_stage is None. Its prep sheet must stay
    readable before any interview proxy fills the stage in — Help: 'the sheet
    stays on the card to read'. The opposite direction: a pre-history closed
    card mis-clicked into Interview and straight back also ends with
    highest_stage None, but the sheet that click built carries prep_stage —
    it stays hidden."""
    def card(status, highest, has_prep, prep_stage=None):
        return {"status": status, "highest_stage": highest, "has_prep": has_prep,
                "prep_stage": prep_stage, "reached_at": None, "reached_inferred": False,
                "history": []}
    got = _run_ts(tmp_path, _kanban_code(), ["prepPanel"], [
        ["prepPanel", [card("Rejected", None, True)]],                    # pre-ledger sheet, no proxy yet
        ["prepPanel", [card("Ghosted", "Screening Call", True)]],         # stage filled in by a proxy
        ["prepPanel", [card("Rejected", "Interview", True, "Interview")]],  # a recorded reach
        ["prepPanel", [card("Interview", "Interview", False)]],           # live, sheet still building
        ["prepPanel", [card("Rejected", None, True, "Interview")]],       # mis-click on a pre-ledger dead card
        ["prepPanel", [card("Rejected", "Applied", True, "Interview")]],  # mis-click on a newer card
        ["prepPanel", [card("Applied", "Applied", True)]],                # sheet left by a mis-click
        ["prepPanel", [card("Rejected", None, False)]],                   # no sheet at all
    ])
    assert got == ["readonly", "readonly", "readonly", "live", None, None, None, None]


def test_help_explains_tilde_and_the_ten_minute_rule():
    help_text = " ".join(_src("app/help/page.tsx").split())
    assert ("A &quot;~&quot; marks a stage or date that is not confirmed yet" in help_text)
    assert "reached an interview stage · ~Mar 10" in help_text
    assert ("move the card straight back to where it was within ten minutes and it will not "
            "count") in help_text
    assert "Moving it on instead — to Rejected, or to a later stage — keeps the interview" in help_text
    assert "just move the card back within ten minutes and it will not count" not in help_text


# ── 6. The description is optional (Help + Input header) ─────────────────────
def test_help_and_input_say_the_description_is_optional():
    help_text = " ".join(_src("app/help/page.tsx").split())
    assert "Enter the full job description, title, and company name." not in help_text
    assert "The job description is optional" in help_text
    assert "leave it empty when the company reached out to you directly" in help_text
    assert "you get the general version of your CV" in help_text
    assert "You still get tailored docs." not in help_text
    header = " ".join(_src("app/input/page.tsx").split())
    assert "Paste a job description to save it directly and generate tailored documents." not in header
    assert "The description is optional" in header
    assert "with no description you get the general version of your CV" in header


# ── 7. The CV QA badge names the CV ──────────────────────────────────────────
def test_cv_qa_badge_labels_name_the_cv():
    badge = _src("components/CVQABadge.tsx")
    for label in ("CV QA: Passed", "CV QA: Review", "CV QA: Hallucination", "CV QA: Needs review"):
        assert f"label: '{label}'" in badge
    assert "label: 'Passed'" not in badge and "label: 'Review'" not in badge


# ── 8. "Did you send this CV?" is an accessible group ────────────────────────
def test_sent_cv_question_is_an_accessible_group():
    panel = _src("components/DocumentPanel.tsx")
    block = panel[panel.index("Did you send this CV?") - 400:panel.index("Could not save") + 80]
    assert "const sentQuestionId = useId()" in panel
    assert 'id={sentQuestionId}' in block
    assert 'role="group"' in block and "aria-labelledby={sentQuestionId}" in block
    assert 'role="status"' in block                       # the saved note is announced
    assert '<span role="alert"' in block                 # the error is announced
    assert "disabled={savingSent !== null && !saving}" in block  # only the OTHER options
    assert 'size="sm"' in block and 'size="xs"' not in block


# ── 6. Kanban: "How far did it get?" on a closed card (2026-10-28) ───────────
def _closed_card(status="Rejected", highest="Interview", outcome=None, inferred=False):
    return {"status": status, "highest_stage": highest, "reached_at": None,
            "reached_inferred": inferred, "history": [], "outcome": outcome}


@needs_node
def test_kanban_outcome_helpers(tmp_path):
    answered = _closed_card(outcome={"stage": "Screening Call", "at": None,
                                     "what_happened": "They chose another candidate"})
    ghosted = dict(answered, status="Ghosted",
                   outcome={"stage": "Applied", "what_happened": None, "at": None})
    got = _run_ts(tmp_path, _kanban_code(),
                  ["endedLine", "outcomePrefill", "outcomeStages", "withHint",
                   "outcomeQuestion", "outcomeHints"], [
        ["endedLine", [answered]],
        ["endedLine", [ghosted]],
        ["endedLine", [_closed_card()]],                         # no answer yet
        ["endedLine", [dict(answered, status="Interview")]],     # live again
        ["outcomePrefill", [answered]],                          # the user's answer wins
        ["outcomePrefill", [_closed_card(highest="Screening Call", inferred=True)]],  # a guess to confirm
        ["outcomePrefill", [_closed_card(highest=None)]],
        ["outcomePrefill", [_closed_card(highest="To Apply")]],
        ["outcomeStages", [_closed_card()]],
        ["outcomeStages", [_closed_card(highest="Accepted")]],
        ["withHint", ["", "No feedback given"]],
        ["withHint", ["Two rounds.", "No feedback given"]],
        ["withHint", ["no feedback given at all", "No feedback given"]],
        # the user ended it (I Withdrew)
        ["endedLine", [dict(answered, status="I Withdrew",
                            outcome={"stage": "Screening Call", "at": None,
                                     "what_happened": "Pay didn’t match"})]],
        ["endedLine", [dict(answered, status="I Withdrew",
                            outcome={"stage": "Offer", "at": None, "what_happened": None})]],
        ["outcomeQuestion", [dict(answered, status="I Withdrew")]],
        ["outcomeQuestion", [ghosted]],
        ["outcomeQuestion", [answered]],
        ["outcomeHints", [dict(answered, status="I Withdrew")]],
        ["outcomeHints", [answered]],
    ])
    assert got[0] == "Ended after the screening call — “They chose another candidate”"
    assert got[1] == "Went quiet after applying"
    assert got[2] is None and got[3] is None
    assert got[4:8] == ["Screening Call", "Screening Call", "Applied", "Applied"]
    assert got[8] == ["Applied", "Screening Call", "Interview", "Final Interview", "Offer"]
    assert got[9][-1] == "Accepted"
    assert got[10:13] == ["No feedback given", "Two rounds; No feedback given",
                          "no feedback given at all"]
    assert got[13] == "You withdrew after the screening call — “Pay didn’t match”"
    assert got[14] == "You turned down the offer"
    assert got[15:18] == ["How far did it get before you withdrew?",
                          "How far did it get before it went quiet?", "How far did it get?"]
    assert "Took another offer" in got[18] and "Never heard back" not in got[18]
    assert "They chose another candidate" in got[19]


def test_kanban_asks_on_close_and_stats_shows_the_endings():
    page = _src("app/kanban/page.tsx")
    # closing a live card opens the question inside the card — never a pop-up;
    # so does switching how it closed before the user answered
    assert "if (!(from && CLOSED_STATUSES.includes(from) && answered))" in page
    assert "handleStatusChange(app.app_id, v, app.status, !!app.outcome)" in page
    assert "onDone={(saved) => closeOutcome(app.app_id, saved)}" in page
    assert "Dialog" not in page
    assert "const CLOSED_STATUSES: string[] = ['Rejected', 'Ghosted', WITHDREW]" in page
    # any closed card in the list can take details later, without opening it first
    assert "'+ Add details'" in page and "openOutcome(app.app_id)" in page
    # a closed card stays under a filter it no longer matches only while its
    # question is on screen; folding the card away puts the question away
    assert "(a.app_id === outcomeEdit && a.app_id === expandedRow)" in page
    assert "leavesView(outcomeEdit, false)" in page
    stats = _src("app/stats/page.tsx")
    assert "How processes ended" in stats and "endings.notes.map" in stats
    assert "endedHow(n.status, n.stage)" in stats and "You withdrew from" in stats
    api_ts = _src("lib/api.ts")
    assert "setOutcome:" in api_ts and "/outcome`" in api_ts


# ── 9. A CV being generated says so (2026-11-03) ─────────────────────────────
def test_kanban_card_says_generating_while_the_server_is_at_it():
    page = _src("app/kanban/page.tsx")
    # the server's list of generations in flight, polled only while not empty
    assert "queryFn: api.jobs.generating" in page
    assert "query.state.data?.job_ids.length ? 5000 : false" in page
    # a click on a job that is already generating keeps its card busy
    assert "clickedGen.has(jobId) || !!serverGenerating?.includes(jobId)" in page
    assert "disabled={genBusy}" in page and "'Generating…' : 'Generate'" in page
    assert "busy={genBusy}" in page
    # one card's click never turns every card into "Generating..."
    assert "generateDocsMutation.isPending" not in page
    # a generation that ended refreshes the board, so View appears by itself
    assert "prevGenerating.current.some((id) => !now.includes(id))" in page
    panel = _src("components/DocumentPanel.tsx")
    assert "disabled={generating || busy}" in panel
    assert "This usually takes 5–10 minutes" in panel
    assert "generating: () => fetchJSON<GeneratingJobs>" in _src("lib/api.ts")
    assert "Its Kanban card says Generating…" in " ".join(_src("app/input/page.tsx").split())
