"""Phase 27c "Is this posting real?" — the web surface.

Same approach as test_phase27_web_surface: small helpers are cut out of the
.tsx files, transpiled with the repo's own TypeScript and run in node (skipped
when node / TypeScript are missing); markup checks scan the source. The point
is a SUBTLE surface: one muted chip only for Mixed / Check first, the panel
fetched only when visible, and only the "gone" fact on a Kanban card.
"""
import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

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


REVIEW = "app/review/page.tsx"
KANBAN = "app/kanban/page.tsx"
DRAWER = "components/JobDetailDrawer.tsx"
PANEL = "components/PostingCheckPanel.tsx"
HELP = "app/help/page.tsx"


def _read(kind):
    return {"read": kind, "label": kind, "reasons": ["why"]}


# ── Review row chip ──────────────────────────────────────────────────────────
@needs_node
def test_chip_only_for_mixed_and_check_first(tmp_path):
    code = _block(REVIEW, "function postingChipText", "// end postingChipText")
    out = _run_ts(tmp_path, code, ["postingChipText"], [
        ["postingChipText", [_read("looks_active")]],
        ["postingChipText", [None]],
        ["postingChipText", []],
        ["postingChipText", [_read("mixed")]],
        ["postingChipText", [_read("check_first")]],
        ["postingChipText", [_read("something_new")]],
        ["postingChipText", [_read("not_enough_info")]],   # no date, no board check: no chip
    ])
    assert out == [None, None, None, "Mixed signals", "Check first", None, None]


def test_chip_is_one_muted_pill_right_after_the_age_pill():
    src = _src(REVIEW)
    i = src.index("{agePill(job.date_posted)}")
    chip = src[i:src.index("</div>", i)]
    assert "postingChipText(job.posting_read)" in chip
    assert "rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground" in chip
    assert "join('; ')" in chip
    # muted only: no amber / red styling on the chip
    assert not re.search(r"amber|red-|emerald", chip)
    assert src.count("postingChipText(job.posting_read)") == 2   # condition + text, one chip


# ── Panel placement and fetch gating ─────────────────────────────────────────
def test_review_panel_is_gated_by_isExpanded_and_sits_above_company_intel():
    src = _src(REVIEW)
    tag = "<PostingCheckPanel jobId={job.id} enabled={isExpanded} />"
    assert src.count("<PostingCheckPanel") == 1 and tag in src
    expanded = src.index("{/* Expanded content */}")
    panel = src.index(tag)
    intel = src.index("Company Intel\n", expanded)
    assert expanded < panel < intel


def test_drawer_panel_is_gated_by_open_and_sits_above_company_intel():
    src = _src(DRAWER)
    tag = "<PostingCheckPanel jobId={job.id} enabled={open} />"
    assert tag in src
    assert src.index(tag) < src.index("<CompanyIntelPanel")


def test_kanban_panel_sits_above_company_intel_in_the_expanded_row():
    src = _src(KANBAN)
    tag = "<PostingCheckPanel jobId={app.job_id} enabled />"
    assert tag in src
    expanded = src.index("{isExpanded && (")
    assert expanded < src.index(tag) < src.index("<CompanyIntelPanel", expanded)


def test_panel_fetches_only_when_enabled_and_reads_on_demand():
    src = _src(PANEL)
    assert "queryKey: ['posting-check', jobId]" in src
    assert "enabled: enabled && !!jobId" in src
    assert "api.postingCheck.get(jobId)" in src
    assert "api.postingCheck.readPosting(jobId)" in src
    assert "invalidateQueries({ queryKey: ['posting-check', jobId] })" in src
    assert "'Read the posting'" in src and "'Reading…'" in src
    assert "disabled={readMutation.isPending}" in src
    assert "!data.quality_available" in src
    # the read label is shown muted, facts carry their source
    assert "Posting check" in src
    for label in ("Company job board", "Our scanner", "The posting", "Public source",
                  "Your history"):
        assert f"'{label}'" in src
    # caution facts in amber; quotes muted italics
    assert "text-amber-700 dark:text-amber-400" in src
    assert "italic text-muted-foreground" in src
    # only plain web links become clickable
    assert "/^https?:\\/\\//i" in src


def test_api_client_paths():
    src = _src("lib/api.ts")
    assert "/api/jobs/${id}/posting-check`" in src
    assert "/api/jobs/${id}/posting-check/quality`" in src
    types = _src("lib/types.ts")
    assert "read: 'looks_active' | 'mixed' | 'check_first' | 'not_enough_info'" in types
    assert "posting_read?: PostingRead | null" in types
    assert types.count("listing_gone_since?: string | null") == 2   # Job + Application


# ── Kanban card: the gone fact only ──────────────────────────────────────────
@needs_node
def test_kanban_gone_line(tmp_path):
    code = (_block(KANBAN, "function stageDate", "// An inferred entry")
            + _block(KANBAN, "function goneLine", "// end goneLine"))
    year = date.today().year
    out = _run_ts(tmp_path, code, ["goneLine"], [
        ["goneLine", [{"company": "Harbor Lights Media", "listing_status": "gone",
                       "listing_gone_since": f"{year}-10-02 12:00:00"}]],
        ["goneLine", [{"company": "Harbor Lights Media", "listing_status": "gone",
                       "listing_gone_since": None}]],
        ["goneLine", [{"company": "Harbor Lights Media", "listing_status": "listed",
                       "listing_gone_since": None}]],
        ["goneLine", [{"company": "Harbor Lights Media", "listing_status": None}]],
        ["goneLine", [{"company": "Harbor Lights Media"}]],
    ])
    assert out == ["No longer on Harbor Lights Media's job board since Oct 2",
                   "No longer on Harbor Lights Media's job board", None, None, None]


def test_kanban_card_shows_nothing_else_from_the_posting_check():
    src = _src(KANBAN)
    title_cell = _block(KANBAN, "{/* Title */}", "{/* Company */}")
    assert "goneLine(app)" in title_cell
    assert "text-xs text-muted-foreground" in title_cell
    assert "posting_read" not in src
    assert "postingChipText" not in src


# ── Help ─────────────────────────────────────────────────────────────────────
def test_help_section():
    src = _src(HELP)
    assert 'title="Is This Posting Real?" id="posting-check"' in src
    assert '<TocLink href="#posting-check"' in src
    # after Company Intel, before the Jobs tab
    assert (src.index('id="company-intel"') < src.index('id="posting-check"')
            < src.index('id="jobs-tab"'))
    section = _block(HELP, 'id="posting-check"', 'id="jobs-tab"')
    text = re.sub(r"\s+", " ", section)
    assert "not a score" in text
    assert "two complete checks" in text
    assert "Read the posting" in text
    assert "never hidden" in text
    assert "when you track a job" in text
    for read in ("Looks active", "Mixed signals", "Check before investing time"):
        assert read in text
