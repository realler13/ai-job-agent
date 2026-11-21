"""UNMOCKED CV parser/writer tests (Phase 27, workstream A1).

Until 2026-10 every test patched fill_docx_with_tailored_data, which is how the
writer shipped base bullets for eight months: the PROFESSIONAL SUMMARY branch
swallowed the EXPERIENCE header, so the experience branch never ran. Nothing in
this file mocks the parser or the writer — each test writes a real .docx and
re-opens it with python-docx.

The base is built from a committed FICTIONAL career (tests/fixtures/) with the
same topology as the real bases; the real base_cv*.docx files are private and
gitignored, so the tests that use them skip when they are absent.

Patching note: trim_cv_to_length uses the module-level cv_tailor.call_llm, so
patch `job_agent.cv_tailor.call_llm` (fake signatures need **kw).
"""

import copy
import glob
import json
import os
from unittest.mock import patch

import pytest
from docx import Document
from docx.oxml.ns import qn
from lxml import etree

from job_agent import cv_tailor as t
from tests.fixtures.fictional_cv import PARA_ID, build_fixture_docx, load_fictional_career
from tests.fixtures.persona import YEARS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_BASES = sorted(f for f in glob.glob(os.path.join(ROOT, "base_cv*.docx")) if "~$" not in f)
REAL_DOMAIN_BASES = [f for f in REAL_BASES if os.path.basename(f) != "base_cv.docx"]
SECTION_HEADERS = ["PROFESSIONAL SUMMARY", "EXPERIENCE", "VOLUNTEER EXPERIENCE", "SKILLS", "EDUCATION"]


def _fixture_roles():
    """[(title, company_line, [bullets])] straight from the fictional career file."""
    items = dict(load_fictional_career()["sections"])["EXPERIENCE"]
    roles = []
    for kind, text in items:
        if kind == "title":
            roles.append([text, None, []])
        elif kind == "line":
            roles[-1][1] = text
        else:
            roles[-1][2].append(text)
    return [tuple(r) for r in roles]


ROLES = _fixture_roles()


@pytest.fixture
def base(tmp_path):
    return build_fixture_docx(tmp_path / "fixture_base.docx")


@pytest.fixture
def legacy_base(tmp_path):
    return build_fixture_docx(tmp_path / "fixture_legacy_base.docx", legacy=True)


def _texts(path):
    return [p.text for p in Document(str(path)).paragraphs]


def _xml(el):
    return etree.tostring(el) if el is not None else b""


def _signature(path):
    """Per paragraph: (text, style, numPr XML, ind XML, whole pPr XML)."""
    rows = []
    for p in Document(str(path)).paragraphs:
        pPr = p._p.pPr
        rows.append((p.text, p.style.name,
                     _xml(pPr.find(qn("w:numPr")) if pPr is not None else None),
                     _xml(pPr.find(qn("w:ind")) if pPr is not None else None),
                     _xml(pPr)))
    return rows


def _paragraph_xml(path):
    """Whole XML of every paragraph (runs included)."""
    return [etree.tostring(p._p) for p in Document(str(path)).paragraphs]


def _real_indexes(data):
    return [i for i, r in enumerate(data["experience"]) if r["description"]]


def _with_sentinels(data, counts):
    """Copy of *data* with SENTINEL bullets on every real role (counts cycle) and
    a two-line sentinel summary. Returns (tailored, {index: [bullets]})."""
    tailored = copy.deepcopy(data)
    tailored["summary"] = "SENTINEL_SUMMARY first paragraph.\nSENTINEL_SUMMARY second paragraph."
    expected = {}
    for n, i in enumerate(_real_indexes(tailored)):
        k = counts[n % len(counts)]
        expected[i] = [f"SENTINEL_R{i:02d}_B{j} for this role only" for j in range(k)]
        tailored["experience"][i]["description"] = "\n".join("- " + b for b in expected[i])
    return tailored, expected


def _bullet_block(paras, start):
    """Texts of the consecutive list paragraphs from index *start*, and the index after them."""
    j, block = start, []
    while j < len(paras) and paras[j].style.name.startswith("List"):
        block.append(paras[j].text)
        j += 1
    return block, j


# ---------------------------------------------------------------- parser ----

class TestParser:
    def test_fixture_topology(self, base):
        """Convention kept: each real role = an empty title placeholder + the
        company/date entry carrying the bullets."""
        data = t.parse_docx_to_structure(base)
        assert len(data["experience"]) == 2 * len(ROLES)
        for k, (title, line, bullets) in enumerate(ROLES):
            placeholder, real = data["experience"][2 * k], data["experience"][2 * k + 1]
            assert placeholder == {"title": title, "company": "", "description": ""}
            assert real["title"] == line
            assert real["description"].split("\n") == bullets
        assert data["summary"].startswith(f"Video producer with {YEARS}")
        assert "\n" not in data["summary"]

    def test_words_inside_bullets_never_change_the_section(self, base):
        """skills / educational / education / volunteer / summary / experience
        inside a bullet used to end the Experience section (substring match)."""
        data = t.parse_docx_to_structure(base)
        all_bullets = "\n".join(r["description"] for r in data["experience"]).lower()
        for word in ("skills", "educational", "education", "volunteer", "summary", "experience"):
            assert word in all_bullets, f"fixture lost its {word!r} trap"
        # every role after the trap bullets is still parsed
        assert [r["title"] for r in data["experience"]][-1] == ROLES[-1][1]
        assert data["experience"][-1]["description"].split("\n") == ROLES[-1][2]

    def test_company_line_containing_volunteer_is_a_role_not_a_section(self, base):
        assert "Volunteer" in ROLES[-1][1]
        data = t.parse_docx_to_structure(base)
        assert data["experience"][-1]["title"] == ROLES[-1][1]

    def test_static_sections_are_not_roles(self, base):
        data = t.parse_docx_to_structure(base)
        blob = json.dumps(data)
        for static in ("Youth Film Lab", "edit supervision", "Film Studies"):
            assert static not in blob

    def test_list_paragraph_reading_like_a_header_is_a_bullet(self, tmp_path):
        doc = Document()
        doc.add_paragraph("SUMMARY")
        doc.add_paragraph("One line.")
        doc.add_paragraph("PROFESSIONAL EXPERIENCE")
        doc.add_paragraph("Editor")
        doc.add_paragraph("Nowhere Films | 2020 – 2022")
        doc.add_paragraph("SKILLS", style="List Bullet")
        doc.add_paragraph("EDUCATION", style="List Bullet")
        doc.add_paragraph("- Summary")          # literal bullet character, Normal style
        doc.add_paragraph("Skills")             # the real header, mixed case
        doc.add_paragraph("Not a role", style="List Bullet")
        path = tmp_path / "headers.docx"
        doc.save(str(path))
        data = t.parse_docx_to_structure(str(path))
        assert data["summary"] == "One line."
        assert [r["title"] for r in data["experience"]] == ["Editor", "Nowhere Films | 2020 – 2022"]
        assert data["experience"][1]["description"] == "SKILLS\nEDUCATION\n- Summary"

    def test_missing_file_returns_empty_structure(self):
        assert t.parse_docx_to_structure("/nonexistent/base.docx") == {"summary": "", "experience": []}

    @pytest.mark.skipif(not REAL_DOMAIN_BASES, reason="private base CVs not present")
    @pytest.mark.parametrize("path", REAL_DOMAIN_BASES, ids=os.path.basename)
    def test_real_domain_bases_keep_the_placeholder_entry_convention(self, path):
        """Each role = an empty title placeholder + the company entry with bullets."""
        data = t.parse_docx_to_structure(path)
        roles = len(data["experience"]) // 2
        assert roles >= 1
        assert [bool(r["description"]) for r in data["experience"]] == [False, True] * roles
        assert data["summary"]


# ---------------------------------------------------------------- writer ----

class TestWriterSentinel:
    COUNTS = [7, 1, 4, 3]   # base has 5 / 3 / 4 / 2 bullets → grow, shrink, same, grow

    def _write(self, base, tmp_path):
        data = t.parse_docx_to_structure(base)
        tailored, expected = _with_sentinels(data, self.COUNTS)
        out = str(tmp_path / "out.docx")
        t.fill_docx_with_tailored_data(base, out, copy.deepcopy(tailored))
        return tailored, expected, out

    def test_sentinel_bullets_land_under_their_own_role(self, base, tmp_path):
        """THE regression test for the 8-month bug: every role's bullets must sit
        between its own company line and the next title line. (Fails on the
        pre-fix writer, which never reached the experience section.)"""
        _, expected, out = self._write(base, tmp_path)
        paras = Document(out).paragraphs
        texts = [p.text for p in paras]
        for k, (title, line, _) in enumerate(ROLES):
            at = texts.index(line)
            assert texts.count(line) == 1
            # the title line sits directly above the company line: placeholder stayed empty
            assert texts[at - 1] == title
            block, after = _bullet_block(paras, at + 1)
            assert block == expected[2 * k + 1], f"role {k} ({line})"
            next_line = ROLES[k + 1][0] if k + 1 < len(ROLES) else "VOLUNTEER EXPERIENCE"
            assert texts[after] == next_line

    def test_summary_written_and_headers_unchanged(self, base, tmp_path):
        tailored, _, out = self._write(base, tmp_path)
        texts = _texts(out)
        a, b = texts.index("PROFESSIONAL SUMMARY"), texts.index("EXPERIENCE")
        assert texts[a + 1:b] == tailored["summary"].split("\n")
        assert [x for x in texts if x in SECTION_HEADERS] == SECTION_HEADERS
        base_texts = _texts(base)
        assert texts[:a + 1] == base_texts[:a + 1]                      # name + contact lines
        tail = texts.index("VOLUNTEER EXPERIENCE")
        assert texts[tail:] == base_texts[base_texts.index("VOLUNTEER EXPERIENCE"):]  # static sections

    def test_parser_reads_back_exactly_what_was_written(self, base, tmp_path):
        tailored, expected, out = self._write(base, tmp_path)
        reread = t.parse_docx_to_structure(out)
        assert reread["summary"] == tailored["summary"]
        assert [r["title"] for r in reread["experience"]] == [r["title"] for r in tailored["experience"]]
        for i, role in enumerate(reread["experience"]):
            assert role["description"] == "\n".join(expected.get(i, []))

    def test_written_bullets_keep_the_bullet_style(self, base, tmp_path):
        _, expected, out = self._write(base, tmp_path)
        wanted = {b for bullets in expected.values() for b in bullets}
        styles = {p.style.name for p in Document(out).paragraphs if p.text in wanted}
        assert styles == {"List Bullet"}

    def test_empty_title_placeholders_never_receive_bullets(self, base, tmp_path):
        data = t.parse_docx_to_structure(base)
        tailored = copy.deepcopy(data)
        for i, role in enumerate(tailored["experience"]):
            if not role["description"]:
                role["description"] = f"- INJECTED_UNDER_TITLE_{i}"
        out = str(tmp_path / "out.docx")
        t.fill_docx_with_tailored_data(base, out, tailored)
        assert "INJECTED_UNDER_TITLE" not in "\n".join(_texts(out))
        assert _signature(out) == _signature(base)

    def test_title_mismatch_keeps_base_bullets(self, base, tmp_path):
        """Position alone is not enough: bullets are written only under the
        header line they were generated for."""
        data = t.parse_docx_to_structure(base)
        tailored, expected = _with_sentinels(data, [3])
        # swap the two real 'Producer' entries (company lines + their bullets)
        tailored["experience"][1], tailored["experience"][3] = \
            tailored["experience"][3], tailored["experience"][1]
        out = str(tmp_path / "out.docx")
        t.fill_docx_with_tailored_data(base, out, tailored)
        reread = t.parse_docx_to_structure(out)
        assert reread["experience"][1]["description"] == data["experience"][1]["description"]
        assert reread["experience"][3]["description"] == data["experience"][3]["description"]
        assert reread["experience"][5]["description"] == "\n".join(expected[5])

    def test_empty_tailored_content_keeps_base_content(self, base, tmp_path):
        data = t.parse_docx_to_structure(base)
        tailored = copy.deepcopy(data)
        tailored["summary"] = ""
        tailored["experience"][1]["description"] = ""
        out = str(tmp_path / "out.docx")
        t.fill_docx_with_tailored_data(base, out, tailored)
        assert _signature(out) == _signature(base)

    def test_writer_does_not_mutate_its_input(self, base, tmp_path):
        data = t.parse_docx_to_structure(base)
        tailored, _ = _with_sentinels(data, self.COUNTS)
        snapshot = copy.deepcopy(tailored)
        t.fill_docx_with_tailored_data(base, str(tmp_path / "out.docx"), tailored)
        assert tailored == snapshot

    def test_multi_line_summary_then_single_line(self, base, tmp_path):
        data = t.parse_docx_to_structure(base)
        three = dict(data, summary="First.\n\nSecond.\nThird.")
        out3 = str(tmp_path / "three.docx")
        t.fill_docx_with_tailored_data(base, out3, three)
        assert t.parse_docx_to_structure(out3)["summary"] == "First.\nSecond.\nThird."
        # a 3-paragraph summary shrinks back to one without leaving strays
        out1 = str(tmp_path / "one.docx")
        t.fill_docx_with_tailored_data(out3, out1, dict(data, summary="Only one."))
        texts = _texts(out1)
        a, b = texts.index("PROFESSIONAL SUMMARY"), texts.index("EXPERIENCE")
        assert texts[a + 1:b] == ["Only one."]

    @pytest.mark.skipif(not REAL_BASES, reason="private base CVs not present")
    @pytest.mark.parametrize("path", REAL_BASES, ids=os.path.basename)
    def test_real_base_smoke(self, path, tmp_path):
        data = t.parse_docx_to_structure(path)
        tailored, expected = _with_sentinels(data, [5, 2, 4])
        out = str(tmp_path / "out.docx")
        t.fill_docx_with_tailored_data(path, out, copy.deepcopy(tailored))
        reread = t.parse_docx_to_structure(out)
        assert reread["summary"] == tailored["summary"]
        assert len(reread["experience"]) == len(data["experience"])
        for i, role in enumerate(reread["experience"]):
            assert role["title"] == data["experience"][i]["title"]
            assert role["description"] == "\n".join(expected.get(i, []))
        assert t.verify_docx_roundtrip(out, tailored, base_docx_path=path) == {"ok": True, "mismatches": []}
        ids = [p._p.get(PARA_ID) for p in Document(out).paragraphs if p._p.get(PARA_ID)]
        assert len(ids) == len(set(ids))


class TestB0Identity:
    """B0: writing a base's own parsed data back must change nothing — same
    paragraph texts, same styles, same numbering XML, same pPr, same runs."""

    def test_fixture_base(self, base, tmp_path):
        out = str(tmp_path / "b0.docx")
        t.fill_docx_with_tailored_data(base, out, t.parse_docx_to_structure(base))
        assert _signature(out) == _signature(base)
        assert _paragraph_xml(out) == _paragraph_xml(base)

    def test_legacy_fixture_base(self, legacy_base, tmp_path):
        """Includes a three-run bullet: a paragraph whose text is already right
        is not rewritten, so its runs (bold fragment) survive."""
        out = str(tmp_path / "b0.docx")
        t.fill_docx_with_tailored_data(legacy_base, out, t.parse_docx_to_structure(legacy_base))
        assert _signature(out) == _signature(legacy_base)
        assert _paragraph_xml(out) == _paragraph_xml(legacy_base)
        assert max(len(p.runs) for p in Document(out).paragraphs) == 3

    @pytest.mark.skipif(not REAL_BASES, reason="private base CVs not present")
    @pytest.mark.parametrize("path", REAL_BASES, ids=os.path.basename)
    def test_real_bases(self, path, tmp_path):
        out = str(tmp_path / "b0.docx")
        data = t.parse_docx_to_structure(path)
        t.fill_docx_with_tailored_data(path, out, copy.deepcopy(data))
        assert _signature(out) == _signature(path)
        assert _paragraph_xml(out) == _paragraph_xml(path)
        assert t.verify_docx_roundtrip(out, data, base_docx_path=path)["ok"]


class TestFormattingPreserved:
    """Legacy layout: per-role numPr, spacing on the first/last bullet, ind on
    one role, a multi-run bullet, an empty spacer after the summary."""
    NUM_IDS = [19, 21, 22, 23]

    def _write(self, legacy_base, tmp_path, counts):
        data = t.parse_docx_to_structure(legacy_base)
        tailored, expected = _with_sentinels(data, counts)
        out = str(tmp_path / "out.docx")
        t.fill_docx_with_tailored_data(legacy_base, out, tailored)
        paras = Document(out).paragraphs
        texts = [p.text for p in paras]
        blocks = []
        for _, line, _ in ROLES:
            at = texts.index(line)
            _, after = _bullet_block(paras, at + 1)
            blocks.append(paras[at + 1:after])
        return expected, paras, blocks

    @staticmethod
    def _num_id(p):
        return p._p.pPr.numPr.numId.val

    @pytest.mark.parametrize("counts", [[7, 5, 6, 4], [2, 1, 2, 1], [5, 3, 4, 2]],
                             ids=["grow", "shrink", "same"])
    def test_each_role_keeps_its_own_numbering_and_indent(self, legacy_base, tmp_path, counts):
        expected, paras, blocks = self._write(legacy_base, tmp_path, counts)
        for k, block in enumerate(blocks):
            assert [p.text for p in block] == expected[2 * k + 1]
            assert {self._num_id(p) for p in block} == {self.NUM_IDS[k]}
            assert {p.style.name for p in block} == {"List Bullet"}
            has_ind = {p._p.pPr.find(qn("w:ind")) is not None for p in block}
            assert has_ind == {k == 3}
        for p in paras:
            if p._p.pPr is not None:
                tags = [child.tag for child in p._p.pPr]
                assert len(tags) == len(set(tags)), f"duplicated pPr child in {p.text[:30]!r}"

    def test_first_and_last_bullet_keep_their_spacing(self, legacy_base, tmp_path):
        _, _, blocks = self._write(legacy_base, tmp_path, [7, 2, 3, 4])
        for block in blocks[:3]:           # roles with >= 3 base bullets
            assert block[0].paragraph_format.space_before is not None
            assert block[-1].paragraph_format.space_after is not None
            for middle in block[1:-1]:
                assert middle.paragraph_format.space_before is None
                assert middle.paragraph_format.space_after is None

    def test_rewritten_bullets_are_single_run_and_spacer_survives(self, legacy_base, tmp_path):
        _, paras, blocks = self._write(legacy_base, tmp_path, [7, 5, 6, 4])
        assert all(len(p.runs) == 1 for block in blocks for p in block)
        texts = [p.text for p in paras]
        assert texts[texts.index("EXPERIENCE") - 1] == ""   # spacer after the summary

    def test_added_bullets_do_not_duplicate_paragraph_ids(self, legacy_base, tmp_path):
        """A grown role gets copies of an existing bullet; w14:paraId must stay
        unique per paragraph."""
        _, paras, blocks = self._write(legacy_base, tmp_path, [7, 5, 6, 4])
        ids = [p._p.get(PARA_ID) for p in paras if p._p.get(PARA_ID)]
        assert len(ids) == len(set(ids))
        assert len(ids) == len(Document(legacy_base).paragraphs)       # only the copies lack one
        assert sum(1 for block in blocks for p in block if p._p.get(PARA_ID) is None) == 2 + 2 + 2 + 2

    def test_static_sections_keep_their_own_numbering(self, legacy_base, tmp_path):
        """No 'canonical' restyling: SKILLS/EDUCATION bullets are not touched."""
        _, paras, _ = self._write(legacy_base, tmp_path, [7, 1, 4, 3])
        start = [p.text for p in paras].index("VOLUNTEER EXPERIENCE")
        base_sig = _signature(legacy_base)
        base_start = [row[0] for row in base_sig].index("VOLUNTEER EXPERIENCE")
        out_sig = [(p.text, p.style.name, _xml(p._p.pPr)) for p in paras[start:]]
        assert out_sig == [(r[0], r[1], r[4]) for r in base_sig[base_start:]]


# ---------------------------------------------------- _apply_revised_json ----

def _single_entry_layout():
    """One entry per role (title + company), two roles sharing the title 'Producer'."""
    return {"summary": "S", "experience": [
        {"title": "Producer", "company": "Tidewater Signal Studios", "description": "t1\nt2\nt3"},
        {"title": "Producer", "company": "Lanternfish Media Collective", "description": "l1\nl2\nl3"},
        {"title": "Editor", "company": "Copperline Broadcast Group", "description": "c1\nc2"},
    ]}


class TestApplyRevisedJson:
    def test_same_shape_revision_maps_by_position(self, base):
        data = t.parse_docx_to_structure(base)
        revised = json.loads(t._tailored_data_to_json(data))
        for i, rr in enumerate(revised["experience"]):
            if rr["bullets"]:
                rr["bullets"] = [f"NEW_{i}_{j}" for j in range(len(rr["bullets"]))]
        out = t._apply_revised_json(data, revised)
        for i, role in enumerate(out["experience"]):
            n = len(data["experience"][i]["description"].split("\n")) if data["experience"][i]["description"] else 0
            assert role["description"] == "\n".join(f"NEW_{i}_{j}" for j in range(n))
            assert role["title"] == data["experience"][i]["title"]

    def test_reordered_revision_maps_by_company_line(self, base):
        data = t.parse_docx_to_structure(base)
        revised = {"experience": [
            {"title": role["title"], "bullets": [f"REORDER_{i}_{j}" for j in range(3)]}
            for i, role in reversed(list(enumerate(data["experience"]))) if role["description"]]}
        out = t._apply_revised_json(data, revised)
        for i, role in enumerate(out["experience"]):
            if data["experience"][i]["description"]:
                assert role["description"] == "\n".join(f"REORDER_{i}_{j}" for j in range(3))
            else:
                assert role["description"] == ""

    def test_empty_title_placeholders_are_frozen(self, base):
        """A revision that hangs bullets on a title line ('Producer' appears
        twice) used to populate BOTH placeholders."""
        data = t.parse_docx_to_structure(base)
        injection = {"experience": [{"title": "Producer", "bullets": ["UNWARRANTED_1", "UNWARRANTED_2"]}]}
        out = t._apply_revised_json(data, injection)
        assert out["experience"] == data["experience"]
        # same in a same-shape answer
        revised = json.loads(t._tailored_data_to_json(data))
        for rr in revised["experience"]:
            if not rr["bullets"]:
                rr["bullets"] = ["UNWARRANTED"]
        out = t._apply_revised_json(data, revised)
        assert out["experience"] == data["experience"]

    def test_repeated_titles_map_by_position_when_shape_is_kept(self):
        td = _single_entry_layout()
        revised = {"experience": [
            {"title": "Producer", "bullets": ["T1", "T2", "T3"]},
            {"title": "Producer", "bullets": ["L1", "L2", "L3"]},
            {"title": "Editor", "bullets": ["C1", "C2"]}]}
        out = t._apply_revised_json(td, revised)
        assert [r["description"] for r in out["experience"]] == ["T1\nT2\nT3", "L1\nL2\nL3", "C1\nC2"]
        assert [r["company"] for r in out["experience"]] == [r["company"] for r in td["experience"]]

    def test_ambiguous_repeated_title_is_rejected(self):
        """One 'Producer' with no company line cannot be attributed — the old
        title-only match copied it under BOTH employers."""
        td = _single_entry_layout()
        for revised in (
            {"experience": [{"title": "Producer", "bullets": ["X1", "X2", "X3"]}]},
            {"experience": [{"title": "Producer", "bullets": ["X1", "X2", "X3"]},
                            {"title": "Producer", "bullets": ["Y1", "Y2", "Y3"]}]},
        ):
            out = t._apply_revised_json(td, revised)
            assert out["experience"] == td["experience"]

    def test_company_line_settles_a_repeated_title(self):
        td = _single_entry_layout()
        # one role only, and the two Producers swapped — both carry their company
        out = t._apply_revised_json(td, {"experience": [
            {"title": "Producer", "company": "Lanternfish Media Collective", "bullets": ["L1", "L2", "L3"]}]})
        assert [r["description"] for r in out["experience"]] == ["t1\nt2\nt3", "L1\nL2\nL3", "c1\nc2"]
        out = t._apply_revised_json(td, {"experience": [
            {"title": "Producer", "company": "Lanternfish Media Collective", "bullets": ["L1", "L2", "L3"]},
            {"title": "Producer", "company": "Tidewater Signal Studios", "bullets": ["T1", "T2", "T3"]},
            {"title": "Editor", "company": "Copperline Broadcast Group", "bullets": ["C1", "C2"]}]})
        assert [r["description"] for r in out["experience"]] == ["T1\nT2\nT3", "L1\nL2\nL3", "C1\nC2"]

    def test_one_revised_entry_never_feeds_two_roles(self):
        """The first 'Producer' has no company line, so the revised entry fits it
        by position AND fits the second by company — claimed twice = ambiguous."""
        td = {"summary": "S", "experience": [
            {"title": "Producer", "company": "", "description": "a1\na2\na3"},
            {"title": "Producer", "company": "Lanternfish Media Collective", "description": "l1\nl2\nl3"}]}
        out = t._apply_revised_json(td, {"experience": [
            {"title": "Producer", "company": "Lanternfish Media Collective", "bullets": ["L1", "L2", "L3"]},
            {"title": "Somebody Else", "bullets": ["Z1"]}]})
        assert out["experience"] == td["experience"]

    def test_bullets_given_as_one_string_are_split_into_lines(self):
        td = _single_entry_layout()
        out = t._apply_revised_json(td, {"experience": [
            {"title": "Editor", "bullets": "- C1\n- C2"}]})
        assert out["experience"][2]["description"] == "- C1\n- C2"

    def test_floor_protects_the_first_role_with_bullets(self, base):
        data = t.parse_docx_to_structure(base)
        assert not data["experience"][0]["description"]          # index 0 = empty placeholder
        assert t._min_bullets_for_role(1, data["experience"]) == 3
        assert t._min_bullets_for_role(3, data["experience"]) == 2
        assert t._min_bullets_for_role(0) == 3                   # no list → old rule
        revised = json.loads(t._tailored_data_to_json(data))
        revised["experience"][1]["bullets"] = revised["experience"][1]["bullets"][:2]   # 5 → 2 (< 3)
        revised["experience"][5]["bullets"] = revised["experience"][5]["bullets"][:2]   # 4 → 2 (ok)
        out = t._apply_revised_json(data, revised)
        assert out["experience"][1]["description"] == data["experience"][1]["description"]
        assert len(out["experience"][5]["description"].split("\n")) == 2


# ------------------------------------------------------------------ trim ----

def _llm(answer):
    def fake(prompt, model=None, **kw):
        return json.dumps(answer)
    return fake


JOB = {"title": "Producer", "company": "Acme", "description": "d"}


class TestTrimDeletionOnly:
    def _data(self, base):
        return t.parse_docx_to_structure(base)

    def _answer(self, data):
        return json.loads(t._tailored_data_to_json(data))

    def test_pure_deletion_is_accepted(self, base):
        data = self._data(base)
        answer = self._answer(data)
        del answer["experience"][1]["bullets"][3]      # 5 → 4
        del answer["experience"][5]["bullets"][0]      # 4 → 3
        with patch("job_agent.cv_tailor.call_llm", side_effect=_llm(answer)):
            out = t.trim_cv_to_length(data, JOB, 2.3)
        assert out is not None
        before1 = data["experience"][1]["description"].split("\n")
        assert out["experience"][1]["description"].split("\n") == before1[:3] + before1[4:]
        assert out["experience"][5]["description"].split("\n") == data["experience"][5]["description"].split("\n")[1:]
        for i in (0, 2, 3, 4, 6, 7):
            assert out["experience"][i] == data["experience"][i]
        assert out["summary"] == data["summary"]
        assert t._trim_violations(data, out) == []

    def test_rewrite_is_rejected(self, base):
        data = self._data(base)
        answer = self._answer(data)
        del answer["experience"][1]["bullets"][3]
        answer["experience"][5]["bullets"][1] = answer["experience"][5]["bullets"][1].replace("30", "300")
        with patch("job_agent.cv_tailor.call_llm", side_effect=_llm(answer)):
            assert t.trim_cv_to_length(data, JOB, 2.3) is None

    def test_tightened_bullet_is_rejected(self, base):
        data = self._data(base)
        answer = self._answer(data)
        answer["experience"][1]["bullets"][0] = answer["experience"][1]["bullets"][0][:20]
        with patch("job_agent.cv_tailor.call_llm", side_effect=_llm(answer)):
            assert t.trim_cv_to_length(data, JOB, 2.3) is None

    def test_reordered_or_moved_bullets_are_rejected(self, base):
        data = self._data(base)
        reordered = self._answer(data)
        reordered["experience"][1]["bullets"] = reordered["experience"][1]["bullets"][::-1][:4]
        moved = self._answer(data)
        moved["experience"][3]["bullets"][0] = moved["experience"][1]["bullets"].pop(0)
        for answer in (reordered, moved):
            with patch("job_agent.cv_tailor.call_llm", side_effect=_llm(answer)):
                assert t.trim_cv_to_length(data, JOB, 2.3) is None

    def test_summary_is_never_changed_by_a_trim(self, base):
        data = self._data(base)
        answer = self._answer(data)
        answer["summary"] = "A shorter, REWRITTEN summary."
        del answer["experience"][1]["bullets"][4]
        with patch("job_agent.cv_tailor.call_llm", side_effect=_llm(answer)):
            out = t.trim_cv_to_length(data, JOB, 2.3)
        assert out["summary"] == data["summary"]
        assert len(out["experience"][1]["description"].split("\n")) == 4

    def test_first_real_role_keeps_its_floor(self, base):
        """Index 0 is an empty placeholder; the 3-bullet floor must protect the
        first role that HAS bullets (index 1)."""
        data = self._data(base)
        answer = self._answer(data)
        answer["experience"][1]["bullets"] = answer["experience"][1]["bullets"][:2]    # 5 → 2: below floor
        del answer["experience"][5]["bullets"][3]                                      # 4 → 3: fine
        with patch("job_agent.cv_tailor.call_llm", side_effect=_llm(answer)):
            out = t.trim_cv_to_length(data, JOB, 2.3)
        assert out["experience"][1]["description"] == data["experience"][1]["description"]
        assert len(out["experience"][5]["description"].split("\n")) == 3

    def test_trim_that_removes_nothing_returns_none(self, base):
        data = self._data(base)
        with patch("job_agent.cv_tailor.call_llm", side_effect=_llm(self._answer(data))):
            assert t.trim_cv_to_length(data, JOB, 2.3) is None

    def test_survivors_are_the_input_lines_byte_for_byte(self):
        td = {"summary": "S", "experience": [
            {"title": "Producer", "company": "Tidewater", "description": "- keep one  \n\n- drop me\n-  keep two\n- keep three"}]}
        answer = {"experience": [{"title": "Producer", "bullets": ["- keep one", "-  keep two", "- keep three"]}]}
        with patch("job_agent.cv_tailor.call_llm", side_effect=_llm(answer)):
            out = t.trim_cv_to_length(td, JOB, 2.3)
        assert out["experience"][0]["description"] == "- keep one  \n-  keep two\n- keep three"

    def test_violations_helper(self):
        before = _single_entry_layout()
        after = copy.deepcopy(before)
        after["experience"][0]["description"] = "t1\nt3"
        assert t._trim_violations(before, after) == []
        after["experience"][0]["description"] = "t1\nt3 reworded"
        assert t._trim_violations(before, after)
        swapped = copy.deepcopy(before)
        swapped["experience"][0], swapped["experience"][1] = swapped["experience"][1], swapped["experience"][0]
        assert t._trim_violations(before, swapped)
        assert t._trim_violations(before, dict(before, summary="other"))
        assert t._trim_violations(before, dict(before, experience=before["experience"][:2]))

    def test_trim_prompt_is_deletion_only(self):
        assert "DELETION ONLY" in t._PROMPT_TRIM_CV
        assert "tighten" not in t._PROMPT_TRIM_CV.lower()


# ------------------------------------------------------------- roundtrip ----

class TestVerifyDocxRoundtrip:
    def _written(self, base, tmp_path):
        data = t.parse_docx_to_structure(base)
        tailored, expected = _with_sentinels(data, [6, 2, 4, 3])
        out = str(tmp_path / "out.docx")
        t.fill_docx_with_tailored_data(base, out, copy.deepcopy(tailored))
        return data, tailored, expected, out

    def test_ok_on_a_correct_write(self, base, tmp_path):
        _, tailored, _, out = self._written(base, tmp_path)
        assert t.verify_docx_roundtrip(out, tailored) == {"ok": True, "mismatches": []}
        assert t.verify_docx_roundtrip(out, tailored, base_docx_path=base) == {"ok": True, "mismatches": []}

    def test_reports_an_altered_bullet(self, base, tmp_path):
        _, tailored, expected, out = self._written(base, tmp_path)
        doc = Document(out)
        target = expected[3][1]
        hit = next(p for p in doc.paragraphs if p.text == target)
        hit.runs[0].text = target.replace("this role only", "this role and some more")
        doc.save(out)
        result = t.verify_docx_roundtrip(out, tailored)
        assert result["ok"] is False
        assert [(m["where"], m["index"], m["reason"]) for m in result["mismatches"]] == \
            [("experience", 3, "bullets differ")]
        assert result["mismatches"][0]["title"] == ROLES[1][1]

    def test_reports_base_bullets_shipped_instead_of_tailored(self, base, tmp_path):
        """The 8-month bug as the checker sees it: file holds base bullets,
        intent was tailored bullets → one mismatch per real role."""
        data = t.parse_docx_to_structure(base)
        tailored, _ = _with_sentinels(data, [3])
        out = str(tmp_path / "out.docx")
        t.fill_docx_with_tailored_data(base, out, dict(data, summary=tailored["summary"]))
        result = t.verify_docx_roundtrip(out, tailored)
        assert result["ok"] is False
        assert [m["index"] for m in result["mismatches"]] == [1, 3, 5, 7]

    def test_reports_summary_and_missing_bullet_and_extra_bullet(self, base, tmp_path):
        _, tailored, _, out = self._written(base, tmp_path)
        wrong = copy.deepcopy(tailored)
        wrong["summary"] = "Another summary."
        wrong["experience"][1]["description"] += "\n- one more bullet"
        wrong["experience"][5]["description"] = "\n".join(wrong["experience"][5]["description"].split("\n")[1:])
        result = t.verify_docx_roundtrip(out, wrong)
        assert [(m["where"], m.get("index")) for m in result["mismatches"]] == \
            [("summary", None), ("experience", 1), ("experience", 5)]

    def test_whitespace_and_bullet_markers_are_normalised(self, base, tmp_path):
        _, tailored, _, out = self._written(base, tmp_path)
        loose = copy.deepcopy(tailored)
        loose["summary"] = "  " + loose["summary"].replace(" ", "   ") + "\n\n"
        loose["experience"][1]["description"] = "\n".join(
            "•   " + b.lstrip("- ").replace(" ", "  ") for b in loose["experience"][1]["description"].split("\n"))
        assert t.verify_docx_roundtrip(out, loose)["ok"] is True

    def test_bullets_intended_for_a_placeholder_are_a_mismatch(self, base, tmp_path):
        _, tailored, _, out = self._written(base, tmp_path)
        tailored["experience"][0]["description"] = "- should never be written"
        result = t.verify_docx_roundtrip(out, tailored)
        assert result["ok"] is False and result["mismatches"][0]["index"] == 0

    def test_role_count_and_header_differences(self, base, tmp_path):
        _, tailored, _, out = self._written(base, tmp_path)
        fewer = dict(tailored, experience=tailored["experience"][:-2])
        assert any(m["reason"] == "role count differs" for m in t.verify_docx_roundtrip(out, fewer)["mismatches"])
        renamed = copy.deepcopy(tailored)
        renamed["experience"][3]["title"] = "Somebody Else Inc | Remote | 2020 – 2023"
        assert any(m["reason"] == "header line differs" and m["index"] == 3
                   for m in t.verify_docx_roundtrip(out, renamed)["mismatches"])

    def test_static_section_change_is_caught_with_the_base(self, base, tmp_path):
        _, tailored, _, out = self._written(base, tmp_path)
        doc = Document(out)
        skills = next(p for p in doc.paragraphs if p.text.startswith("Post: edit supervision"))
        skills.runs[0].text = "Post: something else entirely."
        doc.save(out)
        assert t.verify_docx_roundtrip(out, tailored)["ok"] is True      # not checked without the base
        result = t.verify_docx_roundtrip(out, tailored, base_docx_path=base)
        assert result["ok"] is False and result["mismatches"][0]["where"] == "static"

    def test_unreadable_file_is_never_ok(self, tmp_path):
        tailored = {"summary": "", "experience": []}
        assert t.verify_docx_roundtrip(str(tmp_path / "missing.docx"), tailored)["ok"] is False
        garbage = tmp_path / "garbage.docx"
        garbage.write_bytes(b"not a zip")
        result = t.verify_docx_roundtrip(str(garbage), tailored)
        assert result["ok"] is False and result["mismatches"][0]["where"] == "file"
