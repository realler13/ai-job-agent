"""Build a base-CV .docx from the committed FICTIONAL career (fictional_career.md).

A deployment's base CVs are private and gitignored, so the unmocked writer
tests need a stand-in with the same topology: Normal-style header / title / company lines
and 'List Bullet' bullets. Usage:

    from tests.fixtures.fictional_cv import build_fixture_docx, load_fictional_career
    base = build_fixture_docx(tmp_path / "base.docx")               # domain-base layout
    base = build_fixture_docx(tmp_path / "old.docx", legacy=True)   # base_cv.docx layout

legacy=True mirrors a hand-made legacy base_cv.docx: every bullet carries its own
w:numPr (a different numId per role), the first and last bullet of a block carry
w:spacing, one role's bullets carry w:ind, one bullet is split over three runs,
an empty spacer paragraph follows the summary, and every paragraph has the
w14:paraId / w14:textId attributes Word writes.
"""

from pathlib import Path

from docx import Document
from docx.shared import Pt

CAREER_MD = Path(__file__).with_name("fictional_career.md")

# numIds per bullet block in legacy mode (role 1..4, then the static sections).
_LEGACY_ROLE_NUM_IDS = [19, 21, 22, 23]
_LEGACY_STATIC_NUM_IDS = {"VOLUNTEER EXPERIENCE": 24, "SKILLS": 25, "EDUCATION": 26}
W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
PARA_ID = "{%s}paraId" % W14
TEXT_ID = "{%s}textId" % W14


def load_fictional_career() -> dict:
    """Parse fictional_career.md into
    {"name", "contact", "sections": [(header, [("para"|"title"|"line"|"bullet", text), ...]), ...]}."""
    career = {"name": "", "contact": "", "sections": []}
    in_comment = False
    for raw in CAREER_MD.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("<!--"):
            in_comment = True
        if in_comment:
            in_comment = "-->" not in line
            continue
        if not line:
            continue
        if line.startswith("# "):
            career["name"] = line[2:].strip()
        elif line.startswith("## "):
            career["sections"].append((line[3:].strip(), []))
        elif not career["sections"]:
            career["contact"] = line
        elif line.startswith("### "):
            career["sections"][-1][1].append(("title", line[4:].strip()))
        elif line.startswith("- "):
            career["sections"][-1][1].append(("bullet", line[2:].strip()))
        else:
            kind = "line" if career["sections"][-1][0] == "EXPERIENCE" else "para"
            career["sections"][-1][1].append((kind, line))
    return career


def _bold(paragraph, text, size=None):
    run = paragraph.add_run(text)
    run.bold = True
    if size:
        run.font.size = Pt(size)
    return run


def _set_num(paragraph, num_id):
    num_pr = paragraph._p.get_or_add_pPr().get_or_add_numPr()
    num_pr.get_or_add_ilvl().val = 0
    num_pr.get_or_add_numId().val = num_id


def build_fixture_docx(path, legacy: bool = False) -> str:
    """Write the fictional base CV to *path* and return the path as str."""
    career = load_fictional_career()
    doc = Document()
    _bold(doc.add_paragraph(), career["name"].upper(), size=14)
    doc.add_paragraph(career["contact"])

    role_no = -1
    for header, items in career["sections"]:
        _bold(doc.add_paragraph(), header, size=12)
        block = []  # the bullet paragraphs of the current block

        def _close_block():
            # legacy: the first and last bullet of a block carry spacing
            if legacy and block:
                block[0].paragraph_format.space_before = Pt(2)
                block[-1].paragraph_format.space_after = Pt(6)
            block.clear()

        for kind, text in items:
            if kind == "para":
                doc.add_paragraph(text)
            elif kind == "title":
                _close_block()
                role_no += 1
                _bold(doc.add_paragraph(), text)
            elif kind == "line":
                doc.add_paragraph(text)
            else:
                p = doc.add_paragraph(style="List Bullet")
                if legacy and header == "EXPERIENCE" and role_no == 0 and len(block) == 1:
                    # one multi-run bullet (bold fragment in the middle)
                    head, _, tail = text.partition(" ")
                    p.add_run(head + " ")
                    p.add_run(tail[:12]).bold = True
                    p.add_run(tail[12:])
                else:
                    p.add_run(text)
                if legacy:
                    num_id = (_LEGACY_ROLE_NUM_IDS[role_no] if header == "EXPERIENCE"
                              else _LEGACY_STATIC_NUM_IDS.get(header, 30))
                    _set_num(p, num_id)
                    if header == "EXPERIENCE" and role_no == 3:
                        p.paragraph_format.left_indent = Pt(54)
                block.append(p)
        _close_block()
        if legacy and header == "PROFESSIONAL SUMMARY":
            doc.add_paragraph()  # empty spacer between the summary and EXPERIENCE

    if legacy:
        for i, p in enumerate(doc.paragraphs):
            p._p.set(PARA_ID, f"{0x1A000000 + i:08X}")
            p._p.set(TEXT_ID, "77777777")

    doc.save(str(path))
    return str(path)
