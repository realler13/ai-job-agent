#!/usr/bin/env python3
"""Create 8 domain-specific base CV .docx files from base-cv-variants.md.

The markdown file is yours (gitignored): one "## <n>. <domain>" section per
variant, each holding a plain-text CV whose first line is the profile's name
in capitals and whose second line is the contact line with the profile's email.

Run from the project root:

    PYTHONPATH=src .venv/bin/python scripts/create_base_cvs.py

Formatting:
  - Helvetica font, 0.5in margins
  - Name paragraph: keepNext, spacing before=200twips after=40twips
  - Contact info: keepNext, spacing after=40twips
  - Section headers: bold 12pt, spacing after=0
  - Role titles: bold, spacing after=0
  - Company lines: spacing after=2pt (25400 EMU)
  - Bullets: 'List Bullet' style
  - Summary text: Normal style
"""

import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, Emu, Twips
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from job_agent.candidate_profile import CANDIDATE  # noqa: E402

# Map variant number -> (section_title, output_filename)
VARIANTS = {
    1: ("producer_core", "base_cv_producer_core.docx"),
    2: ("live_and_experiential", "base_cv_live_experiential.docx"),
    3: ("production_ops_and_management", "base_cv_production_ops.docx"),
    4: ("project_and_program_management", "base_cv_project_program.docx"),
    5: ("podcast_and_audio", "base_cv_podcast_audio.docx"),
    6: ("youtube_and_creators", "base_cv_youtube_creators.docx"),
    7: ("games_production", "base_cv_games_production.docx"),
    8: ("production_admin", "base_cv_production_admin.docx"),
}


_NAME_LINE = CANDIDATE["name"].upper()
# The contact line is recognised by the profile's email address.
_EMAIL = str(CANDIDATE.get("email") or "").strip()


def parse_variants(md_path: Path) -> dict[int, list[str]]:
    """Parse base-cv-variants.md and return a dict of variant_number -> lines."""
    text = md_path.read_text(encoding="utf-8")

    # Split by variant headers: ## 1. producer_core, ## 2. live_and_experiential, etc.
    # Each variant block starts after the second --- (first --- is after the **File:** line)
    variants = {}
    # Find all variant sections
    pattern = r"^## (\d+)\. "
    parts = re.split(r"^## \d+\. .+$", text, flags=re.MULTILINE)

    # Find variant numbers
    numbers = [int(m.group(1)) for m in re.finditer(pattern, text, re.MULTILINE)]

    for num, part in zip(numbers, parts[1:]):
        # Strip the **File:** and **Matches:** lines and the --- delimiters
        lines = part.strip().split("\n")
        # Find the content start: after the second --- or after the blank line following ---
        content_start = 0
        dashes_seen = 0
        for i, line in enumerate(lines):
            if line.strip() == "---":
                dashes_seen += 1
                if dashes_seen >= 1:
                    content_start = i + 1
                    break

        # Also skip any **File:** or **Matches:** lines before ---
        content_lines = lines[content_start:]

        # Strip leading/trailing blank lines
        while content_lines and not content_lines[0].strip():
            content_lines.pop(0)

        # Truncate at trailing --- or ## heading (next section / code mapping block)
        # Scan forward from the end of EDUCATION bullets to find the boundary
        end_idx = len(content_lines)
        for j, cl in enumerate(content_lines):
            stripped = cl.strip()
            if stripped == "---" or stripped.startswith("## "):
                # Only treat as boundary if it appears after actual CV content
                # (skip the initial --- which is handled by content_start)
                if j > 5:
                    end_idx = j
                    break
        content_lines = content_lines[:end_idx]

        # Strip trailing blank lines
        while content_lines and not content_lines[-1].strip():
            content_lines.pop()

        variants[num] = content_lines

    return variants


def set_keepnext(paragraph):
    """Set keepNext on a paragraph (keeps paragraph with next one on same page)."""
    pPr = paragraph._element.get_or_add_pPr()
    kn = pPr.find(qn("w:keepNext"))
    if kn is None:
        kn = paragraph._element.makeelement(qn("w:keepNext"), {})
        pPr.insert(0, kn)


def create_cv_docx(lines: list[str], output_path: Path):
    """Create a .docx file from parsed CV content lines, matching base_cv.docx style."""
    doc = Document()

    # --- Set up document defaults ---
    # Set Normal style font to Helvetica
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Helvetica"

    # Set margins to 0.5in
    section = doc.sections[0]
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)
    section.top_margin = Inches(0.5)
    section.bottom_margin = Inches(0.5)

    # Remove the default empty paragraph
    if doc.paragraphs:
        p = doc.paragraphs[0]._element
        p.getparent().remove(p)

    # --- Parse lines into structured content ---
    i = 0
    section_headers = {"PROFESSIONAL SUMMARY", "SUMMARY", "EXPERIENCE", "PROJECTS",
                       "VOLUNTEER EXPERIENCE", "SKILLS", "EDUCATION", "CERTIFICATIONS"}

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Name line — the profile's name, upper-cased, marks the header
        if line == _NAME_LINE:
            p = doc.add_paragraph()
            p.style = normal_style
            set_keepnext(p)
            # Spacing: before=200twips (127000 EMU), after=40twips (25400 EMU)
            pf = p.paragraph_format
            pf.space_before = Emu(127000)
            pf.space_after = Emu(25400)
            run = p.add_run(line)
            i += 1
            continue

        # Contact info line (contains email)
        if _EMAIL and _EMAIL in line:
            # Replace | with bullet separator to match existing CV style
            contact = line.replace(" | ", " \u2022 ")
            p = doc.add_paragraph()
            p.style = normal_style
            set_keepnext(p)
            pf = p.paragraph_format
            pf.space_after = Emu(25400)
            run = p.add_run(contact)
            i += 1
            continue

        # Section headers
        if line in section_headers:
            p = doc.add_paragraph()
            p.style = normal_style
            pf = p.paragraph_format
            pf.space_after = Emu(0)
            run = p.add_run(line)
            run.bold = True
            run.font.size = Pt(12)
            i += 1
            continue

        # Bullet points (lines starting with -)
        if line.startswith("- "):
            bullet_text = line[2:]
            p = doc.add_paragraph(bullet_text, style="List Bullet")
            i += 1
            continue

        # Check if this is a role title (bold line, followed by company line)
        # Role titles: non-bullet, non-section-header, non-name, non-contact
        # They appear right before company/location/date lines
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Company lines: Company | Location | Date — the date leg may start
            # with a year ("2021–Present") or a month name ("April 2021 –
            # August 2025").
            is_company_line = bool(
                re.match(
                    r".+\s*\|\s*.+\s*\|\s*(?:\d{4}|[A-Z][a-z]+ \d{4})",
                    next_line,
                )
            )
            if is_company_line:
                # This is a role title — convert | to bullet separator
                role_title = line.replace(" | ", " \u2022 ")
                p = doc.add_paragraph()
                p.style = normal_style
                pf = p.paragraph_format
                pf.space_after = Emu(0)
                run = p.add_run(role_title)
                run.bold = True
                i += 1

                # Now add the company line — also convert | to bullet separator
                company_line = lines[i].strip().replace(" | ", " \u2022 ")
                p = doc.add_paragraph()
                p.style = normal_style
                pf = p.paragraph_format
                pf.space_after = Emu(25400)
                run = p.add_run(company_line)
                i += 1
                continue

        # Fallback: summary text or other content
        p = doc.add_paragraph()
        p.style = normal_style
        run = p.add_run(line)
        i += 1

    doc.save(str(output_path))
    print(f"  Created: {output_path.name}")


def main():
    # The variants file is written by hand from the career source bank; it must
    # use the profile's canonical years figure.
    md_path = PROJECT_ROOT / "base-cv-variants.md"
    if not md_path.exists():
        print(f"ERROR: {md_path} not found")
        sys.exit(1)

    print("Parsing base-cv-variants.md...")
    variants = parse_variants(md_path)

    print(f"Found {len(variants)} variants: {sorted(variants.keys())}")

    for num, (domain, filename) in VARIANTS.items():
        if num not in variants:
            print(f"  WARNING: Variant {num} ({domain}) not found in markdown")
            continue

        output_path = PROJECT_ROOT / filename
        content_lines = variants[num]
        print(f"\n  Variant {num}: {domain} ({len(content_lines)} lines)")
        create_cv_docx(content_lines, output_path)

    print("\nDone! All CV files created.")


if __name__ == "__main__":
    main()
