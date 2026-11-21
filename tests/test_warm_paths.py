"""Warm Paths (Spec M3): LinkedIn CSV parsing, normalization, matching."""

import pytest

from job_agent import warm_paths as wp


CSV_WITH_PREAMBLE = '''Notes:
"When exporting your connection data, you may notice that some of the email addresses are missing."

First Name,Last Name,URL,Email Address,Company,Position,Connected On
Jane,Doe,https://www.linkedin.com/in/janedoe,,Apple,"Producer, Brand Studio",01 Jan 2025
Bob,Roe,https://www.linkedin.com/in/bobroe,bob@x.com,"Acme, Inc.",Senior Editor,01 Jan 2025
NoCompany,Person,https://www.linkedin.com/in/nc,,,Freelancer,02 Feb 2025
'''


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "referrer-connections.csv"
    p.write_text(CSV_WITH_PREAMBLE, encoding="utf-8")
    return str(p)


@pytest.fixture
def db(tmp_path):
    from job_agent.db_manager import DBManager
    return DBManager(db_path=str(tmp_path / "wp.db"))


class TestParser:
    def test_skips_notes_preamble(self, csv_file):
        rows = wp.parse_connections_csv(csv_file)
        assert len(rows) == 3
        assert rows[0]["name"] == "Jane Doe"

    def test_quoted_commas_survive(self, csv_file):
        rows = wp.parse_connections_csv(csv_file)
        assert rows[0]["position"] == "Producer, Brand Studio"
        assert rows[1]["company"] == "Acme, Inc."

    def test_utf8_bom_tolerated(self, tmp_path):
        p = tmp_path / "bom.csv"
        p.write_bytes(b"\xef\xbb\xbf" + CSV_WITH_PREAMBLE.encode())
        rows = wp.parse_connections_csv(str(p))
        assert len(rows) == 3

    def test_no_preamble_export_still_parses(self, tmp_path):
        # older exports have no Notes: block
        p = tmp_path / "old.csv"
        p.write_text("\n".join(CSV_WITH_PREAMBLE.splitlines()[3:]), encoding="utf-8")
        rows = wp.parse_connections_csv(str(p))
        assert len(rows) == 3

    def test_missing_header_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("this,is,not,a,linkedin,export\n1,2,3,4,5,6\n")
        with pytest.raises(ValueError):
            wp.parse_connections_csv(str(p))


class TestNormalizer:
    @pytest.mark.parametrize("raw,expected", [
        ("Apple", "apple"),
        ("Apple Inc.", "apple"),
        ("Acme, Inc.", "acme"),
        ("Fernbrook.com", "fernbrook"),
        ("Saltmarsh Bros. Pictures", "saltmarsh bros pictures"),
        ("Northwind", "northwind"),
        ("The Kestrel Picture Company", "the kestrel picture"),
        ("", ""),
    ])
    def test_normalize(self, raw, expected):
        assert wp.normalize_company(raw) == expected

    # Phase 27: the decorated company strings the scrapers emit (contingent and
    # staffing listings), plus the other qualifier shapes job boards and
    # LinkedIn profiles use. A decorated string must reach the plain employer.
    @pytest.mark.parametrize("raw,expected", [
        ("Apple (via WillHire)", "apple"),
        ("Apple (Contractor)", "apple"),
        ("Apple Inc. (via WillHire)", "apple"),
        ("Aquent (staffing)", "aquent"),
        ("Northwind via Talentry (formerly StaffPro Partners)", "northwind"),
        ("Brightfern (formerly Fernlight)", "brightfern"),
        ("Orb (Formerly OrbitMail)", "orb"),
        ("Lantern Forge (formerly Copperleaf Pictures)", "lantern forge"),
        ("MARLOWE | The Brandhouse Group", "marlowe"),
        ("Harbor Lights Media | HLM", "harbor lights media"),
        ("KESTREL Architecture | Planning | Interiors", "kestrel architecture"),
        ("Driftwood (driftwood.agency)", "driftwood"),
        ("Quokka (quokka.app)", "quokka"),
        ("Haven (madeinhaven.com)", "haven"),
        ("Jobs via Dice", "jobs"),
    ])
    def test_trailing_qualifiers_stripped(self, raw, expected):
        assert wp.normalize_company(raw) == expected

    # ...and the look-alikes that must NOT be cut: a closed list of qualifiers,
    # never "drop any parenthetical".
    @pytest.mark.parametrize("raw,expected", [
        ("Applebee's", "applebee s"),
        ("Pineapple Media", "pineapple media"),
        ("Casa Via Wellness Center", "casa via wellness center"),
        ("Via", "via"),
        ("Meridian (Meridian Surgical)", "meridian meridian surgical"),
        ("Northwind Cloud Services (NCS)", "northwind cloud services ncs"),
        ("NORTHWIND (UK) LIMITED", "northwind uk"),
        ("Kestrel Semiconductor Inc (US)", "kestrel semiconductor us"),
        ("(via WillHire)", "via willhire"),          # only a qualifier: left alone
    ])
    def test_non_qualifiers_kept(self, raw, expected):
        assert wp.normalize_company(raw) == expected

    def test_normalizing_twice_changes_nothing(self):
        for raw in ("Apple (via WillHire)", "Northwind via Talentry (formerly StaffPro Partners)",
                    "MARLOWE | The Brandhouse Group"):
            once = wp.normalize_company(raw)
            assert wp.normalize_company(once) == once


class TestImport:
    def test_import_and_idempotent_reimport(self, db, csv_file):
        r1 = wp.import_connections(db, csv_file, owner="referrer")
        assert r1["inserted"] == 3 and r1["updated"] == 0
        r2 = wp.import_connections(db, csv_file, owner="referrer")
        assert r2["inserted"] == 0 and r2["updated"] == 3  # upsert on URL

    def test_owner_separation(self, db, csv_file):
        wp.import_connections(db, csv_file, owner="referrer")
        wp.import_connections(db, csv_file, owner="candidate")
        conn = db._get_thread_conn()
        counts = dict(conn.execute(
            "SELECT owner, COUNT(*) FROM warm_connections GROUP BY owner").fetchall())
        assert counts == {"referrer": 3, "candidate": 3}


class TestMatching:
    def test_exact_match_after_normalization(self, db, csv_file):
        wp.import_connections(db, csv_file, owner="referrer")
        found = wp.find_warm_paths(db, "Apple Inc.")   # suffix differs → still exact
        assert len(found["matches"]) == 1
        assert found["matches"][0]["name"] == "Jane Doe"
        assert found["matches"][0]["matched_via"] == "exact"

    def test_suffix_variants_match_exact(self, db, csv_file):
        wp.import_connections(db, csv_file, owner="referrer")
        found = wp.find_warm_paths(db, "Acme Incorporated")  # ≡ "Acme, Inc."
        assert found["matches"]
        assert found["matches"][0]["matched_via"] == "exact"

    def test_prefix_rule_catches_subsidiary(self, db, tmp_path):
        p = tmp_path / "s-connections.csv"
        p.write_text(
            "First Name,Last Name,URL,Email Address,Company,Position,Connected On\n"
            "Sub,Sidiary,https://l/in/s,,Apple Vietnam,Producer,01 Jan 2025\n")
        wp.import_connections(db, str(p), owner="referrer")
        found = wp.find_warm_paths(db, "Apple")
        assert found["matches"] and found["matches"][0]["matched_via"] == "prefix"

    def test_subset_trap_goes_to_manual_confirm_not_matches(self, db, tmp_path):
        # token_set_ratio scores 100 on ANY word-subset regardless of order —
        # the measured false-positive class. A NON-prefix subset ("Global Apple
        # Fanclub" for target "Apple") must land in `possible`, never `matches`.
        # (The leading-prefix form is auto-accepted deliberately: subsidiaries
        # like "Apple Vietnam".)
        p = tmp_path / "t-connections.csv"
        p.write_text(
            "First Name,Last Name,URL,Email Address,Company,Position,Connected On\n"
            "Joke,Entry,https://l/in/j,,Global Apple Fanclub,Commissioner,01 Jan 2025\n")
        wp.import_connections(db, str(p), owner="referrer")
        found = wp.find_warm_paths(db, "Apple")
        assert found["matches"] == []
        assert found["possible"] and found["possible"][0]["matched_via"] == "needs_confirm"

    def test_short_or_stoplisted_target_never_fuzzy(self, db, tmp_path):
        p = tmp_path / "g-connections.csv"
        p.write_text(
            "First Name,Last Name,URL,Email Address,Company,Position,Connected On\n"
            "Some,One,https://l/in/o,,Confidential Media Team,Producer,01 Jan 2025\n")
        wp.import_connections(db, str(p), owner="referrer")
        # 'confidential' is a job-board anonymization string, not a company
        found = wp.find_warm_paths(db, "Confidential")
        assert found["matches"] == [] and found["possible"] == []

    def test_junk_csv_company_skipped(self, db, tmp_path):
        p = tmp_path / "k-connections.csv"
        p.write_text(
            "First Name,Last Name,URL,Email Address,Company,Position,Connected On\n"
            'Author,Person,https://l/in/a,,"Available on Amazon, Barnes & Noble and other stores",Writer,01 Jan 2025\n')
        wp.import_connections(db, str(p), owner="referrer")
        found = wp.find_warm_paths(db, "Amazon")
        assert found["matches"] == []

    def test_different_company_does_not_match(self, db, csv_file):
        wp.import_connections(db, csv_file, owner="referrer")
        found = wp.find_warm_paths(db, "Netflix")
        assert found["matches"] == []

    def test_empty_company(self, db):
        assert wp.find_warm_paths(db, "") == {"matches": [], "possible": []}

    def test_relevance_ranking(self, db, tmp_path):
        p = tmp_path / "x-connections.csv"
        p.write_text(
            "First Name,Last Name,URL,Email Address,Company,Position,Connected On\n"
            "Eng,Ineer,https://l/in/e,,BigCo,Supply Chain Engineer,01 Jan 2025\n"
            "Prod,Ucer,https://l/in/p,,BigCo,Senior Video Producer,01 Jan 2025\n"
            "Rec,Ruiter,https://l/in/r,,BigCo,Creative Recruiter,01 Jan 2025\n")
        wp.import_connections(db, str(p), owner="referrer")
        found = wp.find_warm_paths(db, "BigCo")
        names = [m["name"] for m in found["matches"]]
        # producer + recruiter must outrank the engineer
        assert names.index("Eng Ineer") == 2

    def test_her_own_connections_rank_first(self, db, tmp_path):
        """At a big employer a second imported network can be large; the people the
        candidate can message herself must not fall outside a capped list."""
        d = tmp_path / "referrer-connections.csv"
        d.write_text(
            "First Name,Last Name,URL,Email Address,Company,Position,Connected On\n"
            + "".join(f"Net{i},Contact,https://l/in/d{i},,BigCo,Senior Video Producer,01 Jan 2025\n"
                      for i in range(60)))
        o = tmp_path / "candidate-connections.csv"
        o.write_text(
            "First Name,Last Name,URL,Email Address,Company,Position,Connected On\n"
            "Her,Friend,https://l/in/o1,,BigCo,Supply Chain Engineer,01 Jan 2025\n")
        wp.import_connections(db, str(d), owner="referrer")
        wp.import_connections(db, str(o), owner="candidate")
        found = wp.find_warm_paths(db, "BigCo")
        assert found["matches"][0]["owner"] == "candidate"
        assert len(found["matches"]) == 61

    def test_summary_line(self, db, csv_file):
        wp.import_connections(db, csv_file, owner="referrer")
        s = wp.warm_paths_summary(db, "Apple")
        assert s.startswith("1 warm path")
        assert "Jane Doe" in s
        assert wp.warm_paths_summary(db, "Nonexistent Co") == ""


# ── Phase 27: qualifier stripping on both sides ──────────────────────────────

_HEADER = "First Name,Last Name,URL,Email Address,Company,Position,Connected On\n"


def _import_rows(db, tmp_path, rows, owner="candidate"):
    p = tmp_path / f"{owner}-connections.csv"
    p.write_text(_HEADER + "".join(rows), encoding="utf-8")
    return wp.import_connections(db, str(p), owner=owner)


class TestQualifierMatching:
    APPLE_NETWORK = [
        "Ann,Exact,https://l/in/ann,,Apple,Producer,01 Jan 2025\n",
        "Ben,Suffix,https://l/in/ben,,Apple Inc.,Video Editor,01 Jan 2025\n",
        "Cat,Division,https://l/in/cat,,Apple Vietnam,Recruiter,01 Jan 2025\n",
        "Dan,Diner,https://l/in/dan,,Applebee's,Server,01 Jan 2025\n",
        "Eve,Fruit,https://l/in/eve,,Pineapple Media,Producer,01 Jan 2025\n",
        "Fay,Vendor,https://l/in/fay,,WillHire,Account Manager,01 Jan 2025\n",
        "Gus,Fan,https://l/in/gus,,Global Apple Fanclub,Commissioner,01 Jan 2025\n",
    ]

    @pytest.mark.parametrize("job_company", ["Apple (via WillHire)", "Apple (Contractor)"])
    def test_apple_contingent_strings_reach_apple_connections(self, db, tmp_path, job_company):
        _import_rows(db, tmp_path, self.APPLE_NETWORK)
        found = wp.find_warm_paths(db, job_company)
        via = {m["name"]: m["matched_via"] for m in found["matches"]}
        assert via == {"Ann Exact": "exact", "Ben Suffix": "exact", "Cat Division": "prefix"}

    def test_lookalikes_and_the_staffing_vendor_never_merge(self, db, tmp_path):
        _import_rows(db, tmp_path, self.APPLE_NETWORK)
        names = {m["name"] for m in wp.find_warm_paths(db, "Apple (via WillHire)")["matches"]}
        assert not names & {"Dan Diner", "Eve Fruit", "Fay Vendor", "Gus Fan"}
        # ...and the other way round: an Applebee's job is not an Apple job.
        assert [m["name"] for m in wp.find_warm_paths(db, "Applebee's")["matches"]] == ["Dan Diner"]

    def test_fuzzy_still_never_auto_accepted(self, db, tmp_path):
        _import_rows(db, tmp_path, self.APPLE_NETWORK)
        found = wp.find_warm_paths(db, "Apple (via WillHire)")
        possible = {m["name"]: m["matched_via"] for m in found["possible"]}
        assert possible.get("Gus Fan") == "needs_confirm"     # subset trap stays manual

    def test_connection_side_qualifier_is_stripped_too(self, db, tmp_path):
        _import_rows(db, tmp_path, [
            "Hal,Temp,https://l/in/hal,,Northwind via Talentry (formerly StaffPro Partners),Producer,01 Jan 2025\n",
            "Ivy,Tagline,https://l/in/ivy,,MARLOWE | The Brandhouse Group,Producer,01 Jan 2025\n",
        ])
        assert [m["name"] for m in wp.find_warm_paths(db, "Northwind")["matches"]] == ["Hal Temp"]
        assert [m["name"] for m in wp.find_warm_paths(db, "MARLOWE")["matches"]] == ["Ivy Tagline"]
        assert wp.find_warm_paths(db, "Talentry")["matches"] == []   # the vendor is not the employer

    def test_job_board_anonymizer_matches_nobody(self, db, tmp_path):
        _import_rows(db, tmp_path, [
            "Jon,Board,https://l/in/jon,,Jobs Network Ltd,Recruiter,01 Jan 2025\n",
            "Kim,Dice,https://l/in/kim,,Dice,Recruiter,01 Jan 2025\n",
        ])
        found = wp.find_warm_paths(db, "Jobs via Dice")
        assert found == {"matches": [], "possible": []}

    def test_reimport_refreshes_stored_norms(self, db, tmp_path):
        """company_norm is stored at import time — the post-deploy re-import
        (idempotent) is what brings old rows up to the new normalizer. A short
        name shows it: "kta" is under the 4-char prefix anchor, so the stale
        stored norm cannot match until the row is re-imported."""
        rows = ["Wil,Agent,https://l/in/wil,,KTA | Kestrel Talent Agency,Agent,01 Jan 2025\n"]
        _import_rows(db, tmp_path, rows)
        conn = db._get_thread_conn()
        conn.execute("UPDATE warm_connections SET company_norm = 'kta kestrel talent agency'")
        conn.commit()
        assert wp.find_warm_paths(db, "KTA")["matches"] == []          # stale stored norm
        result = _import_rows(db, tmp_path, rows)
        assert result["inserted"] == 0 and result["updated"] == 1
        found = wp.find_warm_paths(db, "KTA")["matches"]
        assert [(m["name"], m["matched_via"]) for m in found] == [("Wil Agent", "exact")]
