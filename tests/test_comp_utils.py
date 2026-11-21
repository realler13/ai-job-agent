"""Tests for comp_utils.extract_comp (per-job comp + employment type)."""
from job_agent.comp_utils import extract_comp


def test_structured_yearly_range():
    c = extract_comp({"min_amount": 120000, "max_amount": 150000,
                      "interval": "yearly", "job_type": "fulltime", "description": ""})
    assert c["comp_text"] == "$120K–$150K/yr"
    assert c["employment_type"] == "Full-time"
    assert c["comp_yearly_min"] == 120000 and c["comp_yearly_max"] == 150000


def test_structured_hourly_converts_to_yearly():
    c = extract_comp({"min_amount": 45, "max_amount": 55,
                      "interval": "hourly", "job_type": "contract", "description": ""})
    assert "/hr" in c["comp_text"] and "≈" in c["comp_text"]
    assert c["employment_type"] == "Contract"
    assert c["comp_yearly_min"] == 45 * 2080
    assert c["comp_yearly_max"] == 55 * 2080


def test_nan_strings_ignored():
    """pandas NaN survives json.dumps(default=str) as the STRING 'nan'."""
    c = extract_comp({"min_amount": "nan", "max_amount": "nan",
                      "interval": "nan", "job_type": "nan", "description": "no comp here"})
    assert c["comp_text"] is None
    assert c["employment_type"] is None


def test_description_fallback_yearly_range():
    c = extract_comp({"description": "Compensation: $130,000 - $160,000 per year DOE."})
    assert c["comp_text"] == "$130K–$160K/yr"


def test_description_fallback_k_notation():
    c = extract_comp({"description": "Salary range $95k–$110k depending on experience."})
    assert c["comp_text"] == "$95K–$110K/yr"


def test_description_fallback_hourly_single():
    c = extract_comp({"description": "Pay: $50/hour, 40 hrs/week."})
    assert c["comp_text"] is not None and "/hr" in c["comp_text"]


def test_w2_and_c2c_keywords():
    c = extract_comp({"job_type": "contract",
                      "description": "This is a W2 contract role. No C2C."})
    # Both mentioned in text; job_type contract leads
    assert c["employment_type"].startswith("Contract")
    assert "W2" in c["employment_type"]


def test_1099_keyword_alone():
    c = extract_comp({"description": "Freelance 1099 engagement."})
    assert c["employment_type"] == "1099"


def test_ambiguous_mid_range_amount_skipped():
    """$800 with no unit: could be weekly or daily — must not guess."""
    c = extract_comp({"description": "Stipend of $800 - $900 provided."})
    assert c["comp_text"] is None


def test_multi_job_type():
    c = extract_comp({"job_type": "parttime, fulltime, contract", "description": ""})
    # at most two labels, joined
    assert c["employment_type"] is not None
    assert len(c["employment_type"].split("/")) <= 2


# ── LLM fallback (call_llm mocked — hermetic) ────────────────────────────────
from unittest.mock import patch

from job_agent.comp_utils import llm_extract_comp

_DESC = "We pay competitively. Compensation discussed at interview. " * 3


def test_llm_found_monthly_case():
    ans = '{"found": true, "yearly_min": 72000, "yearly_max": 96000, "basis": "$6K-8K/mo x12", "employment_type": "Contract"}'
    with patch("job_agent.gpt_filter.call_llm", return_value=ans):
        c = llm_extract_comp("Producer", _DESC)
    assert c["comp_source"] == "llm"
    assert c["comp_text"].startswith("~$72K–$96K/yr (est.")
    assert "6K-8K/mo" in c["comp_text"]
    assert c["employment_type"] == "Contract"


def test_llm_part_time_hours_passthrough():
    ans = '{"found": true, "yearly_min": 52000, "yearly_max": 52000, "basis": "$50/hr x 20h/wk", "employment_type": null}'
    with patch("job_agent.gpt_filter.call_llm", return_value=ans):
        c = llm_extract_comp("Editor", _DESC)
    assert c["comp_yearly_min"] == 52000 and c["comp_yearly_max"] == 52000
    assert "20h/wk" in c["comp_text"]


def test_llm_not_found_returns_none():
    with patch("job_agent.gpt_filter.call_llm", return_value='{"found": false}'):
        assert llm_extract_comp("Producer", _DESC) is None


def test_llm_sanity_rails_reject_nonsense():
    ans = '{"found": true, "yearly_min": 5000000, "yearly_max": 9000000, "basis": "x", "employment_type": null}'
    with patch("job_agent.gpt_filter.call_llm", return_value=ans):
        assert llm_extract_comp("Producer", _DESC) is None


def test_llm_short_description_skipped_without_call():
    with patch("job_agent.gpt_filter.call_llm") as mock:
        assert llm_extract_comp("Producer", "short") is None
        mock.assert_not_called()


def test_llm_garbage_answer_returns_none():
    with patch("job_agent.gpt_filter.call_llm", return_value="I cannot determine that."):
        assert llm_extract_comp("Producer", _DESC) is None


def test_between_and_phrasing_apple_style():
    c = extract_comp({"description":
        "The base pay range for this role is between $75.03 and $112.79/hr, "
        "and your base pay will depend on your skills."})
    assert c["comp_text"] is not None and "/hr" in c["comp_text"]
    assert c["comp_yearly_min"] == round(75.03 * 2080)


def test_plain_and_without_between_not_matched():
    """Bare 'and' must not join unrelated amounts."""
    c = extract_comp({"description": "Budgets of $500 and $900 per project."})
    assert c["comp_text"] is None


# ── Comp floor (location-aware) ─────────────────────────────────────────────
import sqlite3

import pytest

from job_agent.comp_utils import (
    COMP_FLOOR_SECONDARY_METRO_ONSITE, COMP_FLOOR_YEARLY, apply_comp_floor, floor_for_location)
from tests.fixtures import persona


class _FakeDB:
    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute(
            "CREATE TABLE scrapes (id INTEGER PRIMARY KEY, tier TEXT, "
            "location TEXT, comp_yearly_min INTEGER, comp_yearly_max INTEGER, "
            "reasoning TEXT)")

    def _get_thread_conn(self):
        return self._conn


def test_floor_values_come_from_the_profile():
    """The applied floors are search.comp_floor_yearly / comp_floor_secondary_metro."""
    assert COMP_FLOOR_YEARLY == persona.COMP_FLOOR_YEARLY
    assert COMP_FLOOR_SECONDARY_METRO_ONSITE == persona.COMP_FLOOR_SECONDARY_METRO
    assert COMP_FLOOR_SECONDARY_METRO_ONSITE > COMP_FLOOR_YEARLY


@pytest.mark.parametrize("value,expected", [
    (72000, 72000), ("72000", 72000), ("72,000", 72000), (72000.0, 72000),
    (None, 41000), ("", 41000),            # missing or empty -> the default
])
def test_floor_from_profile_reads_the_key_or_falls_back(monkeypatch, value, expected):
    from job_agent import candidate_profile, comp_utils
    search = dict(candidate_profile.SEARCH.raw())
    if value is None:
        search.pop("comp_floor_yearly", None)
    else:
        search["comp_floor_yearly"] = value
    monkeypatch.setattr(candidate_profile, "SEARCH",
                        candidate_profile.Profile(search, candidate_profile.PROFILE.source))
    assert comp_utils._floor_from_profile("comp_floor_yearly", 41000) == expected


def test_floor_from_profile_rejects_text(monkeypatch):
    from job_agent import candidate_profile, comp_utils
    search = dict(candidate_profile.SEARCH.raw(), comp_floor_yearly="sixty thousand")
    monkeypatch.setattr(candidate_profile, "SEARCH",
                        candidate_profile.Profile(search, candidate_profile.PROFILE.source))
    with pytest.raises(ValueError, match="comp_floor_yearly"):
        comp_utils._floor_from_profile("comp_floor_yearly", 41000)


def test_floor_for_location_secondary_metro_onsite_is_higher():
    assert floor_for_location("Beaverton, OR") == COMP_FLOOR_SECONDARY_METRO_ONSITE
    assert floor_for_location("Portland, OR") == COMP_FLOOR_SECONDARY_METRO_ONSITE


def test_floor_for_location_primary_metro_and_remote_use_the_base_floor():
    assert floor_for_location(persona.HOME) == COMP_FLOOR_YEARLY
    assert floor_for_location("Remote - Portland, OR") == COMP_FLOOR_YEARLY  # remote wins
    assert floor_for_location("") == COMP_FLOOR_YEARLY


def test_apply_comp_floor_secondary_metro_onsite_untiered():
    """Below the floor is terminal: the row leaves the feed (tier NULL) rather
    than dropping one tier, which would keep it visible in the digest."""
    db = _FakeDB()
    db._conn.execute("INSERT INTO scrapes VALUES (1, 'strong', 'Hillsboro, OR', NULL, 90000, NULL)")
    assert apply_comp_floor(db) == 1
    tier, reasoning = db._conn.execute("SELECT tier, reasoning FROM scrapes WHERE id=1").fetchone()
    assert tier is None
    assert "secondary-metro on-site" in reasoning
    assert f"${COMP_FLOOR_SECONDARY_METRO_ONSITE // 1000}K" in reasoning and "removed from feed" in reasoning


def test_apply_comp_floor_primary_90k_survives_secondary_90k_does_not():
    db = _FakeDB()
    db._conn.execute("INSERT INTO scrapes VALUES (1, 'strong', 'Bellevue, WA', NULL, 90000, NULL)")
    db._conn.execute("INSERT INTO scrapes VALUES (2, 'worth_a_look', 'Gresham, OR', NULL, 90000, NULL)")
    assert apply_comp_floor(db) == 1
    assert db._conn.execute("SELECT tier FROM scrapes WHERE id=1").fetchone()[0] == "strong"
    assert db._conn.execute("SELECT tier FROM scrapes WHERE id=2").fetchone()[0] is None


def test_apply_comp_floor_na_untouched_and_idempotent():
    db = _FakeDB()
    db._conn.execute("INSERT INTO scrapes VALUES (1, 'strong', 'Portland, OR', NULL, NULL, NULL)")
    db._conn.execute("INSERT INTO scrapes VALUES (2, 'strong', 'Remote', NULL, 45000, NULL)")
    assert apply_comp_floor(db) == 1        # only the remote row under the base floor
    assert apply_comp_floor(db) == 0        # untiered rows never re-processed
    assert db._conn.execute("SELECT tier FROM scrapes WHERE id=1").fetchone()[0] == "strong"
    assert db._conn.execute("SELECT tier FROM scrapes WHERE id=2").fetchone()[0] is None


def test_floor_judges_ceiling_and_display_shows_fork():
    """If the range CEILING clears the floor, the row stays, and the card shows
    the yearly FORK, not a midpoint average that reads like a lowball
    ('$20-$30/hr' would otherwise display '(approx $52K/yr)')."""
    c = extract_comp({"description": "Pay: $20 - $30 per hour, DOE."})
    assert c["comp_text"] == "$20–$30/hr (≈$42K–$62K/yr)"   # fork, not $52K avg
    db = _FakeDB()
    db._conn.execute("INSERT INTO scrapes VALUES (1, 'worth_a_look', 'Remote', 41600, 62400, NULL)")
    db._conn.execute("INSERT INTO scrapes VALUES (2, 'worth_a_look', 'Remote', 35000, 48000, NULL)")
    assert apply_comp_floor(db) == 1  # ceiling 62.4K survives; ceiling 48K is untiered
    assert db._conn.execute("SELECT tier FROM scrapes WHERE id=1").fetchone()[0] == "worth_a_look"
    assert db._conn.execute("SELECT tier FROM scrapes WHERE id=2").fetchone()[0] is None
