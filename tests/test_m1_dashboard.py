from radar.m1_app.dashboard import (
    _detail,
    _table,
    headline,
    record_from_error,
    summarize,
)
from radar.contracts.models import VendorProfile


def make(vendor_id, name, tier, usage, risk, coverage, providers=None, error=""):
    return {
        "vendor_id": vendor_id,
        "name": name,
        "criticality": tier,
        "ai_usage": usage,
        "strength": "Probable" if usage == "Yes" else "None",
        "coverage": coverage,
        "risk": risk,
        "documents": 3,
        "signals": 2,
        "providers": providers or [],
        "use_cases": [],
        "reasons": [],
        "rationale": "",
        "action": "",
        "error": error,
    }


RECORDS = [
    make("V-001", "Alpha", "Low", "No", "None identified", 85.0),
    make("V-002", "Bravo", "Critical", "Yes", "Critical", 90.0, ["OpenAI"]),
    make("V-003", "Charlie", "High", "Yes", "High", 55.0, ["OpenAI", "Anthropic"]),
    make("V-004", "Delta", "Medium", "Inconclusive", "Medium", 40.0),
]


def test_summary_counts_and_averages():
    s = summarize(RECORDS)
    assert s["total"] == 4 and s["completed"] == 4 and s["errors"] == 0
    assert s["ai_yes"] == 2
    assert s["high_risk"] == 2
    assert s["avg_coverage"] == 67.5
    assert s["low_coverage"] == 2
    assert s["providers"][0] == ("OpenAI", 2)


def test_priority_orders_by_risk_then_tier_then_low_coverage():
    s = summarize(RECORDS)
    assert [r["vendor_id"] for r in s["priority"]] == ["V-002", "V-003", "V-004", "V-001"]


def test_matrix_places_vendors_by_tier_and_usage():
    s = summarize(RECORDS)
    assert s["matrix"]["Critical"]["Yes"] == ["V-002"]
    assert s["matrix"]["Low"]["No"] == ["V-001"]
    assert s["matrix"]["Medium"]["Inconclusive"] == ["V-004"]


def test_errors_are_counted_but_excluded_from_metrics_and_sorted_last():
    broken = make("V-005", "Echo", "Not assessed", "Inconclusive", "Not assessed", 0.0, error="Boom")
    s = summarize(RECORDS + [broken])
    assert s["total"] == 5 and s["completed"] == 4 and s["errors"] == 1
    assert s["avg_coverage"] == 67.5
    assert s["ordered"][-1]["vendor_id"] == "V-005"
    assert all(r["vendor_id"] != "V-005" for r in s["priority"])
    assert "1 vendor failed to process" in headline(s)


def test_headline_handles_singular_and_empty():
    one = summarize([make("V-001", "Alpha", "High", "Yes", "High", 90.0)])
    assert headline(one).startswith("1 of 1 assessed vendor show evidence")
    assert "1 is rated High or Critical" in headline(one)
    assert headline(summarize([])) == "No vendors could be assessed in this run."
    two = summarize([
        make("V-001", "Alpha", "High", "Yes", "High", 90.0),
        make("V-002", "Bravo", "Critical", "Yes", "Critical", 80.0),
    ])
    assert headline(two) == "2 of 2 assessed vendors show evidence of AI use, and 2 are rated High or Critical risk."


def test_html_output_escapes_vendor_text():
    nasty = make("V-009", "<script>alert(1)</script>", "High", "Yes", "High", 50.0)
    nasty["rationale"] = "<img src=x onerror=1>"
    assert "<script>" not in _table([nasty])
    assert "<img" not in _detail(nasty)


def test_record_from_error_defaults():
    vendor = VendorProfile("V-007", "Foxtrot", "", "", "", "", "", "", "", "")
    record = record_from_error(vendor, "ValueError: bad")
    assert record["risk"] == "Not assessed"
    assert record["error"] == "ValueError: bad"
