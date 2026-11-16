from datetime import date
from pathlib import Path

from fineprint.ingest import extract_term_dates, split_clauses

ROOT = Path(__file__).resolve().parents[1]


def test_every_ground_truth_sentence_is_in_its_lease(truth):
    for lease_id, t in truth.items():
        text = (ROOT / "data" / "leases" / f"{lease_id}.txt").read_text()
        for trap in t["traps"]:
            assert trap["evidence"], trap["id"]
            for e in trap["evidence"]:
                assert e in text
        for d in t["deadlines"]:
            assert d["evidence"] in text


def test_conflict_traps_have_two_evidence_sentences(truth):
    for t in truth.values():
        for trap in t["traps"]:
            if trap["id"].endswith("conflict"):
                assert len(trap["evidence"]) == 2


def test_clean_lease_has_no_traps(truth):
    assert truth["lease_e_clean"]["traps"] == []


def test_split_clauses(lease_a_clauses):
    assert len(lease_a_clauses) == 17
    assert lease_a_clauses[0].id == "s01"
    assert lease_a_clauses[1].title == "Term And Renewal"
    assert "automatically renews" in lease_a_clauses[1].text


def test_extract_term_dates(lease_a_text):
    assert extract_term_dates(lease_a_text) == (date(2027, 1, 1), date(2027, 12, 31))


def test_extract_term_dates_missing():
    assert extract_term_dates("no dates here") == (None, None)


def test_fallback_chunking_for_unstructured_text():
    clauses = split_clauses("First paragraph.\n\nSecond paragraph.")
    assert [c.id for c in clauses] == ["s01", "s02"]
