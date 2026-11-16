from datetime import date

from fineprint.tools import compute_deadline_date, make_tools

KD = {"lease_start": date(2027, 1, 1), "lease_end": date(2027, 12, 31), "move_in": date(2027, 1, 1)}


def _tools(tmp_path, kd=KD):
    return {t.name: t for t in make_tools(tmp_path, kd)}


def test_compute_deadline_date():
    assert compute_deadline_date("lease_end", -90, KD) == date(2027, 10, 2)
    assert compute_deadline_date("move_in", 7, KD) == date(2027, 1, 8)
    assert compute_deadline_date("move_in", 7, {"move_in": None}) is None


def test_compute_deadline_tool_unknown_anchor(tmp_path):
    t = _tools(tmp_path, {"lease_start": None, "lease_end": None, "move_in": None})
    assert t["compute_deadline"].invoke({"anchor": "move_in", "offset_days": 7}) == "UNKNOWN"
    assert _tools(tmp_path)["compute_deadline"].invoke({"anchor": "lease_end", "offset_days": -90}) == "2027-10-02"


def test_calendar_event_written_and_valid(tmp_path):
    t = _tools(tmp_path)
    assert "Added" in t["add_calendar_event"].invoke({"title": "Notice, due; soon", "iso_date": "2027-10-02"})
    ics = (tmp_path / "lease_deadlines.ics").read_text()
    assert ics.startswith("BEGIN:VCALENDAR") and ics.rstrip().endswith("END:VCALENDAR")
    assert "DTSTART;VALUE=DATE:20271002" in ics and "Notice\\, due\\; soon" in ics


def test_calendar_rejects_bad_date(tmp_path):
    out = _tools(tmp_path)["add_calendar_event"].invoke({"title": "x", "iso_date": "next tuesday"})
    assert out.startswith("ERROR") and not (tmp_path / "lease_deadlines.ics").exists()


def test_email_draft(tmp_path):
    t = _tools(tmp_path)
    assert t["draft_questions_email"].invoke({"questions": []}).startswith("ERROR")
    t["draft_questions_email"].invoke({"questions": ["Q one?", " ", "Q two?"]})
    body = (tmp_path / "questions_for_landlord.md").read_text()
    assert "1. Q one?" in body and "2. Q two?" in body


def test_concurrent_calendar_writes_do_not_lose_events(tmp_path):
    """Regression: parallel tool calls used to race on the .ics read-modify-write."""
    from concurrent.futures import ThreadPoolExecutor
    add = _tools(tmp_path)["add_calendar_event"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: add.invoke({"title": f"event {i}", "iso_date": "2027-03-01"}), range(40)))
    assert (tmp_path / "lease_deadlines.ics").read_text().count("BEGIN:VEVENT") == 40
