"""Tools the executor agent may call (LangChain tool calling).

They are deliberately deterministic and side-effect-light: they write an .ics file
and a markdown email draft into an output directory. Execution only ever happens
AFTER the plan has passed validation.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool, tool


def compute_deadline_date(anchor: str, offset_days: int, key_dates: dict[str, date | None]) -> date | None:
    base = key_dates.get(anchor)
    if base is None:
        return None
    return base + timedelta(days=offset_days)


def _ics_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def make_tools(out_dir: str | Path, key_dates: dict[str, date | None]) -> list[BaseTool]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ics_path = out / "lease_deadlines.ics"
    log_path = out / "actions.jsonl"
    # The agent runtime executes parallel tool calls concurrently; the .ics update is a
    # read-modify-write, so it must be serialised or events get lost.
    lock = threading.Lock()

    def _log(entry: dict) -> None:
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    @tool
    def compute_deadline(anchor: Literal["move_in", "lease_start", "lease_end"], offset_days: int) -> str:
        """Compute a calendar date from an anchor date and an offset in days (negative = before the anchor).
        Returns an ISO date (YYYY-MM-DD) or 'UNKNOWN' if the anchor date is not known."""
        d = compute_deadline_date(anchor, offset_days, key_dates)
        return d.isoformat() if d else "UNKNOWN"

    @tool
    def add_calendar_event(title: str, iso_date: str, notes: str = "") -> str:
        """Add an all-day event to the tenant's deadline calendar (.ics). iso_date must be YYYY-MM-DD."""
        try:
            d = date.fromisoformat(iso_date)
        except ValueError:
            return f"ERROR: iso_date must be YYYY-MM-DD, got {iso_date!r}"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        event = [
            "BEGIN:VEVENT",
            f"UID:{hashlib.sha1((title + iso_date).encode()).hexdigest()[:16]}@fineprint",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{d.strftime('%Y%m%d')}",
            f"DTEND;VALUE=DATE:{(d + timedelta(days=1)).strftime('%Y%m%d')}",
            f"SUMMARY:{_ics_escape(title)}",
            f"DESCRIPTION:{_ics_escape(notes)}",
            "END:VEVENT",
        ]
        with lock:
            if not ics_path.exists():
                ics_path.write_text("BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//fineprint//EN\nEND:VCALENDAR\n")
            body = ics_path.read_text().replace("END:VCALENDAR\n", "\n".join(event) + "\nEND:VCALENDAR\n")
            ics_path.write_text(body)
            _log({"tool": "add_calendar_event", "title": title, "date": iso_date})
        return f"Added '{title}' on {iso_date}."

    @tool
    def draft_questions_email(questions: list[str]) -> str:
        """Write a short email to the landlord containing the given questions. Returns the file path."""
        qs = [q.strip() for q in questions if q and q.strip()]
        if not qs:
            return "ERROR: no questions provided."
        lines = ["Subject: Questions before I sign the lease", "", "Hello,", "",
                 "Before signing, I would like to clarify a few points in the lease:", ""]
        lines += [f"{i}. {q}" for i, q in enumerate(qs, start=1)]
        lines += ["", "Thank you,", "[Your name]", ""]
        path = out / "questions_for_landlord.md"
        path.write_text("\n".join(lines))
        _log({"tool": "draft_questions_email", "n_questions": len(qs)})
        return f"Draft saved to {path}."

    return [compute_deadline, add_calendar_event, draft_questions_email]
