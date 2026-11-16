"""Split a lease into clause-level chunks and pull out basic dates.

Chunking by section (not by fixed token window) keeps every chunk a coherent legal
unit, so a quote can be traced back to one clause id.
"""
from __future__ import annotations

import re
from datetime import date, datetime

from .schemas import Clause

_SECTION_RE = re.compile(r"^SECTION\s+(\d+)\.\s+([A-Z0-9 ,&'\-/]+?)\.\s+(.*)$")


def split_clauses(text: str) -> list[Clause]:
    clauses: list[Clause] = []
    for line in text.splitlines():
        m = _SECTION_RE.match(line.strip())
        if m:
            n = int(m.group(1))
            clauses.append(Clause(id=f"s{n:02d}", number=n, title=m.group(2).title(), text=m.group(3).strip()))
    if not clauses:
        # Fallback for unstructured text: paragraph chunks.
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        clauses = [Clause(id=f"s{i:02d}", number=i, title=f"Paragraph {i}", text=p)
                   for i, p in enumerate(paras, start=1)]
    return clauses


_TERM_RE = re.compile(
    r"begins on\s+([A-Z][a-z]+ \d{1,2}, \d{4})\s+and ends on\s+([A-Z][a-z]+ \d{1,2}, \d{4})")


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%B %d, %Y").date()


def extract_term_dates(text: str) -> tuple[date | None, date | None]:
    m = _TERM_RE.search(text)
    if not m:
        return None, None
    return _parse_date(m.group(1)), _parse_date(m.group(2))


def format_clauses(clauses: list[Clause]) -> str:
    return "\n\n".join(f"[{c.id}] SECTION {c.number} - {c.title}\n{c.text}" for c in clauses)
