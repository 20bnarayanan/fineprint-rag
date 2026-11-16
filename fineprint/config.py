"""Central configuration. Everything is overridable by environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Risk categories drive three things: retrieval queries, the planner checklist,
# and the deterministic "you said it's not in the document" check.
# keywords are matched with word boundaries against clause text.
RISK_CATEGORIES: dict[str, dict] = {
    "renewal": {"query": "automatic renewal term renews notice of non-renewal", "keywords": ["renew", "renews", "renewal", "month-to-month"]},
    "rent_increase": {"query": "rent increase escalation upon renewal", "keywords": ["increase the monthly rent", "rent increase", "increase rent"]},
    "late_fees": {"query": "late fee late charge rent delinquent grace period", "keywords": ["late fee", "late charge", "delinquent"]},
    "security_deposit": {"query": "security deposit return deductions non-refundable fee", "keywords": ["security deposit", "non-refundable", "deposit"]},
    "utilities": {"query": "utilities water sewer trash electricity allocation of charges", "keywords": ["utilities", "water", "sewer", "electricity"]},
    "maintenance": {"query": "repairs maintenance who pays cost of repair", "keywords": ["repair", "repairs", "maintenance"]},
    "landlord_entry": {"query": "landlord entry notice inspection access to premises", "keywords": ["enter the premises", "entry"]},
    "guests": {"query": "guests overnight visitors unauthorized occupant", "keywords": ["guest", "guests", "unauthorized occupant"]},
    "pets": {"query": "pets animals pet rent pet fee", "keywords": ["pet", "pets", "animal", "animals"]},
    "subletting": {"query": "sublet subletting short-term rental assignment", "keywords": ["sublet", "sublease", "short-term rental", "assign this lease"]},
    "early_termination": {"query": "early termination break lease remaining rent re-rent", "keywords": ["terminate this lease", "early termination", "terminates this lease"]},
    "legal_fees": {"query": "attorney fees legal costs dispute prevailing party", "keywords": ["attorney", "attorney's fees"]},
    "dispute_resolution": {"query": "arbitration jury trial waiver dispute resolution", "keywords": ["arbitration", "jury trial", "arbitrator"]},
    "deadlines": {"query": "within days after move-in deadline notice period must deliver", "keywords": ["within 7 days", "within 10 days", "days after"]},
}

PROMPT_VERSIONS = ("v1", "v2", "v3", "v4")


@dataclass
class Settings:
    planner_model: str = field(default_factory=lambda: os.getenv("FINEPRINT_PLANNER_MODEL", "claude-haiku-5-5"))
    judge_model: str = field(default_factory=lambda: os.getenv("FINEPRINT_JUDGE_MODEL", "claude-sonnet-5-5"))
    executor_model: str = field(default_factory=lambda: os.getenv("FINEPRINT_EXECUTOR_MODEL", "claude-haiku-5-5"))
    embeddings: str = field(default_factory=lambda: os.getenv("FINEPRINT_EMBEDDINGS", "local"))  # local | hashing
    embedding_model: str = field(default_factory=lambda: os.getenv("FINEPRINT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    top_k_per_category: int = int(os.getenv("FINEPRINT_TOP_K", "2"))
    max_repairs: int = int(os.getenv("FINEPRINT_MAX_REPAIRS", "2"))
    # Retries for unparseable model output (transport flakes). Separate from max_repairs, which is for
    # content problems found by validation. Seen live: ~1 in 8 Haiku planner calls returned malformed JSON.
    parse_retries: int = int(os.getenv("FINEPRINT_PARSE_RETRIES", "2"))
    min_quote_chars: int = 20


SETTINGS = Settings()
