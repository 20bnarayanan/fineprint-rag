import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("FINEPRINT_EMBEDDINGS", "hashing")  # no model download in tests

import pytest  # noqa: E402

from fineprint.ingest import split_clauses  # noqa: E402


@pytest.fixture(scope="session")
def truth():
    return json.loads((ROOT / "data" / "traps.json").read_text())


@pytest.fixture(scope="session")
def lease_a_text():
    return (ROOT / "data" / "leases" / "lease_a.txt").read_text()


@pytest.fixture(scope="session")
def lease_a_clauses(lease_a_text):
    return split_clauses(lease_a_text)
