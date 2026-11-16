from fineprint.config import RISK_CATEGORIES
from fineprint.retrieval import HashingEmbeddings, build_index, retrieve_for_checklist


def test_hashing_embeddings_are_deterministic_and_normalised():
    e = HashingEmbeddings()
    a, b = e.embed_query("sublet the premises"), e.embed_query("sublet the premises")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_retrieval_surfaces_the_right_clause(lease_a_clauses):
    index = build_index(lease_a_clauses, HashingEmbeddings())
    ids = [d.metadata["clause_id"] for d in index.similarity_search(RISK_CATEGORIES["landlord_entry"]["query"], k=2)]
    entry = next(c.id for c in lease_a_clauses if c.title == "Landlord Entry")
    assert entry in ids


def test_checklist_retrieval_returns_document_order_and_hits(lease_a_clauses):
    index = build_index(lease_a_clauses, HashingEmbeddings())
    chosen, hits = retrieve_for_checklist(index, lease_a_clauses, k=2)
    assert [c.id for c in chosen] == sorted(c.id for c in chosen)
    assert set(hits) == set(RISK_CATEGORIES)
    assert len(chosen) < len(lease_a_clauses) + 1
