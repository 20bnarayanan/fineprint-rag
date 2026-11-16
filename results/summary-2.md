> ILLUSTRATIVE SAMPLE: INVENTED runs for practice (outputs hand-built, scores computed by the repo's real metric functions). Not measured.

| config | trap recall | high-sev recall | deadline recall | fabricated quotes (1st draft) | fabricated quotes (final) | extra flags / lease | flags on clean lease | escalated | avg drafts |
|---|---|---|---|---|---|---|---|---|---|
| v1-rag-val | 56% | 42% | 40% | 14% | 0% | 5.25 | 6.00 | 20% | 2.00 |
| v2-rag-val | 62% | 75% | 60% | 4% | 0% | 5.00 | 5.00 | 20% | 1.80 |
| v3-rag-val | 81% | 100% | 80% | 2% | 0% | 3.75 | 3.00 | 0% | 1.40 |
| v4-rag-val | 88% | 100% | 100% | 1% | 0% | 2.50 | 2.00 | 0% | 1.20 |
| v4-full-val | 94% | 100% | 100% | 1% | 0% | 2.25 | 2.00 | 0% | 1.20 |
| v4-rag-noval | 75% | 75% | 100% | 5% | 5% | 3.00 | 2.00 | 0% | 1.00 |

_5 synthetic leases per config. Single run per cell: sampling is not pinned (the models reject a temperature setting), so there is run-to-run variance; treat differences of a few points as noise. 'Extra flags' is an upper bound on false positives (unlabeled != wrong)._
