"""Calibrate the LLM judge against plans whose defects are known by construction.

  python -m evals.calibrate_judge            # needs ANTHROPIC_API_KEY
  python -m evals.calibrate_judge --repeats 3

Reports false-alarm rate on correct plans, detection rate on degraded plans,
localization (did the judge name the clause that was dropped?) and, with --repeats,
how often repeated judge calls agree with each other.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fineprint.calibration import build_calibration_cases, score_calibration  # noqa: E402
from fineprint.ingest import format_clauses  # noqa: E402
from fineprint.llm import ensure_api_key, get_chat_model, run_judge  # noqa: E402
from fineprint.config import SETTINGS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--leases", nargs="*")
    args = ap.parse_args(argv)

    ensure_api_key()
    judge = get_chat_model(SETTINGS.judge_model)
    cases = build_calibration_cases(args.leases)
    print(f"{len(cases)} calibration cases x {args.repeats} repeat(s) with judge {SETTINGS.judge_model}")
    results, agree = [], []
    for c in cases:
        verdicts = []
        for _ in range(args.repeats):
            v = run_judge(judge, format_clauses(c["clauses"]), c["plan"])
            verdicts.append(v)
            results.append({"lease_id": c["lease_id"], "name": c["name"], "expect": c["expect"],
                            "verdict": v.verdict, "flagged_clause_ids": [m.clause_id for m in v.missed_risks],
                            "dropped_clause_ids": c["dropped_clause_ids"], "scope_ok": v.scope_ok,
                            "groundedness": v.groundedness})
        agree.append(len({v.verdict for v in verdicts}) == 1)
        print(f"  {c['lease_id']:<14} {c['name']:<28} expect={c['expect']:<4} got={[v.verdict for v in verdicts]}")
    score = score_calibration(results)
    if args.repeats > 1:
        score["self_agreement"] = sum(agree) / len(agree)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "judge_calibration.json").write_text(json.dumps({"score": score, "results": results}, indent=2))
    print("\n" + json.dumps(score, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
