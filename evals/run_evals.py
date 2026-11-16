"""Run the lease-reader evals.

Examples
--------
  # local run, prints a table and writes results/summary.md  (needs ANTHROPIC_API_KEY)
  python -m evals.run_evals --grid default

  # same, but also log every experiment to LangSmith  (needs LANGSMITH_API_KEY)
  python -m evals.run_evals --grid default --langsmith

  # just the four prompt versions, two leases, quick smoke test
  python -m evals.run_evals --grid prompts --leases lease_a lease_e_clean

The default grid isolates one change at a time:
  v1 -> v2 -> v3 -> v4         prompt versions (rag, validator on)
  v4 + full context            what does retrieval cost/gain on a short lease?
  v4 + validator off           what does the validate/repair loop buy us?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.evaluators import ALL_EVALUATORS  # noqa: E402
from fineprint.graph import run_pipeline  # noqa: E402
from fineprint.validators import fabricated_quote_rate  # noqa: E402


@dataclass(frozen=True)
class Config:
    version: str = "v4"
    retrieval: str = "rag"
    validator: bool = True

    @property
    def label(self) -> str:
        return f"{self.version}-{self.retrieval}-{'val' if self.validator else 'noval'}"


GRIDS: dict[str, list[Config]] = {
    "prompts": [Config(v) for v in ("v1", "v2", "v3", "v4")],
    "default": [Config(v) for v in ("v1", "v2", "v3", "v4")]
               + [Config("v4", "full", True), Config("v4", "rag", False)],
}


def load_cases(lease_ids: list[str] | None = None) -> list[dict]:
    truth = json.loads((ROOT / "data" / "traps.json").read_text())
    cases = []
    for lease_id, t in truth.items():
        if lease_ids and lease_id not in lease_ids:
            continue
        text = (ROOT / "data" / "leases" / f"{lease_id}.txt").read_text()
        cases.append({"lease_id": lease_id, "text": text, "truth": t})
    return cases


def run_target(inputs: dict, cfg: Config, llms=None, embeddings=None) -> dict:
    """One pipeline run -> a plain, JSON-serialisable dict (what evaluators see)."""
    state = run_pipeline(inputs["text"], prompt_version=cfg.version, retrieval=cfg.retrieval,
                         use_validator=cfg.validator, run_executor=False, llms=llms, embeddings=embeddings,
                         tags=["eval"], metadata={"lease_id": inputs["lease_id"], "config": cfg.label})
    plan = state.get("plan")
    report = state.get("report")
    return {
        "lease_id": inputs["lease_id"],
        "plan": plan.model_dump() if plan else {"summary": "", "items": [], "not_found": []},
        "status": state["status"],
        "attempts": state["attempts"],
        "first_draft_fab_rate": state.get("first_draft_fab_rate"),
        "first_draft_gate_failures": state.get("first_draft_gate_failures"),
        "final_fab_rate": fabricated_quote_rate(plan, state["clauses"]) if plan else None,
        "judge_verdict": report.judge.verdict if report and report.judge else None,
        "history": state["history"],
    }


def score_row(outputs: dict, truth: dict) -> dict[str, float | None]:
    return {r["key"]: r["score"] for r in (ev(outputs, truth) for ev in ALL_EVALUATORS)}


def run_local(grid: list[Config], cases: list[dict], workers: int = 1,
              target: Callable[[dict, Config], dict] | None = None,
              partial_path: Path | None = None) -> list[dict]:
    """Run every (config, lease) pair. If `partial_path` is given, each finished run is appended to it
    immediately (JSON lines), so an outage or crash partway through never loses completed work."""
    target = target or run_target
    jobs = [(cfg, case) for cfg in grid for case in cases]
    write_lock = threading.Lock()

    def one(job):
        cfg, case = job
        out = target({"lease_id": case["lease_id"], "text": case["text"]}, cfg)
        print(f"  {cfg.label:<18} {case['lease_id']:<14} status={out['status']:<9} drafts={out['attempts']}", flush=True)
        row = {"config": cfg.label, "lease_id": case["lease_id"], "outputs": out,
               "scores": score_row(out, case["truth"])}
        if partial_path is not None:
            with write_lock, partial_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, jobs))
    return [one(j) for j in jobs]


def aggregate(rows: list[dict]) -> dict[str, dict[str, float | None]]:
    by_cfg: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        bucket = by_cfg.setdefault(r["config"], {})
        for k, v in r["scores"].items():
            if v is not None:
                bucket.setdefault(k, []).append(float(v))
    return {cfg: {k: statistics.mean(v) for k, v in keys.items()} for cfg, keys in by_cfg.items()}


def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v:.0%}" if pct else f"{v:.2f}"


def to_markdown(agg: dict[str, dict[str, float | None]], n_leases: int) -> str:
    cols = [("trap_recall", "trap recall", True), ("high_severity_recall", "high-sev recall", True),
            ("deadline_recall", "deadline recall", True),
            ("fabricated_quote_rate_first_draft", "fabricated quotes (1st draft)", True),
            ("fabricated_quote_rate_final", "fabricated quotes (final)", True),
            ("extra_flags_per_lease", "extra flags / lease", False),
            ("clean_lease_flags", "flags on clean lease", False),
            ("escalated", "escalated", True), ("drafts", "avg drafts", False)]
    head = "| config | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    for cfg, vals in agg.items():
        lines.append(f"| {cfg} | " + " | ".join(_fmt(vals.get(k), p) for k, _, p in cols) + " |")
    lines.append("")
    lines.append(f"_{n_leases} synthetic leases per config. Single run per cell: sampling is not "
                 "pinned (the models reject a temperature setting), so there is run-to-run variance; treat differences of a few points as noise. "
                 "'Extra flags' is an upper bound on false positives (unlabeled != wrong)._")
    return "\n".join(lines)


# ----------------------------------------------------------------- LangSmith
def ensure_dataset(client, cases: list[dict]) -> str:
    fingerprint = hashlib.md5(json.dumps([(c["lease_id"], c["text"], c["truth"]) for c in cases],
                                         sort_keys=True).encode()).hexdigest()[:8]
    name = f"fineprint-leases-{fingerprint}"
    if not client.has_dataset(dataset_name=name):
        ds = client.create_dataset(name, description="Synthetic leases with planted traps and ground-truth evidence.")
        client.create_examples(dataset_id=ds.id, examples=[
            {"inputs": {"lease_id": c["lease_id"], "text": c["text"]}, "outputs": c["truth"]} for c in cases])
    return name


def run_langsmith(grid: list[Config], cases: list[dict], workers: int = 2) -> None:
    from langsmith import Client, evaluate  # imported lazily so local mode needs no key
    client = Client()
    dataset = ensure_dataset(client, cases)
    for cfg in grid:
        print(f"LangSmith experiment: {cfg.label} on dataset {dataset}")
        evaluate(lambda inputs, cfg=cfg: run_target(inputs, cfg), data=dataset, evaluators=ALL_EVALUATORS,
                 experiment_prefix=f"fineprint-{cfg.label}", max_concurrency=workers,
                 metadata={**asdict(cfg), "models": "see FINEPRINT_*_MODEL env"})


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=sorted(GRIDS), default="default")
    ap.add_argument("--leases", nargs="*", help="subset of lease ids")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--langsmith", action="store_true", help="also log experiments to LangSmith")
    args = ap.parse_args(argv)

    from fineprint.llm import ensure_api_key
    ensure_api_key()
    cases = load_cases(args.leases)
    grid = GRIDS[args.grid]
    print(f"{len(grid)} configs x {len(cases)} leases = {len(grid) * len(cases)} pipeline runs")
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    partial = out_dir / "runs.partial.jsonl"
    partial.unlink(missing_ok=True)
    rows = run_local(grid, cases, workers=args.workers, partial_path=partial)
    agg = aggregate(rows)
    md = to_markdown(agg, len(cases))
    (out_dir / "summary.md").write_text(md + "\n")
    (out_dir / "runs.json").write_text(json.dumps(rows, indent=2, default=str))
    print("\n" + md)
    if args.langsmith:
        run_langsmith(grid, cases, workers=max(1, args.workers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
