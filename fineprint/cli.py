"""Command line interface:  python -m fineprint analyze data/leases/lease_a.txt"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PROMPT_VERSIONS
from .report import render_markdown


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    _load_env()
    ap = argparse.ArgumentParser(prog="fineprint", description="Read the fine print of a lease, with every claim verified.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyze one lease text file")
    a.add_argument("lease", type=Path)
    a.add_argument("--prompt", choices=PROMPT_VERSIONS, default="v4")
    a.add_argument("--retrieval", choices=["rag", "full"], default="rag")
    a.add_argument("--no-validator", action="store_true", help="ablation: accept the first draft")
    a.add_argument("--no-execute", action="store_true", help="skip the tool-calling executor")
    a.add_argument("--move-in", help="move-in date YYYY-MM-DD (defaults to lease start)")
    a.add_argument("--out", default="outputs", help="where the .ics / email draft are written")
    a.add_argument("--json", action="store_true", help="print the plan as JSON instead of Markdown")
    args = ap.parse_args(argv)

    from .graph import run_pipeline  # imported late so --help works without API keys
    from .llm import ensure_api_key
    ensure_api_key()
    state = run_pipeline(args.lease.read_text(), prompt_version=args.prompt, retrieval=args.retrieval,
                         use_validator=not args.no_validator, run_executor=not args.no_execute,
                         out_dir=args.out, move_in=args.move_in)
    if args.json:
        plan = state.get("plan")
        print(json.dumps({"status": state["status"], "attempts": state["attempts"],
                          "plan": plan.model_dump() if plan else None, "history": state["history"]}, indent=2))
    else:
        print(render_markdown(state))
    return 0 if state["status"] == "accepted" else 2


if __name__ == "__main__":
    sys.exit(main())
