"""Command-line interface for demonstrating the multi-agent system.

    python agents_cli.py demo   [--source live|csic] [--case CASE_ID]   # one case, streamed live
    python agents_cli.py live   [--limit N]                             # all mock-env cases
    python agents_cli.py compare [--source live|csic] [--case CASE_ID]  # 3 configurations

The `demo` mode prints each agent's step as it happens, which is what makes the system
watchable rather than a black box.
"""

from __future__ import annotations

import argparse
import sys

from .cases import load_cases_from_log, load_evaluation_setup
from .llm import LLMClient
from .orchestrator import run_detector, run_mas, run_single_agent
from .tools import Case, ToolContext
from .trace import Step

C = {"Planista": "\033[94m", "Wykonawca": "\033[92m", "Audytor": "\033[95m",
     "Agent": "\033[92m", "Detektor": "\033[96m"}
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def _print_step(s: Step) -> None:
    colour = C.get(s.agent, "")
    body = s.content if len(s.content) < 500 else s.content[:500] + "…"
    tool = f" {DIM}({s.tool}){RESET}" if s.tool else ""
    print(f"  {colour}{BOLD}{s.agent:9s}{RESET} {DIM}{s.kind:11s}{RESET}{tool} {body}")
    print(f"  {DIM}{'':9s} {s.seconds:.1f}s{RESET}")


def _banner(case: Case) -> None:
    truth = f" (prawda: {case.label})" if case.label else ""
    print(f"\n{BOLD}=== Zdarzenie {case.case_id}{truth} ==={RESET}")
    print(f"{DIM}{case.summary()[:300]}{RESET}\n")


def _setup(source: str) -> tuple[ToolContext, list[Case], float]:
    """Reference knowledge always comes from CSIC; cases may come from the mock environment."""
    ctx, csic_cases, threshold = load_evaluation_setup(n_eval_attack=8, n_eval_benign=8)
    cases = load_cases_from_log() if source == "live" else csic_cases
    return ctx, cases, threshold


def _pick(cases: list[Case], case_id: str | None, prefer_attack: bool = True) -> Case:
    if case_id:
        for c in cases:
            if c.case_id == case_id:
                return c
        raise SystemExit(f"nie znaleziono sprawy '{case_id}'")
    if prefer_attack:
        for c in cases:
            if c.label == "attack":
                return c
    return cases[0]


def cmd_demo(args) -> None:
    llm = _client()
    ctx, cases, _ = _setup(args.source)
    case = _pick(cases, args.case)
    _banner(case)
    print(f"{BOLD}--- system wieloagentowy (na żywo) ---{RESET}")
    res = run_mas(llm, ctx, case, on_step=_print_step)
    d = res.decision
    print(f"\n{BOLD}DECYZJA:{RESET} {d.decision}  "
          f"{BOLD}wiarygodność:{RESET} {d.confidence:.2f}  "
          f"{BOLD}rund audytu:{RESET} {res.audit_rounds}")
    print(f"{BOLD}REKOMENDACJA:{RESET} {d.recommendation}")
    print(f"{DIM}{res.llm_calls} wywołań modelu, {res.seconds:.1f}s{RESET}")
    if case.label:
        ok = (case.label == "attack") == d.is_alert
        print(f"{BOLD}Zgodność z prawdą:{RESET} {'TAK' if ok else 'NIE'}")


def cmd_compare(args) -> None:
    llm = _client()
    ctx, cases, threshold = _setup(args.source)
    case = _pick(cases, args.case)
    _banner(case)
    for name, fn in (("detektor", lambda: run_detector(ctx, case, threshold)),
                     ("pojedynczy agent", lambda: run_single_agent(llm, ctx, case)),
                     ("system wieloagentowy", lambda: run_mas(llm, ctx, case))):
        r = fn()
        ok = "" if case.label is None else (
            "  [zgodne]" if (case.label == "attack") == r.decision.is_alert else "  [błąd]")
        print(f"  {BOLD}{name:22s}{RESET} {r.decision.decision:9s} "
              f"conf={r.decision.confidence:.2f}  {r.seconds:5.1f}s  "
              f"{r.llm_calls} wywołań{ok}")


def cmd_live(args) -> None:
    llm = _client()
    ctx, cases, threshold = _setup("live")
    cases = cases[: args.limit]
    print(f"Analiza {len(cases)} zdarzeń z środowiska testowego\n")
    hits = 0
    for c in cases:
        r = run_mas(llm, ctx, c)
        ok = (c.label == "attack") == r.decision.is_alert
        hits += ok
        mark = "OK " if ok else "BŁĄD"
        print(f"  {c.case_id:10s} prawda={str(c.label):7s} -> {r.decision.decision:9s} "
              f"conf={r.decision.confidence:.2f}  {mark}  ({r.seconds:.0f}s)")
    print(f"\nzgodnych: {hits}/{len(cases)}")


def _client() -> LLMClient:
    llm = LLMClient()
    if not llm.available():
        raise SystemExit(f"Backend '{llm.backend}' niedostępny. Uruchom `ollama serve` "
                         f"i `ollama pull {llm.model}`.")
    return llm


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Demonstracja systemu wieloagentowego")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("demo", cmd_demo), ("compare", cmd_compare)):
        sp = sub.add_parser(name)
        sp.add_argument("--source", choices=("live", "csic"), default="live")
        sp.add_argument("--case", default=None)
        sp.set_defaults(func=fn)
    sp = sub.add_parser("live")
    sp.add_argument("--limit", type=int, default=8)
    sp.set_defaults(func=cmd_live)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
