"""Reproduce the full verification end-to-end: data -> Exp 0/A/B -> figures + REPORT.md.

Usage:
    ./.venv/bin/python run_all.py
Outputs land in results/ (figures/, tables/, REPORT.md).
"""

from zeroday_verify.report import run_full

if __name__ == "__main__":
    res = run_full()
    print("\n=== SUMMARY ===")
    print("Exp 0 (SQL sanity):", {k: round(v, 3) for k, v in res["exp0"].items()})
    print("Exp A metrics:", {k: round(v, 3) for k, v in res["expA"]["metrics"].items()})
    print("Exp B aggregated AUROC:", {k: round(v, 3) for k, v in res["expB_aggregated"].items()})
    print("See results/REPORT.md and results/figures/.")
