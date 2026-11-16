"""Run both reference agents over all synthetic scenarios and write sample reports.

Usage: python3 demo.py
Writes sample_report.md and sample_report.json next to this file (overwriting any
hand-derived placeholder with real output).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from memory_eval.data import load_scenarios  # noqa: E402
from memory_eval.runner import build_report, render_markdown  # noqa: E402


def main():
    scenarios = load_scenarios()
    report = build_report(scenarios)
    with open(os.path.join(HERE, "sample_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(os.path.join(HERE, "sample_report.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    print(render_markdown(report))


if __name__ == "__main__":
    main()
