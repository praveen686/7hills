"""Research scripts for india_fno strategies."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[4] / "research_artefacts" / "results"


@contextmanager
def tee_to_results(script_name: str):
    """Context manager that tees stdout to a timestamped file in research_artefacts/results/.

    Usage:
        with tee_to_results("s1_vrp_rndr"):
            print("This goes to both console and file")
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    outpath = RESULTS_DIR / f"{script_name}_{ts}.txt"

    original_stdout = sys.stdout

    class Tee:
        def __init__(self, file, stream):
            self.file = file
            self.stream = stream

        def write(self, data):
            self.stream.write(data)
            self.file.write(data)

        def flush(self):
            self.stream.flush()
            self.file.flush()

    with open(outpath, "w") as f:
        tee = Tee(f, original_stdout)
        sys.stdout = tee
        try:
            yield outpath
        finally:
            sys.stdout = original_stdout

    print(f"\nResults saved to {outpath}")
