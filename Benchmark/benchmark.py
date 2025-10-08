# benchmarks/benchmark.py
#
# Entrypoint for GUI benchmarks.
#
# Usage examples:
#   python -m benchmarks.benchmark --scenario scatter_single
#   python benchmarks/benchmark.py --scenario all --sizes 1000 5000 --reps 2 --verbose

import sys
import argparse
from PyQt5 import QtWidgets

from .benchmark_scenarios import (
    scatter_single,
    scatter_simul,
    transforms,
    scaling,
    streaming,
    utils,
)

SCENARIO_MAP = {
    "scatter_single": scatter_single.run,
    "scatter_simul": scatter_simul.run,
    "transforms": transforms.run,
    "scaling": scaling.run,
    "streaming": streaming.run,
    "all": None,
}


def parse_args(argv):
    p = argparse.ArgumentParser(description="GUI Benchmark for MCVGraph")
    p.add_argument("--sizes", type=int, nargs="*", default=[1000, 5000, 15000],
                   help="Point counts for scatter dataset.")
    p.add_argument("--reps", type=int, default=3,
                   help="Repetitions per size.")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="Timeout per measurement (s).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed.")
    p.add_argument("--csv", type=str, default="bench_results.csv",
                   help="CSV output path.")
    p.add_argument("--json", type=str, default="bench_results.json",
                   help="JSON output path.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--scenario", type=str,
                   choices=list(SCENARIO_MAP.keys()),
                   default="all",
                   help="Which scenario to run.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    all_rows = []
    if args.scenario == "all":
        for name, fn in SCENARIO_MAP.items():
            if fn is not None:
                all_rows += fn(app,
                               sizes=args.sizes,
                               reps=args.reps,
                               timeout_s=args.timeout,
                               seed=args.seed,
                               verbose=args.verbose)
    else:
        fn = SCENARIO_MAP[args.scenario]
        all_rows = fn(app,
                      sizes=args.sizes,
                      reps=args.reps,
                      timeout_s=args.timeout,
                      seed=args.seed,
                      verbose=args.verbose)

    utils.write_outputs(all_rows, args.csv, args.json, args.verbose)
    app.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
