"""Flattens raw `vllm bench serve` JSON results into one CSV.

One row per (env, workload, concurrency, trial), carrying every scalar the
benchmark reported plus the environment metadata, so the analysis step never
has to guess field names.

Example:
    python benchmarks/collect_results.py --out benchmarks/results/summary.csv
"""
import argparse
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_NAME = re.compile(r"^(?P<workload>\w+)_c(?P<concurrency>\d+)_t(?P<trial>\d+)\.json$")


def scalars(result):
    return {
        key: value
        for key, value in result.items()
        if isinstance(value, (int, float, str, bool)) or value is None
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", default=os.path.join(HERE, "results"))
    parser.add_argument("--out", default=os.path.join(HERE, "results", "summary.csv"))
    args = parser.parse_args()

    rows = []
    for env_dir in sorted(glob.glob(os.path.join(args.results_root, "*", ""))):
        env = os.path.basename(os.path.dirname(env_dir))
        try:
            with open(os.path.join(env_dir, "environment.json")) as f:
                meta = json.load(f).get("meta", {})
        except FileNotFoundError:
            meta = {}

        for path in sorted(glob.glob(os.path.join(env_dir, "*.json"))):
            match = RUN_NAME.match(os.path.basename(path))
            if not match:
                continue
            with open(path) as f:
                result = json.load(f)
            rows.append({
                "env": env,
                "workload": match["workload"],
                "concurrency": int(match["concurrency"]),
                "trial": int(match["trial"]),
                **{f"meta_{k}": v for k, v in meta.items()},
                **scalars(result),
            })

    if not rows:
        raise SystemExit(f"No results found under {args.results_root}")

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
