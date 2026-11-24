"""Runs the frozen benchmark matrix against any OpenAI-compatible endpoint.

Wraps `vllm bench serve` so every environment (local GPU, cloud GPU, TPU,
CPU) is measured by the same client with the same workloads. Uses the local
`vllm` CLI if present, otherwise runs it from the vLLM docker image.

Example:
    python benchmarks/run_benchmarks.py --env local-gpu-fp8 \
        --base-url http://localhost:8000 \
        --meta hardware="RTX 4070 Ti 12GB" --meta engine=vllm --meta precision=fp8
"""
import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = "vllm/vllm-openai:gemma4"


def load_workloads(path):
    with open(path) as f:
        return json.load(f)


def num_prompts(cfg, concurrency):
    return max(cfg["min_prompts"], cfg["prompts_per_slot"] * concurrency)


def fetch_json(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.load(resp)
    except Exception as e:
        return {"error": str(e)}


def bench_command(args, cfg, workload, concurrency, result_name, in_docker):
    sizes = cfg["workloads"][workload]
    bench = [
        "bench", "serve",
        "--backend", cfg["backend"],
        "--base-url", args.base_url,
        "--endpoint", cfg["endpoint"],
        "--model", cfg["model"],
        "--dataset-name", "random",
        "--random-input-len", str(sizes["input_len"]),
        "--random-output-len", str(sizes["output_len"]),
        "--num-prompts", str(num_prompts(cfg, concurrency)),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--num-warmups", str(cfg["num_warmups"]),
        "--ignore-eos",
        "--temperature", str(cfg["temperature"]),
        "--seed", str(cfg["seed"]),
        "--percentile-metrics", cfg["percentile_metrics"],
        "--metric-percentiles", cfg["metric_percentiles"],
        "--save-result",
        "--result-filename", result_name,
        "--disable-tqdm",
    ]
    if args.header:
        bench += ["--header", *args.header]

    if not in_docker:
        return ["vllm", *bench, "--result-dir", args.out_dir]

    return [
        "docker", "run", "--rm", "--network", "host",
        "--user", f"{os.getuid()}:{os.getgid()}",
        # No passwd entry exists for this uid in the image; torch crashes
        # looking up the username unless these are set.
        "-e", "USER=bench", "-e", "LOGNAME=bench", "-e", "HOME=/tmp",
        "-v", f"{args.out_dir}:/results",
        "--entrypoint", "vllm", args.image,
        *bench, "--result-dir", "/results",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--env", required=True, help="Environment label, e.g. local-gpu-fp8")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Server root, without /v1")
    parser.add_argument("--workloads", default=os.path.join(HERE, "workloads.json"))
    parser.add_argument("--only", nargs="*", help="Subset of workload names to run")
    parser.add_argument("--concurrencies", nargs="*", type=int, help="Override concurrency levels")
    parser.add_argument("--trials", type=int, help="Override trial count")
    parser.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE",
                        help="Environment metadata recorded alongside results")
    parser.add_argument("--header", nargs="*", help="Extra request headers, KEY=VALUE (e.g. Authorization=Bearer ...)")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image used when `vllm` is not installed")
    parser.add_argument("--results-root", default=os.path.join(HERE, "results"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    cfg = load_workloads(args.workloads)
    workloads = args.only or list(cfg["workloads"])
    concurrencies = args.concurrencies or cfg["concurrencies"]
    trials = args.trials or cfg["trials"]

    args.out_dir = os.path.abspath(os.path.join(args.results_root, args.env))
    os.makedirs(args.out_dir, exist_ok=True)

    in_docker = shutil.which("vllm") is None
    if in_docker and not shutil.which("docker"):
        sys.exit("Neither `vllm` nor `docker` found on PATH.")

    meta = dict(item.split("=", 1) for item in args.meta)
    environment = {
        "env": args.env,
        "base_url": args.base_url,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "client": f"docker:{args.image}" if in_docker else "local vllm CLI",
        "server_version": fetch_json(f"{args.base_url}/version"),
        "served_models": fetch_json(f"{args.base_url}/v1/models"),
        "meta": meta,
        "workloads": cfg,
    }
    with open(os.path.join(args.out_dir, "environment.json"), "w") as f:
        json.dump(environment, f, indent=2)

    runs = list(itertools.product(workloads, concurrencies, range(1, trials + 1)))
    for i, (workload, concurrency, trial) in enumerate(runs, 1):
        name = f"{workload}_c{concurrency}_t{trial}.json"
        if os.path.exists(os.path.join(args.out_dir, name)) and not args.dry_run:
            print(f"[{i}/{len(runs)}] skip {name} (already exists)")
            continue

        cmd = bench_command(args, cfg, workload, concurrency, name, in_docker)
        print(f"[{i}/{len(runs)}] {name}")
        if args.dry_run:
            print("  " + " ".join(cmd))
            continue
        if subprocess.run(cmd).returncode != 0:
            print(f"  FAILED: {name}", file=sys.stderr)


if __name__ == "__main__":
    main()
