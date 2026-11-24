# Benchmarks

Part 2 of the Gemma 4 + Voxel51 Agent project: the same model measured on
every environment by the same client.

- [`workloads.json`](workloads.json): the frozen matrix. Checkpoint, seed,
  temperature, short/medium/long request sizes, concurrency 1/4/8, warmups
  and trials. Change it only before a campaign starts, never between
  environments.
- [`run_benchmarks.py`](run_benchmarks.py): runs the matrix against any
  OpenAI-compatible endpoint with `vllm bench serve`, using the local `vllm`
  CLI if installed and the vLLM docker image otherwise. Raw JSON results and
  an `environment.json` (server version, served model, `--meta` fields) are
  written to `results/<env>/`. Finished cells are skipped on rerun.
- [`collect_results.py`](collect_results.py): flattens all runs into one CSV
  for analysis.

## Controlled setup

All environments serve `google/gemma-4-E2B-it` with online fp8 weights.
bf16 does not fit on the 12GB local GPU (the weights alone take ~9.7 GiB),
so fp8 is the common denominator. Anything that differs on an environment
(engine, KV-cache dtype, runtime, a different checkpoint on CPU) must be
passed as `--meta key=value` so it lands in the results.

## Local GPU baseline

```bash
docker compose -f docker/compose.bench.yml up -d
python benchmarks/run_benchmarks.py --env local-gpu-fp8 \
    --meta hardware="RTX 4070 Ti 12GB" --meta engine=vllm --meta precision=fp8
python benchmarks/collect_results.py
docker compose -f docker/compose.bench.yml down
```

Close other GPU-heavy apps first; the desktop's own VRAM use affects how
much cache vLLM can claim.

## Agent-task check

[`agent_eval.py`](agent_eval.py) runs a plain OpenAI tool-calling loop with
read-only FiftyOne tools over the `tpu_blog_demo` dataset (24 images with
ground truth, predictions, uniqueness and CLIP labels). Expected answers are
computed from the dataset, so grading needs no hardcoded numbers. It reports
task success, valid tool-call rate, turns, tokens and seconds per task.

```bash
python benchmarks/agent_eval.py --env local-gpu-fp8 \
    --base-url http://localhost:8000/v1 --repeats 3
```

This is a proxy for the Voxel51 Agent, not the agent app itself. It runs on
the client machine, so it can target a remote endpoint (pass `--api-key` for
the ngrok bearer token). It needs `fiftyone` and `openai` installed.
Temperature is 0 for reproducibility, so repeats on one environment are
mostly identical; they matter more for catching nondeterminism across
environments.

## Analysis and cost

[`analyze.py`](analyze.py) aggregates `results/summary.csv` across trials
into `results/analysis.csv` and draws the blog charts into `blogs/images/`
(throughput, p95 time to first token, time per output token, and cost). Cost
per million output tokens uses the hourly prices in
[`pricing.json`](pricing.json), assuming the accelerator is billed for the
whole run and kept saturated at that concurrency. Fill the prices in and
verify them against Google Cloud's pricing pages first; environments with a
`null` price are skipped, and `--price ENV=USD_PER_HOUR` overrides on the
command line. Needs `pandas` and `matplotlib`.

```bash
python benchmarks/collect_results.py
python benchmarks/analyze.py
```
