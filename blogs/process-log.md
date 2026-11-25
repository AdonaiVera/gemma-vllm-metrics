# Process log

Raw chronological notes for the Part 2 blog: what was done, what was tried
and dropped, and why. Keep appending. The polished story is in
[`part2-draft.md`](part2-draft.md).

## 2026-11-21 - Audit and plan

- Read the sprint plan (TorchTPU Content Sprint, Nov 11-25, 2026) and the
  whole repo. The repo at that point was a metrics panel only: no benchmark
  harness, no results, NVIDIA-only compose file.
- Checked local `main` against `origin/main`: identical (`efd358a`), so
  GitHub had nothing extra.
- Decision: build Part 2 inside this repo in new folders (`benchmarks/`,
  `blogs/`) so the work builds visibly on Part 1.
- Credit request: estimated about $250 (TPU v6e-1 ~30 h, cloud GPU ~15 h,
  CPU ~10 h, storage, 2x buffer). Credits were approved.

## 2026-11-21 - Research (docs + search)

- vLLM-TorchTPU: v6e "supported, not covered by CI"; Gemma 4 validated for
  31B and 26B-A4B only; E2B/E4B not found. Needs Python 3.12 and a pinned
  vLLM version. Tool calling and `/metrics` not clearly documented.
- A field report: plain `google/gemma-4-E2B-it` serves on one v6e chip; the
  QAT variant fails to load. This killed the QAT checkpoint as the controlled
  baseline.
- Pricing (from search snippets, verify): v6e $2.70/chip-hour on-demand,
  $1.35 DWS Flex-start, $1.89 calendar mode.
- Fetching the main TPU pricing/docs pages was blocked on a redirect; I did
  not work around it.
- Checked whether newer/better open models exist. Qwen leads general
  rankings, but the project is about testing Google products, and Gemma 4 E2B
  is the model that fits the 12 GB card. Stayed with Gemma.

## 2026-11-21/22 - Controlled checkpoint

| Attempt | Result |
|---|---|
| bf16 `google/gemma-4-E2B-it`, `--gpu-memory-utilization 0.90` | Failed: only 10.02 of 11.57 GiB free at startup (desktop holds ~1.4 GB VRAM) |
| bf16, 0.85 | Out of memory loading weights (9.74 GiB allocated) |
| online fp8 (`--quantization fp8`), 0.85, 9,216 context, 8 seqs | Works. KV cache 1.39 GiB = 48,512 tokens, ~16.9x concurrency at 9,216 tokens |

Chosen: online fp8 of the plain checkpoint on every environment. Still to
verify on TPU that fp8 works for E2B there.

Housekeeping: disk at 96% (40 GB free); the HF cache volume gained ~10 GB of
bf16 weights that are no longer needed.

## 2026-11-22 - Harness

- `benchmarks/workloads.json`: frozen matrix.
- `benchmarks/run_benchmarks.py`: wraps `vllm bench serve`, local CLI or the
  docker image.
- Bug: docker `--user <uid>` has no passwd entry in the image; torch crashed
  looking up the username. Fixed by setting `USER`, `LOGNAME`, `HOME`.
- Smoke runs showed "Peak concurrent requests: 16" at `--max-concurrency 8`.
  Probably a counting quirk in vLLM's output; not investigated. TODO.
- Full local baseline: 27/27 runs, 0 failed requests (see the draft for the
  table).

## 2026-11-22 - Agent-task check

- `benchmarks/agent_eval.py` over the existing `tpu_blog_demo` dataset
  (24 quickstart images). Expected answers computed from the data.
- Result: 67% success, 100% valid tool calls. Two consistent model failures
  (skipped the right tool and guessed; didn't do the final subtraction). Traced
  one run to confirm the tools were not at fault.
- Temperature 0 makes repeats near-identical.

## 2026-11-22 - Cost and charts

- `benchmarks/analyze.py` + `benchmarks/pricing.json`. Cost per 1M output
  tokens assumes the accelerator is saturated and billed for the full run.
- Charts follow the dataviz guidance: fixed color per environment, 2px
  lines, ringed markers, hairline grids, direct end labels, one panel per
  workload on a shared y-axis. Images go to `blogs/images/`.
- Bug caught: the first real run wrote an empty cost chart because TPU
  prices existed in `pricing.json` before any TPU data did. Fixed so a cost
  scenario only counts when an environment present in the data has a price.

## 2026-11-22 - Cloud access check (read-only)

- `gcloud` active account/project was `the work account` / `the production project`
  (a production project billed to a regular billing account). Nothing was
  created there.
- A second, personal project exists with billing disabled, and the only open
  billing accounts are regular ones. No sprint credit grant was visible, so
  TPU provisioning is waiting on the credits landing or a confirmed project.
- CPU path: official GGUF repo is `ggml-org/gemma-4-E2B-it-GGUF`
  (`llama-server -hf ...`). Quantization will differ from the GPU/TPU runs, so
  it is labeled as not controlled.

## 2026-11-22 - TPU runbook (nothing created yet)

- Project is `the production project` (production, regular billing account), so the
  run must be: one labeled resource, one run, collect metrics, delete that
  exact resource, with everything validated before creating anything.
- Read-only findings: TPU and Compute APIs enabled; no TPU VMs, queued
  resources or compute instances existed; v6e-1 is offered in many zones
  (us-east5-b chosen); runtime images `v2-alpha-tpuv6e`, `v6e-ubuntu-2204`,
  `v6e-ubuntu-2404`.
- `gcloud alpha compute tpus queued-resources create` has
  `--max-run-duration` (used with flex-start), so a hard cost backstop
  exists. Plain `tpu-vm create` does not.
- Wrote `deploy/tpu/` (config, preflight, create, run_on_tpu, remote,
  teardown, README). Preflight passed against the real project; dry runs of
  create/teardown/run_on_tpu print the intended commands; the account guard
  was tested.
- Playwright tools were not available in this session, so console
  screenshots will come from the user, or from rendered terminal output.

## Open items

- TPU: minimal load test of E2B (fp8) on vLLM-TorchTPU; fall back to the JAX
  backend if needed and disclose.
- Check `/metrics` and tool calling on TPU.
- Cloud GPU and CPU deployments; fill prices in `pricing.json`.
- Run the 27-run matrix and agent check on each environment.
- Compare CUDA graphs vs `--enforce-eager` on the local GPU.
- More agent tasks if the current set doesn't separate environments.
- Images to capture: TPU serving screenshot, cost chart (needs prices).
  Done: environments diagram (`benchmarks/diagram.py`) and agent results
  chart (in `analyze.py`).
- Decide whether raw results are committed to the public repo.
- Remove the duplicate PDF and fix the `url` in `fiftyone.yml`.

## Sources

- [vLLM-TorchTPU docs](https://docs.vllm.ai/projects/vllm-torchtpu/en/latest/)
- [TorchTPU](https://github.com/google-pytorch/torch_tpu)
- [vLLM TPU / tpu-inference](https://github.com/vllm-project/tpu-inference)
- [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
- [Gemma 4 on TPU v6e field report](https://dev.to/gde/self-hosted-gemma-4-on-tpu-v6e-deployment-sre-with-antigravity-2g71)
- [Cloud TPU pricing](https://cloud.google.com/tpu/pricing)
- [DWS pricing](https://cloud.google.com/products/dws/pricing)

## Thinking-mode confound (2026-11-23)

- First CPU agent run: 83% vs GPU 67%; CPU passed `top_pred_class`, GPU failed it.
- Token counts exposed it: CPU used ~10-20x more completion tokens per task.
  `reasoning_content` appeared in llama.cpp responses: its `--reasoning auto`
  detects Gemma 4's thinking from the template and enables it. vLLM ran
  with thinking off.
- Fix: `agent_eval.py --thinking off|on` sets `chat_template_kwargs.enable_thinking`
  per request (works on both engines) and stores tool-call traces. Default off.
- Old runs archived as `results/<env>/agent_unpinned/` (not comparable).
- Thinking off on both: GPU 12/18, CPU 12/18, identical traces. Failures:
  `top_pred_class` (checks person/car/bicycle/motorcycle/truck, never calls the
  label-discovery tool, answers car) and `person_gap` (answers 93, expected 18).
- Thinking-on needs a reasoning parser on vLLM (`--reasoning-parser`), not
  configured; run it on CPU first as a labeled variant.

## TPU attempt 1: no capacity (2026-11-23)

- Created `gemma-bench-qr` (v6e-1, us-east5-b, FLEX_START, maxRunDuration 10800s,
  validUntil +1h) in `the production project`. `--provisioning-model flex-start` was accepted.
- Stayed `WAITING_FOR_RESOURCES` for the full hour, then `FAILED: Max start time
  passed`. No VM was granted, so nothing billed.
- Deleted the failed queued resource with `teardown.sh`; verified 0 nodes and 0
  queued resources remain.
- Takeaway for the blog: Flex-start trades price for availability. A v6e-1 was not
  available in us-east5-b within an hour.

## Thinking-on variant, CPU (2026-11-23)

- 15/18 (83%), valid tool calls 100%, avg 37 s per task (vs 3.2 s off).
- `top_pred_class` 3/3 with thinking (1,921 tokens, ~100 s), `person_gap` 0/3.
- Thinking helps the label-discovery task; it does not fix the subtraction one.

## TPU attempt 2: no capacity again (2026-11-23)

- Zone `us-east5-b` is not on Google's v6e Flex-start list (asia-northeast1-b,
  us-east5-a, us-south1-ai1b per the docs). On-demand (STANDARD) rejects
  `--max-run-duration`, so it has no hard stop.
- Retried Flex-start in `us-east5-a` (a listed zone): waited the full hour,
  `FAILED: Max start time passed`. Deleted; 0 nodes / 0 queued resources remain.
- Two one-hour Flex-start waits in v6e-1 got no capacity.

## TPU v6e-1 run (2026-11-23)

- Capacity: 4 queued waits failed/expired (Flex-start us-east5-b, Flex-start
  us-east5-a, on-demand us-east5-a/b in a 2h window). The on-demand request in
  us-east5-b was granted at 18:46:43Z after ~1h27m in that window. STANDARD
  provisioning rejects `--max-run-duration`, so no hard stop; used a local
  watchdog (extended twice) plus exact-name teardown. Per-zone v6e quota is 0
  in several zones (e.g. us-east4-a); only us-east5-* accepted requests.
- vLLM-TorchTPU 0.33.0 installed cleanly (~4 min). Serving E2B with fp8, tool
  flags and async scheduling worked; first start compiled for ~15 min.
- Bug (mine): `remote.sh bench` died silently (`pip list` under `set -e` in a uv
  venv) and the local SSH step stalled; ~1.5 h of VM time passed with no
  benchmark running. Fixed, reran detached on the VM. Lesson: check the VM's
  own process/result files, not just the local script.
- Results: 27/27 cells, 0 failed requests. Agent check (thinking off): 67%,
  100% valid tool calls, same per-task pattern as GPU/CPU. `/metrics` exposes
  403 `vllm:` series.
- Headline numbers (E2B fp8): c1 decode 47 tok/s (local RTX 4070 Ti: 101-128);
  long-prompt TTFT p50 172 ms at c1 (GPU 583 ms); c8 throughput 315-357 tok/s
  (GPU 416-947). Cost on-demand ~$2.1-2.4 per 1M output tokens at c8 (flex $1.1),
  ~$16 at c1.
- Billing window: ACTIVE 18:46:43Z to teardown ~21:32Z (~2h45m, ~$7.4 at the
  unverified $2.70/h). Verified 0 nodes / 0 queued resources / 0 instances.

## Audit after the TPU run (2026-11-23)

- vllm-torchtpu auto-selects an fp8_e5m2 KV cache on v6e (`serve.log`). The local
  GPU run used the default KV dtype. Weights match (online fp8) but KV precision
  does not: a second uncontrolled difference to disclose. `environment.json`
  corrected (the script had written `auto`).
- `packages` metadata was empty (version listing came back blank); libtpu 0.0.50
  and vllm 0.33.0 come from the log/server.
- Not captured: HBM use, TPU utilization, a cloud-GPU leg, GPU/TPU thinking-on.

## Blog update (2026-11-23)

- Filled the TPU, CPU, cost and conclusion sections from `analysis.csv`.
- Charts fixed: cost chart now uses each env's own price basis (electricity for
  local, on-demand for TPU; CPU unpriced) on a log scale; end labels no longer
  collide.
- Local GPU price is an assumption ($0.043/h: 285 W x $0.15/kWh), not measured.
- Effective cost of the real run: ~$7.4 / 334k output tokens = ~$22 per 1M,
  vs ~$2.3 on the best-case chart.
- Open: cloud GPU leg, TPU per-token metric anomaly on the long workload (6.4 ms
  vs ~21 ms implied by throughput), warm-up too short for TPU tail latency.

## Cloud GPU run (2026-11-24)

- Added `deploy/gpu/` (preflight, create, sweep, watchdog, run, teardown), copied from the TPU scripts, with Google's own hard stop (`--max-run-duration` + DELETE) in addition to a software watchdog.
- First create failed with a stockout in us-central1-a. A sweep over 22 zones in the US and Europe also failed; northamerica-northeast1-b granted an L4 `g2-standard-8` at 20:31Z.
- Detached steps polled through marker files worked: setup took about 10 minutes, the GPU matrix about 55 minutes, the agent check about 1 minute.
- The CPU matrix on the VM's 8 vCPUs was too slow (superseded by the entry below: the rest ran on a second VM). Only the 9 short cells finished before I had to pull results at 22:15Z, and I tore the VM down at 22:20Z, 1 hour 49 minutes after it started (about $1.70 at $0.94/h). The agent check was not run on the cloud CPU.
- Mistakes worth remembering: my progress monitor showed nothing because a `cut` in the filter buffered its output, so the first view of how far along the run was came late; and `pkill -f` matched my own shell again (exit 144) so the first teardown attempt did nothing. The VM was deleted a minute later with the plain teardown script.
- Before publishing I moved the account and project names into an untracked `config.local.sh`.

## Cloud CPU leg finished (2026-11-24)

- The first L4 VM was deleted before the CPU matrix finished, so I rented a second one (asia-southeast1-a after many zones were out of stock) with a 3-hour hard stop and a 170-minute watchdog, and ran only the missing cells with `deploy/gpu/run_cpu_rest.sh`.
- The medium cells and the long concurrency-1 cell finished, and the agent check ran (67%, 7.1 s per task). VM running time: about 1 hour 33 minutes.
- Time to first token on the cloud CPU's medium c4/c8 and long cells looks wrong (median near 0 next to a p95 of minutes); I reported it as unreliable instead of guessing the cause.
- This time I checked the run script's own result before reporting, after a status message that was out of date because the run had already finished.

## Methodology correction: prefix cache (2026-11-24)

- An independent review of the draft noticed that for the TPU and L4 the long-prompt first-token times in trials 2 and 3 are far below trial 1 (L4 8K at c1: 750, 83, 82 ms), while the local GPU is flat (583 ms every trial).
- Cause: the trials reuse the same fixed-seed prompts and vLLM's prefix cache is on by default. The L4 and TPU server logs show an 86% prefix cache hit rate. The local card's KV cache holds 48,512 tokens, so it can't keep sixteen 8K prompts and doesn't hit.
- Consequence: the earlier headline numbers averaged all trials and flattered the L4 (305 ms) and TPU (172 ms) first-token times, which also made the L4 look faster than the local card at 8K. With trial 1 only the order is TPU 383 ms, local 583 ms, L4 750 ms. Long-prompt throughput also drops on the L4 and TPU (c8: L4 325 vs 404, TPU 291 vs 326 tok/s). Everything in the blog now uses trial 1; the all-trials averages stay in `analysis_all_trials.csv`.
- Also found: the TPU's medium and long cells have per-request latencies that don't add up to wall-clock time (about 11 s per long request by duration vs 6.2 s mean end-to-end), so its first-token time is flagged as indicative.
- Better fix for a rerun: disable prefix caching, or use a different seed per trial.
