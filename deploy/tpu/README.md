# One-shot TPU v6e benchmark run

Runs the frozen benchmark matrix and the agent-task check on a single Cloud
TPU v6e-1, then removes it. Target is a **production project**, so everything
is pinned in [`config.sh`](config.sh): account, project, zone, resource names
and a hard run limit. Every script is a dry run unless given `--execute`.

## Order

1. `deploy/tpu/preflight.sh` - read-only checks (APIs, v6e-1 in the zone,
   runtime image, no leftovers) and the worst-case cost.
2. Check in the Cloud console that the billing account on the project has
   the sprint credits, so this does not charge real money.
3. `deploy/tpu/create.sh --execute` - creates ONE labeled v6e-1 as a
   Flex-start queued resource with `--max-run-duration` (hard stop) and
   `--valid-until-duration` (stops waiting for capacity). Asks you to type
   the project name first.
4. Wait until the queued resource is `ACTIVE`.
5. `deploy/tpu/run_on_tpu.sh --execute` - copies the code, installs
   vLLM-TorchTPU, serves `google/gemma-4-E2B-it` (online fp8), runs the 27-run
   matrix on the VM, pulls results back, runs the agent check through an SSH
   tunnel. It stops at the first failure and tells you the VM is still up.
6. `deploy/tpu/teardown.sh --execute` - deletes only
   `gemma-bench-qr` / `gemma-bench-v6e1` by exact name and proves they are
   gone. Do this immediately after the metrics are saved.

## Cost

Worst case at the 3h limit: about $4 (Flex-start) or $8 (on-demand rate).
Prices are unverified; see `benchmarks/pricing.json`.

## Known unverified points (first real run will tell)

- The accepted value of `--provisioning-model` (`flex-start` is assumed). A
  rejected value fails the create call and creates nothing.
- TPU v6e quota and capacity: only the create call reveals them.
- vLLM-TorchTPU install steps, and whether it accepts
  `google/gemma-4-E2B-it` with `--quantization fp8`. The docs only validate
  the 31B and 26B-A4B Gemma 4 models. Failure is a finding, not a bug: the
  fallback is the JAX-based vLLM TPU backend, disclosed in the post.
- Whether `--enable-auto-tool-choice --tool-call-parser gemma4` and `/metrics`
  work on this backend. `remote.sh` keeps these as separate flags
  (`TOOL_FLAGS`, `EXTRA_FLAGS`) so a rejected flag can be dropped and recorded.
