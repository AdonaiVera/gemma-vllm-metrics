# Cloud GPU benchmark run (one NVIDIA L4 VM)

One-shot, cost-guarded run of the same benchmark matrix on a rented GPU, plus
the CPU baseline on the same VM. Everything is a dry run unless you pass
`--execute`.

Setup: copy `config.local.example.sh` to `config.local.sh` and put in your
Google account and project. The scripts refuse to run on any other account.

1. `./preflight.sh` read-only: API, L4 offered in the zone, image, quota, no leftover VM.
2. `./create.sh --execute` creates one VM (`g2-standard-8`, 1x L4) with a hard
   max run time (`--max-run-duration`, termination action DELETE). You type the project name to confirm.
   L4 stock varies: if a zone is out of stock, `./sweep.sh <zone> <zone> ...` tries zones until one grants it and arms the watchdog.
3. `./watchdog.sh` second safety layer (deletes the VM after `WATCHDOG_MIN`).
4. `ZONE=<zone> ./run_on_gpu.sh --execute` copies the harness, runs the GPU matrix, the agent check through an SSH tunnel, then the CPU matrix. Long steps run detached on the VM and are polled through marker files.
5. `./teardown.sh --execute` deletes exactly the named VM and its boot disk, then verifies.

Lessons from the first run: budget the CPU matrix (8 vCPUs are slow, about 10 s per
short request at concurrency 1) or run it separately; the GPU matrix alone took about 55 minutes on an L4.
