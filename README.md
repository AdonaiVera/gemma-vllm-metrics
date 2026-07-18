# Gemma vLLM Metrics — FiftyOne Plugin

A [FiftyOne](https://github.com/voxel51/fiftyone) plugin for monitoring a
locally-served [Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
model. It gives you a real, local OpenAI-compatible inference endpoint (via
[vLLM](https://github.com/vllm-project/vllm)) that you can point any
OpenAI-compatible client at, plus a hybrid panel inside the FiftyOne App that
surfaces vLLM's own consumption metrics live: request volume, token
throughput, latency, GPU KV-cache usage, and queue depth.

## Architecture

```
┌─────────────────────┐        HTTP         ┌──────────────────────────┐
│   docker compose     │◄────────────────────┤  any OpenAI-compatible   │
│   vllm/vllm-openai    │                     │  client (your agent,     │
│   serving Gemma 4     │────────────────────►│  curl, the SDK, etc.)    │
│   :8000/v1  :8000/metrics                   └──────────────────────────┘
└──────────┬────────────┘
           │ HTTP (/v1/models, /metrics)
           ▼
┌─────────────────────────────────────────────┐
│  FiftyOne plugin (this repo)                 │
│  - operators.py: status + metrics operators   │
│  - panel.py: hybrid Panel (Python backend)    │
│  - src/*.tsx: React/VOODO frontend            │
└─────────────────────────────────────────────┘
```

The plugin **never manages the vLLM process itself** — operators in FiftyOne
execute per-invocation, not as daemons, so they're the wrong tool for owning
a long-lived server. vLLM is brought up separately via `docker compose` (see
[`scripts/serve.sh`](scripts/serve.sh)), and the plugin is purely an HTTP
client against it: a status check, a metrics poller, and a panel that
displays both. This mirrors the pattern used by
[`fiftyone-vllm`](https://github.com/Burhan-Q/fiftyone-vllm).

All metrics come straight from vLLM's own built-in Prometheus `/metrics`
endpoint (`prompt_tokens_total`, `generation_tokens_total`,
`time_to_first_token_seconds`, `inter_token_latency_seconds`,
`e2e_request_latency_seconds`, `num_requests_running`,
`num_requests_waiting`, `kv_cache_usage_perc`) — no custom instrumentation
or metrics database. See [`gemma_vllm_metrics/metrics_client.py`](gemma_vllm_metrics/metrics_client.py).
Metric names can drift between vLLM versions -- the client was written and
tested against a real captured `/metrics` response (see
[`tests/fixtures/sample_metrics.txt`](tests/fixtures/sample_metrics.txt)); if
you bump the vLLM image tag, diff a fresh `curl localhost:8000/metrics`
against that fixture before trusting the panel's numbers.

## Why not the full-precision model?

Full BF16 Gemma 4 E4B is ~16GB of weights, which doesn't fit on a 12GB
consumer GPU alongside its KV cache. Instead this defaults to Google's own
[QAT](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)
(quantization-aware training) checkpoint,
[`google/gemma-4-E4B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-E4B-it-qat-w4a16-ct) —
real weights from an official source, ~4GB, serialized in vLLM-native
compressed-tensors W4A16 format (no `--quantization` flag needed; vLLM
auto-detects it from the checkpoint config). Drop to
`google/gemma-4-E2B-it-qat-w4a16-ct` in `docker/.env` for even more headroom.

Even quantized, a single consumer GPU has almost no memory slack once you
account for weights + KV cache + CUDA graph capture. `docker/compose.yml`
also pins `--max-model-len 8192`, `--max-num-seqs 8`, and `--enforce-eager`
(disables CUDA graphs, trading some throughput for the memory their capture
buffers would otherwise need) -- vLLM's defaults (256 concurrent sequences,
full context length, graph capture) are sized for multi-GPU production
serving and will OOM on a 12GB card. If you have more VRAM, raising
`--max-num-seqs` and dropping `--enforce-eager` will improve throughput.

## Prerequisites

- Docker with the NVIDIA Container Toolkit (`docker info | grep -i nvidia`
  should show `nvidia` as an available/default runtime)
- An NVIDIA GPU with **CUDA ≥12.9** driver support (check with `nvidia-smi`).
  Gemma 4 needs a newer CUDA runtime than most `vllm/vllm-openai:latest`
  images assume today — this repo pins the `gemma4` image tag, which targets
  CUDA 12.9.
- ~10GB free disk for the model weights + vLLM's Docker image
- Python 3.10+ with `fiftyone` installed, for the plugin side

## Quickstart

1. **Serve the model:**

   ```bash
   ./scripts/serve.sh
   ```

   This copies `docker/.env.example` to `docker/.env` on first run, brings up
   `vllm/vllm-openai` via `docker compose`, and polls `/v1/models` until
   ready. First run downloads the model weights, so it can take a few
   minutes.

2. **Install the plugin** into your FiftyOne plugins directory:

   ```bash
   PLUGINS_DIR=$(python -c "import fiftyone as fo; print(fo.config.plugins_dir)")
   ln -s "$(pwd)" "$PLUGINS_DIR/gemma_vllm_metrics"
   ```

3. **Launch the FiftyOne App** and open the **Gemma vLLM Metrics** panel
   from the `+ New panel` menu.

4. **Point any OpenAI-compatible client at the endpoint:**

   ```
   Base URL:  http://localhost:8000/v1
   Model:     google/gemma-4-E4B-it-qat-w4a16-ct   (or your MODEL_ID)
   API key:   not required for local vLLM (any placeholder value)
   ```

## Configuration

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `MODEL_ID` | `docker/.env` | `google/gemma-4-E4B-it-qat-w4a16-ct` | HF repo id vLLM serves |
| `VLLM_IMAGE_TAG` | `docker/.env` | `gemma4` | `vllm/vllm-openai` image tag |
| `VLLM_PORT` | `docker/.env` | `8000` | Port for the OpenAI API + `/metrics` |
| `GEMMA_VLLM_BASE_URL` | FiftyOne plugin secret or env var | `http://localhost:8000/v1` | Where the plugin looks for the server |

Plugin secrets are declared in [`fiftyone.yml`](fiftyone.yml)'s `secrets:`
block and resolved via `ctx.secrets`, falling back to the environment
variable of the same name — set `GEMMA_VLLM_BASE_URL` if you're serving on a
different host/port.

## Development

```bash
# Python
pip install -r requirements.txt
pytest

# JS (hybrid panel frontend)
npm install
npm run build      # one-shot build -> dist/index.umd.js
npm run dev         # watch mode
```

Debugging the panel end-to-end: this repo was built and verified using the
`fiftyone-app-playwright` skill/MCP pattern — launch the App with
`remote=True`, drive it with Playwright, and watch for the "reload after
`reload_dataset`" pitfall documented there if you extend this plugin with
dataset-mutating operators.

## License

Apache 2.0 — see [LICENSE](LICENSE). Matches the license of Gemma 4 and vLLM.
