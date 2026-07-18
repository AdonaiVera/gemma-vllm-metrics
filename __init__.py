"""Gemma vLLM Metrics: FiftyOne plugin for monitoring a locally-served Gemma 4 model."""

from .gemma_vllm_metrics.panel import GemmaMetricsPanel


def register(p):
    p.register(GemmaMetricsPanel)
