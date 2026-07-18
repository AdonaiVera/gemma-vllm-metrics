import time

import httpx

from .config import metrics_url, models_url

# Metrics we surface in the panel, mapped to the vLLM Prometheus series that
# feed them. Counters/gauges are read as their latest sample value; histograms
# are summarized as (count, sum) so the caller can derive an average.
COUNTER_METRICS = {
    "prompt_tokens_total": "vllm:prompt_tokens_total",
    "generation_tokens_total": "vllm:generation_tokens_total",
}
GAUGE_METRICS = {
    "num_requests_running": "vllm:num_requests_running",
    "num_requests_waiting": "vllm:num_requests_waiting",
    "kv_cache_usage_perc": "vllm:kv_cache_usage_perc",
}
HISTOGRAM_METRICS = {
    "time_to_first_token_seconds": "vllm:time_to_first_token_seconds",
    "inter_token_latency_seconds": "vllm:inter_token_latency_seconds",
    "e2e_request_latency_seconds": "vllm:e2e_request_latency_seconds",
}


def parse_prometheus_text(text):
    """Parses Prometheus exposition-format text into {metric_name: [(labels_str, value)]}.

    Deliberately minimal: just enough to read vLLM's own metric set, not a
    general-purpose Prometheus client.
    """
    series = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            name_and_labels, value_str = line.rsplit(" ", 1)
            value = float(value_str)
        except ValueError:
            continue

        if "{" in name_and_labels:
            name, labels = name_and_labels.split("{", 1)
            labels = labels.rstrip("}")
        else:
            name, labels = name_and_labels, ""

        series.setdefault(name, []).append((labels, value))

    return series


def _sum_series(series, name):
    return sum(value for _, value in series.get(name, []))


def _latest_series(series, name):
    values = series.get(name, [])
    return values[-1][1] if values else None


def _histogram_stats(series, base_name):
    """Derives (count, avg_seconds) from a Prometheus histogram's _sum/_count."""
    count = _sum_series(series, f"{base_name}_count")
    total = _sum_series(series, f"{base_name}_sum")
    avg = (total / count) if count else None
    return {"count": count, "avg_seconds": avg}


def fetch_raw_metrics(base_url, timeout=5.0):
    """GETs vLLM's /metrics endpoint. Raises httpx exceptions on failure."""
    resp = httpx.get(metrics_url(base_url), timeout=timeout)
    resp.raise_for_status()
    return resp.text


def summarize_metrics_text(text):
    """Pure function: Prometheus exposition text -> flat summary dict.

    Split out from get_metrics_snapshot so it's testable against a static
    fixture without mocking HTTP.
    """
    series = parse_prometheus_text(text)

    summary = {}
    for key, metric_name in COUNTER_METRICS.items():
        summary[key] = _sum_series(series, metric_name)
    for key, metric_name in GAUGE_METRICS.items():
        summary[key] = _latest_series(series, metric_name)
    for key, metric_name in HISTOGRAM_METRICS.items():
        summary[key] = _histogram_stats(series, metric_name)

    return summary


def get_metrics_snapshot(base_url, timeout=5.0):
    """Fetches and summarizes vLLM's current metrics into a flat dict.

    Returns a dict with `ok`, `fetched_at`, and either `error` or the
    summarized metric fields (counters, gauges, histogram count/avg pairs).
    """
    fetched_at = time.time()
    try:
        text = fetch_raw_metrics(base_url, timeout=timeout)
    except Exception as e:
        return {"ok": False, "fetched_at": fetched_at, "error": str(e)}

    snapshot = {"ok": True, "fetched_at": fetched_at}
    snapshot.update(summarize_metrics_text(text))
    return snapshot


def check_server_status(base_url, timeout=5.0):
    """Lightweight health check against vLLM's /v1/models endpoint."""
    try:
        resp = httpx.get(models_url(base_url), timeout=timeout)
        resp.raise_for_status()
        models = [m.get("id") for m in resp.json().get("data", [])]
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)}
