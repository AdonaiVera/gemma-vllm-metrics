import os

import pytest

from gemma_vllm_metrics.metrics_client import (
    parse_prometheus_text,
    summarize_metrics_text,
)

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "sample_metrics.txt"
)


@pytest.fixture
def sample_text():
    with open(FIXTURE_PATH) as f:
        return f.read()


def test_parse_prometheus_text_skips_comments(sample_text):
    series = parse_prometheus_text(sample_text)
    assert "vllm:prompt_tokens_total" in series
    # every parsed series has at least one (labels, value) pair
    for values in series.values():
        assert len(values) > 0


def test_parse_prometheus_text_reads_labels_and_value(sample_text):
    series = parse_prometheus_text(sample_text)
    labels, value = series["vllm:prompt_tokens_total"][0]
    assert value == 20.0
    assert 'model_name="google/gemma-4-E4B-it-qat-w4a16-ct"' in labels


def test_summarize_counters(sample_text):
    summary = summarize_metrics_text(sample_text)
    assert summary["prompt_tokens_total"] == 20.0
    assert summary["generation_tokens_total"] == 41.0


def test_summarize_gauges(sample_text):
    summary = summarize_metrics_text(sample_text)
    assert summary["num_requests_running"] == 0.0
    assert summary["num_requests_waiting"] == 0.0
    assert summary["kv_cache_usage_perc"] == 0.0


def test_summarize_histograms_derive_avg(sample_text):
    summary = summarize_metrics_text(sample_text)

    ttft = summary["time_to_first_token_seconds"]
    assert ttft["count"] == 1.0
    assert ttft["avg_seconds"] == pytest.approx(2.0726122856140137)

    e2e = summary["e2e_request_latency_seconds"]
    assert e2e["count"] == 1.0
    assert e2e["avg_seconds"] == pytest.approx(2.639143705368042)


def test_summarize_missing_metric_is_none_not_error():
    summary = summarize_metrics_text("# empty\n")
    assert summary["prompt_tokens_total"] == 0
    assert summary["num_requests_running"] is None
    assert summary["time_to_first_token_seconds"]["avg_seconds"] is None
