import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Stack,
  Text,
  Heading,
  Orientation,
  Spacing,
  Align,
  Justify,
  HeadingLevel,
  TextColor,
} from "@voxel51/voodo";
import { usePluginClient, MetricsSnapshot, ServerStatus } from "./usePluginClient";
import StatusBadge from "./components/StatusBadge";
import StatTile from "./components/StatTile";
import ThroughputChart from "./components/ThroughputChart";
import LatencyChart from "./components/LatencyChart";

const POLL_INTERVAL_MS = 5000;
const HISTORY_LENGTH = 30;

function fmt(value: number | null | undefined, digits = 1, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `${value.toFixed(digits)}${suffix}`;
}

export default function GemmaMetricsPanel() {
  const client = usePluginClient();

  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [tokensPerSecHistory, setTokensPerSecHistory] = useState<number[]>([]);
  const [ttftHistory, setTtftHistory] = useState<number[]>([]);
  const prevSnapshot = useRef<MetricsSnapshot | null>(null);

  // `usePluginClient()` (and the underlying usePanelEvent() it wraps) may
  // not return a referentially stable value across renders. Depending on it
  // directly in this effect would re-arm the interval -- or worse, refire
  // immediately -- on every render, flooding the backend with requests. A
  // ref sidesteps that: the interval is set up once on mount and always
  // calls whatever the latest poll function is.
  const pollRef = useRef<() => Promise<void>>();

  pollRef.current = useCallback(async () => {
    try {
      const [statusResult, metricsResult] = await Promise.all([
        client.get_status(),
        client.get_metrics(),
      ]);
      setStatus(statusResult);

      if (metricsResult?.ok) {
        const prev = prevSnapshot.current;
        if (
          prev?.ok &&
          prev.fetched_at &&
          metricsResult.fetched_at &&
          metricsResult.generation_tokens_total !== undefined &&
          prev.generation_tokens_total !== undefined
        ) {
          const dt = metricsResult.fetched_at - prev.fetched_at;
          const dTokens = metricsResult.generation_tokens_total - prev.generation_tokens_total;
          if (dt > 0) {
            const tokensPerSec = Math.max(0, dTokens / dt);
            setTokensPerSecHistory((h) => [...h, tokensPerSec].slice(-HISTORY_LENGTH));
          }
        }

        const ttftMs = (metricsResult.time_to_first_token_seconds?.avg_seconds ?? 0) * 1000;
        setTtftHistory((h) => [...h, ttftMs].slice(-HISTORY_LENGTH));

        prevSnapshot.current = metricsResult;
      }
      setMetrics(metricsResult);
    } catch (e) {
      console.error("[gemma-vllm-metrics] poll failed:", e);
    }
  }, [client]);

  useEffect(() => {
    pollRef.current?.();
    const id = setInterval(() => pollRef.current?.(), POLL_INTERVAL_MS);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const totalRequests = metrics?.e2e_request_latency_seconds?.count ?? null;
  const avgE2eMs =
    metrics?.e2e_request_latency_seconds?.avg_seconds != null
      ? metrics.e2e_request_latency_seconds.avg_seconds * 1000
      : null;
  const gpuCachePct =
    metrics?.gpu_cache_usage_perc != null ? metrics.gpu_cache_usage_perc * 100 : null;
  const queueDepth =
    (metrics?.num_requests_running ?? 0) + (metrics?.num_requests_waiting ?? 0);

  return (
    <Stack orientation={Orientation.Column} spacing={Spacing.Lg}>
      <Stack align={Align.Center} justify={Justify.Between}>
        <Heading level={HeadingLevel.H2}>Gemma vLLM Metrics</Heading>
        <StatusBadge status={status} modelId={status?.models?.[0]} />
      </Stack>

      {status && !status.ok && (
        <Text color={TextColor.Destructive}>
          {status.error || "Could not reach the vLLM server. Is it running? See README for scripts/serve.sh."}
        </Text>
      )}

      <Stack orientation={Orientation.Row} spacing={Spacing.Md}>
        <StatTile label="Total Requests" value={fmt(totalRequests, 0)} />
        <StatTile label="Queue Depth" value={fmt(queueDepth, 0)} />
        <StatTile
          label="GPU KV-Cache Usage"
          value={fmt(gpuCachePct, 1, "%")}
          valueColor={gpuCachePct && gpuCachePct > 90 ? TextColor.Warning : undefined}
        />
        <StatTile label="Avg E2E Latency" value={fmt(avgE2eMs, 0, " ms")} />
      </Stack>

      <Stack orientation={Orientation.Row} spacing={Spacing.Md}>
        <ThroughputChart tokensPerSecHistory={tokensPerSecHistory} />
        <LatencyChart ttftHistoryMs={ttftHistory} />
      </Stack>
    </Stack>
  );
}
