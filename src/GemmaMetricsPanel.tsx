import { useCallback, useEffect, useRef, useState } from "react";
import {
  Stack,
  Text,
  Heading,
  Button,
  Divider,
  Orientation,
  Spacing,
  Align,
  Justify,
  Size,
  Variant,
  HeadingLevel,
  TextColor,
  TextVariant,
  ActivityIcon,
  TimerIcon,
  DatabaseIcon,
  SpeedIcon,
  RefreshIcon,
} from "@voxel51/voodo";
import { usePluginClient, MetricsSnapshot, ServerStatus } from "./usePluginClient";
import StatusBadge from "./components/StatusBadge";
import StatTile from "./components/StatTile";
import IconSlot from "./components/IconSlot";
import MetricSparklineCard from "./components/MetricSparklineCard";

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
  const [lastPolledAt, setLastPolledAt] = useState<number | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const prevSnapshot = useRef<MetricsSnapshot | null>(null);

  // Ref avoids re-arming the poll interval when client() isn't referentially stable.
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
      setLastPolledAt(Date.now());
      setPollError(null);
    } catch (e) {
      setPollError(e instanceof Error ? e.message : String(e));
    }
  }, [client]);

  useEffect(() => {
    pollRef.current?.();
    const id = setInterval(() => pollRef.current?.(), POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  const totalRequests = metrics?.e2e_request_latency_seconds?.count ?? null;
  const avgE2eMs =
    metrics?.e2e_request_latency_seconds?.avg_seconds != null
      ? metrics.e2e_request_latency_seconds.avg_seconds * 1000
      : null;
  const kvCachePct =
    metrics?.kv_cache_usage_perc != null ? metrics.kv_cache_usage_perc * 100 : null;
  const queueDepth =
    (metrics?.num_requests_running ?? 0) + (metrics?.num_requests_waiting ?? 0);

  return (
    // Scopes VOODO's dark-mode CSS variables to just this panel.
    <Stack
      className="dark"
      orientation={Orientation.Column}
      spacing={Spacing.Lg}
      style={{ padding: 20 }}
    >
      <Stack align={Align.Center} justify={Justify.Between}>
        <Heading level={HeadingLevel.H2}>Gemma vLLM Metrics</Heading>
        <Stack orientation={Orientation.Row} align={Align.Center} spacing={Spacing.Md}>
          <StatusBadge status={status} modelId={status?.models?.[0]} />
          <Button
            size={Size.Sm}
            variant={Variant.Secondary}
            onClick={() => pollRef.current?.()}
          >
            <Stack orientation={Orientation.Row} align={Align.Center} spacing={Spacing.Xs}>
              <IconSlot icon={RefreshIcon} size={12} />
              <span>Refresh</span>
            </Stack>
          </Button>
        </Stack>
      </Stack>

      {lastPolledAt && (
        <Text variant={TextVariant.Caption} color={TextColor.Placeholder}>
          {`Last updated ${new Date(lastPolledAt).toLocaleTimeString()}`}
        </Text>
      )}

      {status && !status.ok && (
        <Text color={TextColor.Destructive}>
          {status.error || "Could not reach the vLLM server. Is it running? See README for scripts/serve.sh."}
        </Text>
      )}

      {pollError && (
        <Text color={TextColor.Destructive}>{`Refresh failed: ${pollError}`}</Text>
      )}

      <Divider label="Overview" />

      <Stack orientation={Orientation.Row} spacing={Spacing.Md} style={{ flexWrap: "wrap" }}>
        <StatTile label="Total Requests" value={fmt(totalRequests, 0)} icon={ActivityIcon} />
        <StatTile label="Queue Depth" value={fmt(queueDepth, 0)} icon={TimerIcon} />
        <StatTile
          label="GPU KV-Cache Usage"
          value={fmt(kvCachePct, 1, "%")}
          icon={DatabaseIcon}
          valueColor={kvCachePct && kvCachePct > 90 ? TextColor.Warning : undefined}
        />
        <StatTile label="Avg E2E Latency" value={fmt(avgE2eMs, 0, " ms")} icon={SpeedIcon} />
      </Stack>

      <Divider label="Live Trends" />

      <Stack orientation={Orientation.Row} spacing={Spacing.Md} style={{ flexWrap: "wrap" }}>
        <MetricSparklineCard
          label="Generation throughput (tok/s)"
          history={tokensPerSecHistory}
          digits={1}
          stroke="var(--color-semantic-info)"
          valueColor={TextColor.Info}
        />
        <MetricSparklineCard
          label="Avg time-to-first-token (ms)"
          history={ttftHistory}
          digits={0}
          stroke="var(--color-semantic-warning)"
          valueColor={TextColor.Warning}
        />
      </Stack>
    </Stack>
  );
}
