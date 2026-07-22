import { useCallback, useMemo } from "react";
import { usePanelEvent } from "@fiftyone/operators";

const PLUGIN = "@voxel51/gemma-vllm-metrics";
const PANEL = "gemma_vllm_metrics_panel";

export interface ServerStatus {
  ok: boolean;
  base_url?: string;
  models?: string[];
  error?: string;
}

export interface HistogramSummary {
  count: number | null;
  avg_seconds: number | null;
}

export interface MetricsSnapshot {
  ok: boolean;
  fetched_at?: number;
  error?: string;
  prompt_tokens_total?: number;
  generation_tokens_total?: number;
  num_requests_running?: number | null;
  num_requests_waiting?: number | null;
  kv_cache_usage_perc?: number | null;
  time_to_first_token_seconds?: HistogramSummary;
  inter_token_latency_seconds?: HistogramSummary;
  e2e_request_latency_seconds?: HistogramSummary;
}

export interface PluginClient {
  get_status: () => Promise<ServerStatus>;
  get_metrics: () => Promise<MetricsSnapshot>;
}

function callPanelMethod<T>(
  handleEvent: ReturnType<typeof usePanelEvent>,
  method: string
): Promise<T> {
  return new Promise((resolve, reject) => {
    handleEvent(method, {
      operator: `${PLUGIN}/${PANEL}#${method}`,
      params: {},
      callback: (result: any) => {
        if (result?.error) {
          reject(new Error(result.error));
        } else {
          resolve(result?.result ?? result);
        }
      },
    });
  });
}

export function usePluginClient(): PluginClient {
  const handleEvent = usePanelEvent();

  const get_status = useCallback(
    () => callPanelMethod<ServerStatus>(handleEvent, "get_status"),
    [handleEvent]
  );
  const get_metrics = useCallback(
    () => callPanelMethod<MetricsSnapshot>(handleEvent, "get_metrics"),
    [handleEvent]
  );

  // Memoized for consumers depending on referential stability.
  return useMemo(() => ({ get_status, get_metrics }), [get_status, get_metrics]);
}
