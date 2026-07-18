import React from "react";
import { Card, Stack, Text, Orientation, Spacing, TextColor } from "@voxel51/voodo";
import Sparkline from "./Sparkline";

interface Props {
  ttftHistoryMs: number[];
}

export default function LatencyChart({ ttftHistoryMs }: Props) {
  const latest = ttftHistoryMs[ttftHistoryMs.length - 1];
  return (
    <Card compact>
      <Stack orientation={Orientation.Column} spacing={Spacing.Sm}>
        <Text color={TextColor.Secondary}>Avg Time-to-First-Token (ms)</Text>
        <Text color={TextColor.Fg}>{latest !== undefined ? latest.toFixed(0) : "--"}</Text>
        <Sparkline values={ttftHistoryMs} stroke="var(--fo-text-warning, #d8974b)" />
      </Stack>
    </Card>
  );
}
