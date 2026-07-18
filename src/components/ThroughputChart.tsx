import React from "react";
import { Card, Stack, Text, Orientation, Spacing, TextColor } from "@voxel51/voodo";
import Sparkline from "./Sparkline";

interface Props {
  tokensPerSecHistory: number[];
}

export default function ThroughputChart({ tokensPerSecHistory }: Props) {
  const latest = tokensPerSecHistory[tokensPerSecHistory.length - 1];
  return (
    <Card compact>
      <Stack orientation={Orientation.Column} spacing={Spacing.Sm}>
        <Text color={TextColor.Secondary}>Generation Throughput (tok/s)</Text>
        <Text color={TextColor.Fg}>{latest !== undefined ? latest.toFixed(1) : "--"}</Text>
        <Sparkline values={tokensPerSecHistory} stroke="var(--fo-text-info, #4b9fd8)" />
      </Stack>
    </Card>
  );
}
