import React from "react";
import { Card, Stack, Text, CardBackground, Shadow, Orientation, Spacing, TextVariant, TextColor } from "@voxel51/voodo";
import Sparkline from "./Sparkline";

interface Props {
  ttftHistoryMs: number[];
}

export default function LatencyChart({ ttftHistoryMs }: Props) {
  const latest = ttftHistoryMs[ttftHistoryMs.length - 1];
  return (
    <Card background={CardBackground.Elevated} shadow={Shadow.Sm} style={{ flex: 1, minWidth: 240 }}>
      <Stack orientation={Orientation.Column} spacing={Spacing.Sm}>
        <Text variant={TextVariant.Label} color={TextColor.Secondary}>
          Avg time-to-first-token (ms)
        </Text>
        <Text variant={TextVariant.Xl} color={TextColor.Warning}>
          {latest !== undefined ? latest.toFixed(0) : "--"}
        </Text>
        <Sparkline values={ttftHistoryMs} stroke="var(--color-semantic-warning)" />
      </Stack>
    </Card>
  );
}
