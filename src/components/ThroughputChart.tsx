import React from "react";
import { Card, Stack, Text, CardBackground, Shadow, Orientation, Spacing, TextVariant, TextColor } from "@voxel51/voodo";
import Sparkline from "./Sparkline";

interface Props {
  tokensPerSecHistory: number[];
}

export default function ThroughputChart({ tokensPerSecHistory }: Props) {
  const latest = tokensPerSecHistory[tokensPerSecHistory.length - 1];
  return (
    <Card background={CardBackground.Elevated} shadow={Shadow.Sm} style={{ flex: 1, minWidth: 240 }}>
      <Stack orientation={Orientation.Column} spacing={Spacing.Sm}>
        <Text variant={TextVariant.Label} color={TextColor.Secondary}>
          Generation throughput (tok/s)
        </Text>
        <Text variant={TextVariant.Xl} color={TextColor.Info}>
          {latest !== undefined ? latest.toFixed(1) : "--"}
        </Text>
        <Sparkline values={tokensPerSecHistory} stroke="var(--color-semantic-info)" />
      </Stack>
    </Card>
  );
}
