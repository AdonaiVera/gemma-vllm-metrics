import React from "react";
import { Card, Stack, Text, Orientation, Spacing, TextColor } from "@voxel51/voodo";

interface Props {
  label: string;
  value: string;
  valueColor?: TextColor;
}

export default function StatTile({ label, value, valueColor }: Props) {
  return (
    <Card compact>
      <Stack orientation={Orientation.Column} spacing={Spacing.Sm}>
        <Text color={TextColor.Secondary}>{label}</Text>
        <Text color={valueColor ?? TextColor.Fg}>{value}</Text>
      </Stack>
    </Card>
  );
}
