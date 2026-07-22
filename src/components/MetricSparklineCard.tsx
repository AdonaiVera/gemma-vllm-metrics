import { Card, Stack, Text, CardBackground, Shadow, Orientation, Spacing, TextVariant, TextColor } from "@voxel51/voodo";
import Sparkline from "./Sparkline";

interface Props {
  label: string;
  history: number[];
  digits: number;
  stroke: string;
  valueColor: TextColor;
}

export default function MetricSparklineCard({ label, history, digits, stroke, valueColor }: Props) {
  const latest = history[history.length - 1];
  return (
    <Card background={CardBackground.Elevated} shadow={Shadow.Sm} style={{ flex: 1, minWidth: 240 }}>
      <Stack orientation={Orientation.Column} spacing={Spacing.Sm}>
        <Text variant={TextVariant.Label} color={TextColor.Secondary}>
          {label}
        </Text>
        <Text variant={TextVariant.Xl} color={valueColor}>
          {latest !== undefined ? latest.toFixed(digits) : "--"}
        </Text>
        <Sparkline values={history} stroke={stroke} />
      </Stack>
    </Card>
  );
}
