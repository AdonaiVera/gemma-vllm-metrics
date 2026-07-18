import React from "react";
import { Stack, TextBadge, Orientation, Align, Spacing, TextColor } from "@voxel51/voodo";
import type { ServerStatus } from "../usePluginClient";

interface Props {
  status: ServerStatus | null;
  modelId?: string | null;
}

// A plain, self-sized <span> dot rather than a VOODO icon or Pill's
// `isStatus` bullet: both render unconstrained in the installed VOODO
// version and blow up to fill the panel (see StatTile.tsx for the same
// note). A styled span gives the same visual with zero sizing risk.
function Dot({ color }: { color: string }) {
  return (
    <span
      style={{
        display: "inline-block",
        width: 8,
        height: 8,
        borderRadius: "50%",
        backgroundColor: color,
        flexShrink: 0,
      }}
    />
  );
}

export default function StatusBadge({ status, modelId }: Props) {
  if (!status) {
    return (
      <Stack orientation={Orientation.Row} align={Align.Center} spacing={Spacing.Sm}>
        <Dot color="var(--color-content-text-secondary, #888)" />
        <TextBadge color={TextColor.Secondary}>Checking...</TextBadge>
      </Stack>
    );
  }

  if (!status.ok) {
    return (
      <Stack orientation={Orientation.Row} align={Align.Center} spacing={Spacing.Sm}>
        <Dot color="var(--color-semantic-destructive)" />
        <TextBadge color={TextColor.Destructive}>Server unreachable</TextBadge>
      </Stack>
    );
  }

  const label = modelId || status.models?.[0] || "model";
  return (
    <Stack orientation={Orientation.Row} align={Align.Center} spacing={Spacing.Sm}>
      <Dot color="var(--color-semantic-success)" />
      <TextBadge color={TextColor.Success}>{`Serving ${label}`}</TextBadge>
    </Stack>
  );
}
