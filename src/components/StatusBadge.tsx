import React from "react";
import { Pill, SemanticColor, StatusColor } from "@voxel51/voodo";
import type { ServerStatus } from "../usePluginClient";

interface Props {
  status: ServerStatus | null;
  modelId?: string | null;
}

// Neither a custom `icon` nor the built-in `isStatus` bullet render with a
// bounded size in the installed VOODO version; both blow up to fill the
// panel. backgroundColor alone still conveys state, so we skip both.
export default function StatusBadge({ status, modelId }: Props) {
  if (!status) {
    return <Pill backgroundColor={StatusColor.Default}>Checking...</Pill>;
  }

  if (!status.ok) {
    return <Pill backgroundColor={SemanticColor.Destructive}>Server unreachable</Pill>;
  }

  const label = modelId || status.models?.[0] || "model";
  return <Pill backgroundColor={SemanticColor.Success}>{`Serving ${label}`}</Pill>;
}
