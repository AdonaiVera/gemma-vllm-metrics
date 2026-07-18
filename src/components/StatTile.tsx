import React from "react";
import {
  Card,
  Stack,
  Text,
  CardBackground,
  Shadow,
  Orientation,
  Spacing,
  Align,
  TextVariant,
  TextColor,
} from "@voxel51/voodo";
import type { FC } from "react";
import type { IconProps } from "@voxel51/voodo";

interface Props {
  label: string;
  value: string;
  icon: FC<IconProps>;
  valueColor?: TextColor;
}

// The `size` prop on VOODO icon components has no effect in the installed
// version (confirmed: renders at ~940px regardless of the prop). Icons
// default to filling their parent, so a small fixed-size wrapper with
// overflow hidden achieves the same visual result without depending on
// that broken prop.
function IconSlot({ icon: Icon }: { icon: FC<IconProps> }) {
  return (
    <div style={{ width: 14, height: 14, overflow: "hidden", flexShrink: 0 }}>
      <Icon color={TextColor.Secondary} />
    </div>
  );
}

export default function StatTile({ label, value, icon, valueColor }: Props) {
  return (
    <Card background={CardBackground.Elevated} shadow={Shadow.Sm} style={{ flex: 1, minWidth: 160 }}>
      <Stack orientation={Orientation.Column} spacing={Spacing.Sm}>
        <Stack orientation={Orientation.Row} align={Align.Center} spacing={Spacing.Xs}>
          <IconSlot icon={icon} />
          <Text variant={TextVariant.Label} color={TextColor.Secondary}>
            {label}
          </Text>
        </Stack>
        <Text variant={TextVariant.Xxl} color={valueColor ?? TextColor.Fg}>
          {value}
        </Text>
      </Stack>
    </Card>
  );
}
