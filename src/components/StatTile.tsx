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
import IconSlot from "./IconSlot";

interface Props {
  label: string;
  value: string;
  icon: FC<IconProps>;
  valueColor?: TextColor;
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
