import { TextColor } from "@voxel51/voodo";
import type { FC } from "react";
import type { IconProps } from "@voxel51/voodo";

interface Props {
  icon: FC<IconProps>;
  size?: number;
}

// VOODO icons ignore their `size` prop; a fixed-size wrapper constrains them instead.
export default function IconSlot({ icon: Icon, size = 14 }: Props) {
  return (
    <div style={{ width: size, height: size, overflow: "hidden", flexShrink: 0 }}>
      <Icon color={TextColor.Secondary} />
    </div>
  );
}
