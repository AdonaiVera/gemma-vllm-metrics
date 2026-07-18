import React from "react";

interface Props {
  values: number[];
  width?: number;
  height?: number;
  stroke?: string;
}

// Minimal inline SVG sparkline. VOODO has no charting primitives, and pulling
// in a full charting library for a handful of points isn't worth the bundle
// weight, so this is intentionally dependency-free.
export default function Sparkline({ values, width = 160, height = 40, stroke = "currentColor" }: Props) {
  if (values.length < 2) {
    return <svg width={width} height={height} />;
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <polyline
        points={points.join(" ")}
        fill="none"
        stroke={stroke}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
