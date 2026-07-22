import { registerComponent, PluginComponentType } from "@fiftyone/plugins";
import voodoCss from "@voxel51/voodo/theme.css?inline";
import GemmaMetricsPanel from "./GemmaMetricsPanel";

// FiftyOne only loads this bundle's JS, so VOODO's CSS is injected manually.
const STYLE_ID = "voxel51-gemma-vllm-metrics-voodo-css";
if (!document.getElementById(STYLE_ID)) {
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = voodoCss;
  document.head.appendChild(style);
}

// Registers the embedded view used by the Python Panel's composite_view schema.
registerComponent({
  name: "GemmaMetricsPanel",
  label: "Gemma vLLM Metrics",
  component: GemmaMetricsPanel,
  type: PluginComponentType.Component,
  activator: () => true,
});
