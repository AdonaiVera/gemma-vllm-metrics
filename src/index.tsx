import { registerComponent, PluginComponentType } from "@fiftyone/plugins";
import voodoCss from "@voxel51/voodo/theme.css?inline";
import GemmaMetricsPanel from "./GemmaMetricsPanel";

// The FiftyOne App only loads this bundle's JS (via `fiftyone.script` in
// package.json); it never fetches an accompanying CSS file. VOODO's
// components render Tailwind-style utility classes that need voodo.css's
// rules to mean anything, so we inject it as a <style> tag ourselves.
// `?inline` pulls the CSS in as a string at build time instead of Vite
// emitting it as a separate file nothing would load.
const STYLE_ID = "voxel51-gemma-vllm-metrics-voodo-css";
if (!document.getElementById(STYLE_ID)) {
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = voodoCss;
  document.head.appendChild(style);
}

// This registers the embedded React view referenced by the Python Panel's
// render() schema (component: "GemmaMetricsPanel", composite_view: true),
// not a standalone panel. The Python `Panel` subclass + fiftyone.yml
// `panels:` entry is what creates the actual panel-tab entry; composite-view
// resolution looks this up specifically under PluginComponentType.Component.
registerComponent({
  name: "GemmaMetricsPanel",
  label: "Gemma vLLM Metrics",
  component: GemmaMetricsPanel,
  type: PluginComponentType.Component,
  activator: () => true,
});
