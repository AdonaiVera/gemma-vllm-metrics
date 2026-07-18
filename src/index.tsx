import { registerComponent, PluginComponentType } from "@fiftyone/plugins";
import GemmaMetricsPanel from "./GemmaMetricsPanel";

// This registers the embedded React view referenced by the Python Panel's
// render() schema (component: "GemmaMetricsPanel", composite_view: true) --
// NOT a standalone panel. The Python `Panel` subclass + fiftyone.yml
// `panels:` entry is what creates the actual panel-tab entry; composite-view
// resolution looks this up specifically under PluginComponentType.Component.
registerComponent({
  name: "GemmaMetricsPanel",
  label: "Gemma vLLM Metrics",
  component: GemmaMetricsPanel,
  type: PluginComponentType.Component,
  activator: () => true,
});
