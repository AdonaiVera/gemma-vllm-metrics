from fiftyone.operators.panel import Panel, PanelConfig

from .config import resolve_base_url
from .metrics_client import check_server_status, get_metrics_snapshot


class GemmaMetricsPanel(Panel):
    @property
    def config(self):
        return PanelConfig(
            name="gemma_vllm_metrics_panel",
            label="Gemma vLLM Metrics",
            icon="monitor_heart",
            surfaces="grid",
        )

    def on_load(self, ctx):
        ctx.panel.set_data("base_url", resolve_base_url(ctx))

    def render(self, ctx):
        import fiftyone.operators.types as types

        return types.Property(
            types.Object(),
            view=types.View(
                component="GemmaMetricsPanel",
                composite_view=True,
                get_status=self.get_status,
                get_metrics=self.get_metrics,
            ),
        )

    def get_status(self, ctx):
        base_url = resolve_base_url(ctx)
        result = check_server_status(base_url)
        result["base_url"] = base_url
        return result

    def get_metrics(self, ctx):
        base_url = resolve_base_url(ctx)
        return get_metrics_snapshot(base_url)
