import fiftyone.operators as foo
import fiftyone.operators.types as types

from .config import resolve_base_url
from .metrics_client import check_server_status, get_metrics_snapshot


class CheckServerStatus(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="check_server_status",
            label="Check Gemma vLLM server status",
            description="Health-checks the vLLM OpenAI-compatible server.",
            unlisted=True,
        )

    def execute(self, ctx):
        base_url = resolve_base_url(ctx)
        result = check_server_status(base_url)
        result["base_url"] = base_url
        return result


class GetVLLMMetrics(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="get_vllm_metrics",
            label="Get vLLM consumption metrics",
            description="Fetches and summarizes vLLM's /metrics for the panel.",
            unlisted=True,
        )

    def execute(self, ctx):
        base_url = resolve_base_url(ctx)
        return get_metrics_snapshot(base_url)


def register(p):
    p.register(CheckServerStatus)
    p.register(GetVLLMMetrics)
