"""Checks that a served model can do read-only FiftyOne agent tasks.

Runs a plain OpenAI tool-calling loop against any OpenAI-compatible endpoint
with a few read-only tools over a small FiftyOne dataset. Expected answers are
computed from the dataset itself, so grading never relies on hardcoded numbers.

This is a proxy for the Voxel51 Agent, not the agent app itself: it needs no
MCP integration, and it exercises the same server-side pieces that matter
(tool-call parsing, multi-turn context, argument fidelity, latency per task).

Example:
    python benchmarks/agent_eval.py --env local-gpu-fp8 \
        --base-url http://localhost:8000/v1 --repeats 3
"""
import argparse
import json
import os
import re
import time

import fiftyone as fo
from fiftyone import ViewField as F
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = (
    "You answer questions about a FiftyOne dataset. Always use the provided "
    "tools to look up facts; never guess. When you have the answer, reply "
    "with a single short sentence that states it."
)
DETECTION_FIELDS = ["ground_truth", "predictions"]
MAX_TURNS = 8


# -- Tools -------------------------------------------------------------------


class Tools:
    """Read-only tools over one dataset. Also used to compute expected answers."""

    def __init__(self, dataset):
        self.ds = dataset

    def get_dataset_info(self):
        return {
            "name": self.ds.name,
            "num_samples": len(self.ds),
            "detection_fields": DETECTION_FIELDS,
            "other_fields": ["uniqueness", "clip_predictions"],
        }

    def count_labels(self, field):
        if field == "clip_predictions":
            counts = self.ds.count_values("clip_predictions.label")
        elif field in DETECTION_FIELDS:
            counts = self.ds.count_values(f"{field}.detections.label")
        else:
            return {"error": f"unknown field {field!r}"}
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def count_detections(self, field, label=None, min_confidence=None):
        if field not in DETECTION_FIELDS:
            return {"error": f"field must be one of {DETECTION_FIELDS}"}
        expr = None
        if label is not None:
            expr = F("label") == label
        if min_confidence is not None:
            cond = F("confidence") > min_confidence
            expr = cond if expr is None else (expr & cond)
        view = self.ds if expr is None else self.ds.filter_labels(field, expr)
        return {"count": view.count(f"{field}.detections")}

    def top_samples(self, field="uniqueness", n=1):
        if field != "uniqueness":
            return {"error": "only 'uniqueness' is supported"}
        view = self.ds.sort_by(field, reverse=True).limit(int(n))
        return [
            {"filename": os.path.basename(s.filepath), field: round(s[field], 4)}
            for s in view
        ]


TOOL_SPECS = [
    {
        "name": "get_dataset_info",
        "description": "Dataset name, number of samples, and available label fields.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "count_labels",
        "description": "Number of labels per class, sorted most frequent first.",
        "parameters": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": DETECTION_FIELDS + ["clip_predictions"],
                }
            },
            "required": ["field"],
        },
    },
    {
        "name": "count_detections",
        "description": "Count detections in a field, optionally filtered by class and minimum confidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": DETECTION_FIELDS},
                "label": {"type": "string"},
                "min_confidence": {"type": "number"},
            },
            "required": ["field"],
        },
    },
    {
        "name": "top_samples",
        "description": "Samples with the highest values of a numeric field.",
        "parameters": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": ["uniqueness"]},
                "n": {"type": "integer"},
            },
            "required": [],
        },
    },
]
OPENAI_TOOLS = [{"type": "function", "function": spec} for spec in TOOL_SPECS]
SPEC_BY_NAME = {spec["name"]: spec for spec in TOOL_SPECS}
JSON_TYPES = {"string": str, "integer": int, "number": (int, float)}


def call_is_valid(name, arguments):
    """True if the call names a real tool and its arguments fit the schema."""
    spec = SPEC_BY_NAME.get(name)
    if spec is None:
        return False
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return False
    if not isinstance(args, dict):
        return False
    props = spec["parameters"]["properties"]
    if any(key not in props for key in args):
        return False
    if any(key not in args for key in spec["parameters"]["required"]):
        return False
    for key, value in args.items():
        if not isinstance(value, JSON_TYPES[props[key]["type"]]) or isinstance(value, bool):
            return False
        if "enum" in props[key] and value not in props[key]["enum"]:
            return False
    return True


# -- Tasks -------------------------------------------------------------------


def build_tasks(tools):
    """Each task is (id, question, expected token). Expected comes from the data."""
    gt_person = tools.count_detections("ground_truth", "person")["count"]
    pred_person = tools.count_detections("predictions", "person", 0.5)["count"]
    pred_labels = tools.count_labels("predictions")
    top_non_person = next(label for label in pred_labels if label != "person")
    return [
        ("num_samples", "How many samples are in the dataset?", str(len(tools.ds))),
        (
            "gt_person_count",
            "How many ground truth detections are labeled 'person'?",
            str(gt_person),
        ),
        (
            "top_pred_class",
            "Which class has the most predicted detections, ignoring 'person'?",
            top_non_person,
        ),
        (
            "car_high_conf",
            "How many predicted 'car' detections have confidence above 0.8?",
            str(tools.count_detections("predictions", "car", 0.8)["count"]),
        ),
        (
            "most_unique",
            "What is the filename of the most unique sample?",
            os.path.basename(tools.ds.sort_by("uniqueness", reverse=True).first().filepath),
        ),
        (
            "person_gap",
            "How many more 'person' detections with confidence above 0.5 are in "
            "predictions than in ground truth? Give the absolute difference.",
            str(abs(pred_person - gt_person)),
        ),
    ]


def is_correct(answer, expected):
    if not answer:
        return False
    answer = answer.lower()
    if expected.isdigit():
        return re.search(rf"(?<![\d.]){expected}(?![\d.])", answer) is not None
    return expected.lower() in answer


# -- Loop --------------------------------------------------------------------


def run_task(client, model, tools, question, max_tokens, thinking):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    stats = {"turns": 0, "tool_calls": 0, "valid_tool_calls": 0,
             "prompt_tokens": 0, "completion_tokens": 0, "error": None,
             "answer": None, "hit_max_turns": False, "trace": []}
    start = time.perf_counter()

    for _ in range(MAX_TURNS):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=OPENAI_TOOLS,
                tool_choice="auto", temperature=0.0, max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
            )
        except Exception as e:
            stats["error"] = f"{type(e).__name__}: {e}"
            break

        stats["turns"] += 1
        stats["prompt_tokens"] += resp.usage.prompt_tokens
        stats["completion_tokens"] += resp.usage.completion_tokens
        msg = resp.choices[0].message

        if not msg.tool_calls:
            stats["answer"] = msg.content
            break

        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            stats["tool_calls"] += 1
            name, arguments = call.function.name, call.function.arguments
            stats["trace"].append({"name": name, "arguments": arguments})
            if call_is_valid(name, arguments):
                stats["valid_tool_calls"] += 1
                result = getattr(tools, name)(**json.loads(arguments or "{}"))
            else:
                result = {"error": "invalid tool call"}
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result)})
    else:
        stats["hit_max_turns"] = True

    stats["seconds"] = time.perf_counter() - start
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--env", required=True, help="Environment label, e.g. local-gpu-fp8")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--dataset", default="tpu_blog_demo")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--thinking", choices=["off", "on"], default="off",
                        help="Gemma 4 thinking mode, set per request so every engine runs the same way")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Per-turn limit (default 1024 thinking off, 4096 on)")
    parser.add_argument("--results-root", default=os.path.join(HERE, "results"))
    args = parser.parse_args()
    thinking = args.thinking == "on"
    max_tokens = args.max_tokens or (4096 if thinking else 1024)

    tools = Tools(fo.load_dataset(args.dataset))
    tasks = build_tasks(tools)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=300)

    out_dir = os.path.join(args.results_root, args.env,
                           "agent_thinking" if thinking else "agent")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "runs.jsonl")

    rows = []
    with open(out_path, "w") as f:
        for task_id, question, expected in tasks:
            for repeat in range(1, args.repeats + 1):
                stats = run_task(client, args.model, tools, question, max_tokens, thinking)
                stats.update(task=task_id, repeat=repeat, expected=expected,
                             correct=is_correct(stats["answer"], expected))
                rows.append(stats)
                f.write(json.dumps(stats) + "\n")
                print(f"{task_id:16} #{repeat} correct={stats['correct']!s:5} "
                      f"turns={stats['turns']} calls={stats['valid_tool_calls']}/"
                      f"{stats['tool_calls']} {stats['seconds']:.1f}s")

    total = len(rows)
    calls = sum(r["tool_calls"] for r in rows)
    valid = sum(r["valid_tool_calls"] for r in rows)
    summary = {
        "env": args.env,
        "model": args.model,
        "thinking": args.thinking,
        "runs": total,
        "success_rate": sum(r["correct"] for r in rows) / total,
        "valid_tool_call_rate": (valid / calls) if calls else None,
        "avg_turns": sum(r["turns"] for r in rows) / total,
        "avg_seconds": sum(r["seconds"] for r in rows) / total,
        "avg_total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / total,
        "errors": sum(r["error"] is not None for r in rows),
        "hit_max_turns": sum(r["hit_max_turns"] for r in rows),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
