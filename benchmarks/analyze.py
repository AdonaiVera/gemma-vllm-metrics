"""Turns collected benchmark results into an analysis table and blog charts.

Reads results/summary.csv (from collect_results.py) and pricing.json, writes
results/analysis.csv and PNG charts. Cost per million output tokens assumes
the accelerator is billed for the whole run and is saturated at that
concurrency level:

    cost_per_1m = price_per_hour / (output_tokens_per_second * 3600) * 1e6

Example:
    python benchmarks/analyze.py --images-dir blogs/images
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Color follows the environment, not its rank, so adding one never repaints
# the others. Order and hex values come from the dataviz skill's validated
# categorical palette (light mode).
ENV_ORDER = ["local-gpu-fp8", "cloud-gpu-l4-fp8", "tpu-v6e-fp8", "cpu-llamacpp", "cloud-cpu-llamacpp"]
ENV_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
WORKLOAD_ORDER = ["short", "medium", "long"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"


def env_color(env, known):
    return ENV_COLORS[known.index(env) % len(ENV_COLORS)]


def aggregate(df):
    """Mean across trials per (env, workload, concurrency)."""
    agg = df.groupby(["env", "workload", "concurrency"], as_index=False).agg(
        trials=("trial", "count"),
        completed=("completed", "sum"),
        failed=("failed", "sum"),
        output_tok_s=("output_throughput", "mean"),
        output_tok_s_std=("output_throughput", "std"),
        request_s=("request_throughput", "mean"),
        ttft_p50_ms=("median_ttft_ms", "mean"),
        ttft_p95_ms=("p95_ttft_ms", "mean"),
        tpot_p50_ms=("median_tpot_ms", "mean"),
        tpot_p95_ms=("p95_tpot_ms", "mean"),
        e2el_p95_ms=("p95_e2el_ms", "mean"),
    )
    agg["workload"] = pd.Categorical(agg["workload"], WORKLOAD_ORDER, ordered=True)
    return agg.sort_values(["env", "workload", "concurrency"]).reset_index(drop=True)


def add_costs(agg, pricing, overrides):
    """Adds one cost_per_1m_<scenario> column per scenario with a price."""
    prices = {env: dict(spec.get("scenarios", {})) for env, spec in pricing["envs"].items()}
    for env, price in overrides.items():
        prices.setdefault(env, {})["custom"] = price

    present = set(agg["env"])
    scenarios = sorted({
        s for env, p in prices.items() if env in present
        for s, v in p.items() if v is not None
    })
    for scenario in scenarios:
        column = f"cost_per_1m_{scenario}"
        agg[column] = [
            prices.get(row.env, {}).get(scenario) / (row.output_tok_s * 3600) * 1e6
            if prices.get(row.env, {}).get(scenario) is not None and row.output_tok_s > 0
            else None
            for row in agg.itertuples()
        ]
    # One "primary" cost per environment: the billing mode that environment
    # actually used (on demand for cloud, electricity for owned hardware).
    preferred = ["on_demand", "electricity", "custom", "flex_start"]
    basis = {}
    for env, p in prices.items():
        for scenario in preferred:
            if p.get(scenario) is not None:
                basis[env] = (scenario, p[scenario])
                break
    agg["cost_per_1m_primary"] = [
        basis[row.env][1] / (row.output_tok_s * 3600) * 1e6
        if row.env in basis and row.output_tok_s > 0 else None
        for row in agg.itertuples()
    ]
    agg.attrs["cost_basis"] = {e: b for e, b in basis.items() if e in present}
    return scenarios


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def place_labels(fig, pending, gap_pt=12):
    """Direct end labels. At each x the highest line is labelled above its
    marker and the others below theirs, spaced so labels never overlap."""
    fig.canvas.draw()
    ppp = fig.dpi / 72  # pixels per point
    for ax in {p[0] for p in pending}:
        mine = [p for p in pending if p[0] is ax]
        for x in {p[1] for p in mine}:
            group = sorted((p for p in mine if p[1] == x),
                           key=lambda p: -ax.transData.transform((p[1], p[2]))[1])
            prev_px = None
            for i, (_, gx, gy, text, early) in enumerate(group):
                point_px = ax.transData.transform((gx, gy))[1]
                if i == 0 and not early:
                    off = 10
                else:
                    label_px = point_px - 16 * ppp
                    if prev_px is not None and prev_px - label_px < gap_pt * ppp:
                        label_px = prev_px - gap_pt * ppp
                    prev_px = label_px
                    off = (label_px - point_px) / ppp
                ax.annotate(text, (gx, gy), textcoords="offset points", xytext=(0, off),
                            ha="center", fontsize=9, color=INK)


def panel_chart(agg, column, title, ylabel, path, envs, log=False, fmt="{:,.0f}", legend_labels=None):
    """One panel per workload, one line per environment, shared y-axis."""
    workloads = [w for w in WORKLOAD_ORDER if w in set(agg["workload"].astype(str))]
    fig, axes = plt.subplots(1, len(workloads), figsize=(4 * len(workloads) + 1, 3.9),
                             sharey=True, facecolor=SURFACE)
    axes = [axes] if len(workloads) == 1 else list(axes)

    pending = []  # (ax, x, y, text, below) resolved after layout
    for ax, workload in zip(axes, workloads):
        style_axes(ax)
        sub = agg[agg["workload"].astype(str) == workload]
        for env in envs:
            line = sub[(sub["env"] == env)].dropna(subset=[column])
            if line.empty:
                continue
            color = env_color(env, ENV_ORDER)
            ax.plot(line["concurrency"], line[column], color=color, linewidth=2,
                    solid_capstyle="round", solid_joinstyle="round", zorder=2)
            ax.plot(line["concurrency"], line[column], linestyle="none", marker="o",
                    markersize=8, markerfacecolor=color, markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=3)
            last = line.iloc[-1]
            below = last["concurrency"] < sub["concurrency"].max()
            pending.append((ax, last["concurrency"], last[column],
                            fmt.format(last[column]), below))
        if log:
            ax.set_yscale("log")
        ax.set_xticks([1, 4, 8])
        ax.set_xlabel("Concurrent requests", color=INK_2, fontsize=9)
        ax.margins(x=0.12, y=0.18)
        sizes = agg[agg["workload"].astype(str) == workload]
        ax.set_title(workload.capitalize(), loc="left", color=INK, fontsize=11,
                     fontweight="semibold")
    axes[0].set_ylabel(ylabel, color=INK_2, fontsize=9)

    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13, fontweight="semibold")
    shown = [e for e in envs if not agg[(agg["env"] == e)].dropna(subset=[column]).empty]
    if len(shown) > 1 or legend_labels:
        handles = [plt.Line2D([0], [0], color=env_color(e, ENV_ORDER), linewidth=2,
                              marker="o", markersize=8, markerfacecolor=env_color(e, ENV_ORDER),
                              markeredgecolor=SURFACE, markeredgewidth=2,
                              label=(legend_labels or {}).get(e, e)) for e in shown]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.945),
                   frameon=False, ncol=len(shown), labelcolor=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    place_labels(fig, pending)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


TASK_LABELS = {
    "num_samples": "Count samples",
    "gt_person_count": "Count ground-truth persons",
    "top_pred_class": "Most common predicted class",
    "car_high_conf": "Count cars above 0.8 confidence",
    "most_unique": "Find most unique sample",
    "person_gap": "Compare two person counts",
}


def load_agent_runs(results_root):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_root, "*", "agent", "runs.jsonl"))):
        env = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path) as f:
            rows += [{**json.loads(line), "env": env} for line in f if line.strip()]
    return pd.DataFrame(rows)


def agent_chart(runs, envs, path):
    """Success rate per task, one bar per environment, labeled with passes/runs."""
    envs = [e for e in envs if e in set(runs["env"])]
    tasks = [t for t in TASK_LABELS if t in set(runs["task"])]
    n_env = len(envs)
    bar_h = min(0.24, 0.7 / n_env)

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(tasks) * max(n_env, 1) + 1.6), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1)

    for i, env in enumerate(envs):
        color = env_color(env, ENV_ORDER)
        for j, task in enumerate(tasks):
            sub = runs[(runs["env"] == env) & (runs["task"] == task)]
            passed, total = int(sub["correct"].sum()), len(sub)
            y = j + (i - (n_env - 1) / 2) * (bar_h + 0.04)
            ax.barh(y, passed / total, height=bar_h, color=color, zorder=2)
            ax.annotate(f"{passed}/{total}", (passed / total, y), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=9, color=INK)

    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([TASK_LABELS[t] for t in tasks], color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0%", "50%", "100%"])
    ax.set_xlabel("Repeats answered correctly", color=INK_2, fontsize=9)
    fig.suptitle("Agent tasks: success per task", x=0.01, ha="left", color=INK,
                 fontsize=13, fontweight="semibold")
    if n_env > 1:
        handles = [plt.Rectangle((0, 0), 1, 1, color=env_color(e, ENV_ORDER), label=e) for e in envs]
        fig.legend(handles=handles, loc="upper right", frameon=False, ncol=n_env,
                   labelcolor=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--summary", default=os.path.join(HERE, "results", "summary.csv"))
    parser.add_argument("--pricing", default=os.path.join(HERE, "pricing.json"))
    parser.add_argument("--out", default=os.path.join(HERE, "results", "analysis.csv"))
    parser.add_argument("--images-dir", default=os.path.join(HERE, "..", "blogs", "images"))
    parser.add_argument("--scenario", default="on_demand",
                        help="Cost scenario to chart (falls back to the first one available)")
    parser.add_argument("--price", action="append", default=[], metavar="ENV=USD_PER_HOUR",
                        help="Override or add an hourly price (scenario 'custom')")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    agg = aggregate(df)

    with open(args.pricing) as f:
        pricing = json.load(f)
    overrides = {k: float(v) for k, v in (item.split("=", 1) for item in args.price)}
    scenarios = add_costs(agg, pricing, overrides)
    agg.to_csv(args.out, index=False)
    print("wrote", args.out)

    os.makedirs(args.images_dir, exist_ok=True)
    envs = [e for e in ENV_ORDER if e in set(agg["env"])] + \
           sorted(set(agg["env"]) - set(ENV_ORDER))
    # The cloud CPU leg is a partial matrix on the L4 VM; it is reported in a table, not the charts.
    envs = [e for e in envs if e != "cloud-cpu-llamacpp"]
    ENV_COLORS_LEN = len(ENV_COLORS)
    if len(envs) > ENV_COLORS_LEN:
        print(f"warning: {len(envs)} environments but only {ENV_COLORS_LEN} validated colors")

    charts = [
        ("output_tok_s", "Output throughput", "Output tokens per second",
         "throughput_vs_concurrency.png", False, "{:,.0f}"),
        ("ttft_p95_ms", "Time to first token, p95 (log scale)", "Milliseconds",
         "ttft_p95_vs_concurrency.png", True, "{:,.0f}"),
        ("tpot_p50_ms", "Time per output token, median", "Milliseconds",
         "tpot_vs_concurrency.png", False, "{:.1f}"),
    ]
    for column, title, ylabel, name, log, fmt in charts:
        panel_chart(agg, column, title, ylabel, os.path.join(args.images_dir, name),
                    envs, log=log, fmt=fmt)

    runs = load_agent_runs(os.path.dirname(args.summary))
    if not runs.empty:
        agent_chart(runs, envs, os.path.join(args.images_dir, "agent_eval_by_task.png"))

    if scenarios:
        basis = agg.attrs.get("cost_basis", {})
        labels = {e: f"{e} ({b[0].replace('_', ' ')} ${b[1]:g}/h)" for e, b in basis.items()}
        panel_chart(agg, "cost_per_1m_primary",
                    "Cost per million output tokens (log scale, price basis in legend)",
                    "USD per 1M output tokens",
                    os.path.join(args.images_dir, "cost_per_million_tokens.png"),
                    envs, log=True, fmt="${:,.2f}", legend_labels=labels)
    else:
        print("no prices filled in pricing.json; skipping the cost chart")


if __name__ == "__main__":
    main()
