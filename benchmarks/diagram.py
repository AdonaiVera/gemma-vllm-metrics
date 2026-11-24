"""Draws the four-environments overview diagram used in the blog.

Example:
    python benchmarks/diagram.py --out blogs/images/environments_overview.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
LINE = "#c9c8c2"
CARD = "#f3f2ee"

# Same environment colors as analyze.py's charts.
ENVIRONMENTS = [
    ("Local GPU", "RTX 4070 Ti 12GB", "vLLM", "#2a78d6"),
    ("Cloud GPU", "Google Cloud GPU VM", "vLLM", "#eb6834"),
    ("Cloud TPU", "TPU v6e", "vLLM-TorchTPU", "#1baf7a"),
    ("CPU", "CPU VM", "llama.cpp (GGUF)", "#eda100"),
]


def box(ax, x, y, w, h, fill=CARD, edge=LINE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                linewidth=1, edgecolor=edge, facecolor=fill, zorder=1))


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 color=INK_2, linewidth=1.4, zorder=2,
                                 shrinkA=0, shrinkB=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(HERE, "..", "blogs", "images",
                                                      "environments_overview.png"))
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=SURFACE)
    ax.set_xlim(0, 11.3)
    ax.set_ylim(-0.5, 5.6)
    ax.axis("off")

    # Left: the one client every environment is measured with.
    box(ax, 0.2, 1.3, 2.9, 3.0)
    ax.text(0.45, 3.95, "One shared client", fontsize=11, fontweight="semibold", color=INK)
    for i, line in enumerate(["vllm bench serve", "short / medium / long", "concurrency 1, 4, 8",
                              "agent-task check", "(tool calling)"]):
        ax.text(0.45, 3.45 - i * 0.42, line, fontsize=9.5, color=INK_2)

    # Middle: four environments, same OpenAI-compatible API.
    top = 4.55
    for i, (name, hardware, engine, color) in enumerate(ENVIRONMENTS):
        y = top - i * 1.2
        box(ax, 4.4, y - 0.45, 3.6, 0.95)
        ax.add_patch(plt.Circle((4.7, y + 0.02), 0.12, color=color, zorder=3))
        ax.text(4.95, y + 0.18, name, fontsize=10.5, fontweight="semibold", color=INK, va="center")
        ax.text(4.95, y - 0.17, f"{hardware}  -  {engine}", fontsize=9, color=INK_2, va="center")
        arrow(ax, (3.1, 2.8), (4.4, y + 0.02))

    ax.text(6.2, 5.3, "Each exposes an OpenAI-compatible /v1 API and /metrics", fontsize=9, color=INK_2, ha="center")

    # Right: what the server exposes and the same model everywhere.
    box(ax, 8.6, 1.3, 2.5, 3.0)
    ax.text(8.8, 3.95, "Same model", fontsize=11, fontweight="semibold", color=INK)
    for i, line in enumerate(["gemma-4-E2B-it", "online fp8 weights", "same sampling", "",
                              "CPU uses a GGUF:", "labeled, not controlled"]):
        ax.text(8.8, 3.5 - i * 0.38, line, fontsize=9.5, color=INK_2)
    for i in range(4):
        arrow(ax, (8.0, top - i * 1.2 + 0.02), (8.6, 2.8))

    ax.text(0.2, -0.25, "Recorded for every run: engine and runtime versions, hardware, precision, "
            "KV-cache dtype, context length.", fontsize=9, color=INK_2)

    fig.savefig(args.out, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
