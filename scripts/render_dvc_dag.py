"""Render the current ``dvc dag`` terminal output as submission evidence."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def render_dag(output_path: Path) -> None:
    """Execute ``dvc dag`` and save its exact text in a terminal-style PNG."""
    completed = subprocess.run(  # noqa: S603
        ["dvc", "dag"],
        check=True,
        capture_output=True,
        text=True,
    )
    dag = completed.stdout.rstrip()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(12, 9.5), facecolor="#0f172a")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_facecolor("#0f172a")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.add_patch(
        plt.Rectangle((0, 0.925), 1, 0.075, color="#1e293b", transform=axis.transAxes)
    )
    for x, color in ((0.032, "#fb7185"), (0.058, "#fbbf24"), (0.084, "#4ade80")):
        axis.add_patch(plt.Circle((x, 0.962), 0.009, color=color, transform=axis.transAxes))
    axis.text(
        0.5,
        0.962,
        "MachinePulse - DVC pipeline",
        ha="center",
        va="center",
        color="#cbd5e1",
        fontsize=13,
        fontweight="bold",
    )
    axis.text(
        0.055,
        0.885,
        "(.venv) $ dvc dag",
        ha="left",
        va="top",
        color="#67e8f9",
        fontsize=15,
        family="monospace",
        fontweight="bold",
    )
    axis.text(
        0.055,
        0.82,
        dag,
        ha="left",
        va="top",
        color="#e2e8f0",
        fontsize=13,
        family="monospace",
        linespacing=1.2,
    )
    figure.savefig(output_path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    """Parse the destination and render the live DVC DAG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/submission/evidence/dvc-dag.png"),
    )
    args = parser.parse_args()
    render_dag(args.output)


if __name__ == "__main__":
    main()
