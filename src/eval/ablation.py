"""Compare multiple checkpoints on the same split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eval.evaluate import evaluate_checkpoint


def run_ablation(checkpoint_paths: dict[str, Path], test_dataset_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, path in checkpoint_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"checkpoint missing for {name}: {path}")
        model_class = "baseline" if "baseline" in name.lower() else "hybrid"
        rows.append({"name": name, **evaluate_checkpoint(path, test_dataset_path, model_class=model_class)})
    return pd.DataFrame(rows).set_index("name")


def plot_ablation_bar_chart(df: pd.DataFrame, out_path: Path) -> None:
    metrics = list(df.columns)
    n = len(metrics)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, metric in zip(axes_list, metrics, strict=False):
        vals = df[metric].astype(float)
        ax.bar(df.index.tolist(), vals.values)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.25, axis="y")

    for ax in axes_list[len(metrics) :]:
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablation eval across checkpoints.")
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out-md", type=Path, default=Path("reports") / "ablation.md")
    p.add_argument("--out-png", type=Path, default=Path("reports") / "plots" / "ablation.png")
    p.add_argument("--checkpoint", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ckpts = {name: Path(path) for name, path in args.checkpoint}
    df = run_ablation(ckpts, args.split)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    # Avoid optional dependency `tabulate` by formatting markdown ourselves.
    cols = df.columns.tolist()
    lines = ["## Ablation results", ""]
    lines.append("| run | " + " | ".join(f"`{c}`" for c in cols) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(cols)) + "|")
    for name, row in df.iterrows():
        vals = []
        for c in cols:
            v = float(row[c])
            vals.append("nan" if np.isnan(v) else f"{v:.4f}")
        lines.append("| " + str(name) + " | " + " | ".join(vals) + " |")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out_md.parent / "ablation.json").write_text(
        json.dumps(df.to_dict(orient="index"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    plot_ablation_bar_chart(df, args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

