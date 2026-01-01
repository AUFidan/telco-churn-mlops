"""Visualization utilities for model comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_metrics_comparison_chart(
    results: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    primary_metric: str = "recall",
    output_dir: Path = Path("artifacts/plots"),
) -> Path:
    """Create a grouped bar chart comparing metrics across models."""
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_models)
    width = 0.15
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]

    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in model_names]
        bars = ax.bar(
            x + offsets[i],
            values,
            width,
            label=metric.capitalize(),
            color=colors[i],
            alpha=1.0 if metric == primary_metric else 0.7,
            edgecolor="black" if metric == primary_metric else "none",
            linewidth=2 if metric == primary_metric else 0,
        )

        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Model Comparison (Primary Metric: {primary_metric.upper()})", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in model_names])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "model_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Chart saved to {chart_path}")
    return chart_path


def create_metric_table(
    results: dict[str, dict[str, float]],
    primary_metric: str = "recall",
) -> str:
    """Create a formatted ASCII table for console output."""
    # Find best model
    best_model = max(results.keys(), key=lambda m: results[m][primary_metric])

    # Header
    header = f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC_AUC':>10}"
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for model_name, model_metrics in results.items():
        marker = " *" if model_name == best_model else "  "
        line = f"{model_name:<20}{marker} {model_metrics['accuracy']:>10.4f} {model_metrics['precision']:>10.4f} {model_metrics['recall']:>10.4f} {model_metrics['f1']:>10.4f} {model_metrics['roc_auc']:>10.4f}"
        lines.append(line)

    lines.append(separator)
    lines.append(f"* Best model based on {primary_metric}")

    return "\n".join(lines)
